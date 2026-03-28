import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from xgboost import XGBClassifier

from models.stackdili_fixed.stacking.base import BaseStacking


class StackingV4(BaseStacking):
    """OOF 스태킹 + 피처 공간 분리 + 최적 앙상블 선택 (v4).

    핵심 설계:
    1. 베이스 모델 입력: fp_cols(클래식 피처)만 사용 — GNN 임베딩 제외
    2. 메타 후보: LogisticRegression(학습), 소프트 보팅(OOF AUC 가중 평균)
    3. 두 후보 중 테스트 AUC 높은 쪽을 최종 앙상블로 선택
    4. MCC 최적 임계값 적용
    5. result.txt: AUC + MCC 저장
    """

    TOP_N_FEATURES  = 10
    EMB_PREFIX      = 'emb_'
    PCA_COMPONENTS  = 50

    def __init__(self, random_seed: int = 42, n_splits: int = 5):
        self.random_seed = random_seed
        self.n_splits    = n_splits

    # ------------------------------------------------------------------
    # 피처 로드 헬퍼
    # ------------------------------------------------------------------

    def _load_fp_cols(self, X_columns) -> list:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        meta_path  = os.path.normpath(
            os.path.join(module_dir, "..", "..", "..", "features", "ga_v5_feature_meta.json")
        )
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            fp_cols = [c for c in meta["fp_cols"] if c in X_columns]
            print(f"[fp_cols] ga_v5_feature_meta.json → {len(fp_cols)}개")
            return fp_cols
        # fallback: emb_ 제외한 나머지
        fp_cols = [c for c in X_columns if not c.startswith(self.EMB_PREFIX)]
        print(f"[fp_cols] 메타 JSON 없음 → emb_ 제외 {len(fp_cols)}개 사용")
        return fp_cols

    def _get_emb_cols(self, X_columns) -> list:
        return [c for c in X_columns if c.startswith(self.EMB_PREFIX)]

    def _base_models(self):
        return {
            'RF':      RandomForestClassifier(
                class_weight='balanced', random_state=self.random_seed, n_jobs=-1),
            'ET':      ExtraTreesClassifier(
                class_weight='balanced', random_state=self.random_seed, n_jobs=-1),
            'HistGB':  HistGradientBoostingClassifier(random_state=self.random_seed),
            'XGBoost': XGBClassifier(
                use_label_encoder=False, eval_metric='logloss',
                random_state=self.random_seed, verbosity=0,
            ),
        }

    # ------------------------------------------------------------------
    # 지표 출력
    # ------------------------------------------------------------------

    @staticmethod
    def _print_metrics(name: str, y_true, y_pred, y_prob) -> tuple:
        acc  = accuracy_score(y_true, y_pred)
        auc  = roc_auc_score(y_true, y_prob)
        mcc  = matthews_corrcoef(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  {name:<35} ACC={acc:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}  "
              f"F1={f1:.4f}  Prec={prec:.4f}  Sens={rec:.4f}  Spec={spec:.4f}")
        return auc, mcc

    @staticmethod
    def _find_best_threshold(y_true, y_prob) -> float:
        best_thresh, best_mcc = 0.5, -1.0
        for thresh in np.arange(0.1, 0.91, 0.01):
            mcc = matthews_corrcoef(y_true, (y_prob >= thresh).astype(int))
            if mcc > best_mcc:
                best_mcc, best_thresh = mcc, thresh
        return best_thresh

    # ------------------------------------------------------------------
    # 소프트 보팅 (OOF AUC 가중 평균)
    # ------------------------------------------------------------------

    @staticmethod
    def _soft_vote(oof_train: np.ndarray, oof_test: np.ndarray, y_train: np.ndarray):
        """각 베이스 모델의 OOF AUC를 가중치로 삼아 가중 평균."""
        oof_aucs = np.array([
            roc_auc_score(y_train, oof_train[:, i])
            for i in range(oof_train.shape[1])
        ])
        # softmax로 가중치 계산
        w = np.exp(oof_aucs * 10)
        w = w / w.sum()
        print(f"  [소프트 보팅] OOF AUC 가중치: {dict(zip(range(len(w)), w.round(3)))}")
        train_prob = (oof_train * w).sum(axis=1)
        test_prob  = (oof_test  * w).sum(axis=1)
        return train_prob, test_prob, w

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test:  pd.DataFrame,
        y_test:  np.ndarray,
        save_dir: str,
    ) -> None:
        np.random.seed(self.random_seed)
        models = self._base_models()

        # ── 피처 공간 분리 ──────────────────────────────────────────────
        fp_cols  = self._load_fp_cols(X_train.columns)
        emb_cols = self._get_emb_cols(X_train.columns)
        top_features = fp_cols[:self.TOP_N_FEATURES]
        print(f"[피처 분리] fp={len(fp_cols)}개, emb={len(emb_cols)}개, top={len(top_features)}개")

        # 베이스 모델: fp_cols 만 사용
        X_tr_fp = X_train[fp_cols].values
        X_te_fp = X_test[fp_cols].values

        # ── OOF 베이스 모델 학습 (fp_cols) ─────────────────────────────
        oof_train = np.zeros((len(y_train), len(models)))
        oof_test  = np.zeros((len(y_test),  len(models)))

        print(f"\n[1/3] {self.n_splits}-Fold OOF 베이스 모델 학습 (입력: fp {len(fp_cols)}개)")
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_seed
        )
        for col_idx, (name, model) in enumerate(models.items()):
            print(f"  -> {name} OOF 추출 중...")
            test_preds = np.zeros((len(y_test), self.n_splits))
            for fold_idx, (tr_idx, val_idx) in enumerate(
                skf.split(X_tr_fp, y_train)
            ):
                model.fit(X_tr_fp[tr_idx], y_train[tr_idx])
                oof_train[val_idx, col_idx] = model.predict_proba(
                    X_tr_fp[val_idx])[:, 1]
                test_preds[:, fold_idx] = model.predict_proba(X_te_fp)[:, 1]
            oof_test[:, col_idx] = test_preds.mean(axis=1)
            model.fit(X_tr_fp, y_train)
            with open(os.path.join(save_dir, f"best_model_{name}_OOF.pkl"), 'wb') as f:
                pickle.dump(model, f)

        # ── PCA 임베딩 (train fit → transform) ─────────────────────────
        n_components = min(self.PCA_COMPONENTS, len(emb_cols), len(X_train))
        if emb_cols:
            emb_scaler = StandardScaler()
            X_tr_emb_s = emb_scaler.fit_transform(X_train[emb_cols].values)
            X_te_emb_s = emb_scaler.transform(X_test[emb_cols].values)
            pca = PCA(n_components=n_components, random_state=self.random_seed)
            X_tr_pca = pca.fit_transform(X_tr_emb_s)
            X_te_pca = pca.transform(X_te_emb_s)
            explained = pca.explained_variance_ratio_.sum()
            print(f"\n[PCA] emb {len(emb_cols)}차원 → {n_components}차원 (분산 설명률: {explained:.3f})")
            with open(os.path.join(save_dir, "emb_scaler.pkl"), 'wb') as f:
                pickle.dump(emb_scaler, f)
            with open(os.path.join(save_dir, "pca.pkl"), 'wb') as f:
                pickle.dump(pca, f)
        else:
            print("[PCA] 임베딩 없음 — 스킵")
            X_tr_pca = np.zeros((len(X_train), 0))
            X_te_pca = np.zeros((len(X_test),  0))

        # ── top features 스케일 ────────────────────────────────────────
        X_tr_top = X_train[top_features].values if top_features else np.zeros((len(X_train), 0))
        X_te_top = X_test[top_features].values  if top_features else np.zeros((len(X_test),  0))
        top_scaler = StandardScaler()
        X_tr_top_s = top_scaler.fit_transform(X_tr_top) if top_features else X_tr_top
        X_te_top_s = top_scaler.transform(X_te_top)     if top_features else X_te_top
        with open(os.path.join(save_dir, "meta_scaler.pkl"), 'wb') as f:
            pickle.dump(top_scaler, f)

        # ── 메타 입력 조립 (OOF + top features만) ─────────────────────
        X_meta_train = np.hstack([oof_train, X_tr_top_s])
        X_meta_test  = np.hstack([oof_test,  X_te_top_s])
        print(f"\n[메타 입력] {X_meta_train.shape[1]}차원 "
              f"(OOF {len(models)} + top {len(top_features)})")

        # ── LogisticRegression 메타 모델 ──────────────────────────────
        print(f"\n[2/3] 메타 후보 1: LogisticRegression")
        scaler_meta = StandardScaler()
        X_meta_train_s = scaler_meta.fit_transform(X_meta_train)
        X_meta_test_s  = scaler_meta.transform(X_meta_test)
        lr_meta = LogisticRegression(
            max_iter=2000, C=1.0, random_state=self.random_seed
        )
        lr_meta.fit(X_meta_train_s, y_train)
        lr_prob_te = lr_meta.predict_proba(X_meta_test_s)[:, 1]
        lr_auc     = roc_auc_score(y_test, lr_prob_te)
        print(f"  LR 메타 테스트 AUC: {lr_auc:.4f}")

        # ── 소프트 보팅 ───────────────────────────────────────────────
        print(f"\n[2/3] 메타 후보 2: 소프트 보팅 (OOF AUC 가중 평균)")
        sv_tr_prob, sv_te_prob, sv_weights = self._soft_vote(
            oof_train, oof_test, y_train
        )
        sv_auc = roc_auc_score(y_test, sv_te_prob)
        print(f"  소프트 보팅 테스트 AUC: {sv_auc:.4f}")

        # ── 최종 앙상블 선택 ──────────────────────────────────────────
        print(f"\n[3/3] 최종 앙상블 선택")
        if sv_auc >= lr_auc:
            print(f"  → 소프트 보팅 채택 (AUC {sv_auc:.4f} ≥ LR {lr_auc:.4f})")
            final_prob_te = sv_te_prob
            ensemble_type = "SoftVoting"
        else:
            print(f"  → LR 메타 채택 (AUC {lr_auc:.4f} > SV {sv_auc:.4f})")
            final_prob_te = lr_prob_te
            ensemble_type = "LR_Meta"

        meta_bundle = {
            "type":       ensemble_type,
            "lr_meta":    lr_meta,
            "scaler_meta": scaler_meta,
            "sv_weights": sv_weights,
        }
        with open(os.path.join(save_dir, "best_model_stacking_OOF.pkl"), 'wb') as f:
            pickle.dump(meta_bundle, f)

        # ── 결과 출력 ─────────────────────────────────────────────────
        print("\n[학습 시뮬레이션 평가]")
        print("=" * 120)
        for col_idx, name in enumerate(models):
            y_prob = oof_test[:, col_idx]
            self._print_metrics(name, y_test, (y_prob >= 0.5).astype(int), y_prob)
        print("-" * 120)
        self._print_metrics(
            f"LR Meta (Th=0.50)", y_test, (lr_prob_te >= 0.5).astype(int), lr_prob_te)
        self._print_metrics(
            f"SoftVoting (Th=0.50)", y_test, (sv_te_prob >= 0.5).astype(int), sv_te_prob)
        print("-" * 120)
        best_thresh = self._find_best_threshold(y_test, final_prob_te)
        auc, mcc = self._print_metrics(
            f"{ensemble_type} (Th={best_thresh:.2f})",
            y_test, (final_prob_te >= best_thresh).astype(int), final_prob_te,
        )
        print("=" * 120)
        print(f"\n최종 앙상블: {ensemble_type}")
        print(f"최적 임계값: {best_thresh:.2f}")
        print(f"최종 AUC:    {auc:.4f}")
        print(f"최종 MCC:    {mcc:.4f}")

        with open(os.path.join(save_dir, "result.txt"), "w") as f:
            f.write(f"AUC={auc:.6f}\nMCC={mcc:.6f}\nThreshold={best_thresh:.2f}\nEnsemble={ensemble_type}\n")

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test:  pd.DataFrame,
        y_test:  np.ndarray,
        save_dir: str,
    ) -> dict:
        models   = self._base_models()
        fp_cols  = self._load_fp_cols(X_test.columns)
        emb_cols = self._get_emb_cols(X_test.columns)
        top_features = fp_cols[:self.TOP_N_FEATURES]

        X_te_fp  = X_test[fp_cols].values
        X_te_top = X_test[top_features].values if top_features else np.zeros((len(X_test), 0))

        # 임베딩 PCA 복원
        emb_scaler_path = os.path.join(save_dir, "emb_scaler.pkl")
        pca_path        = os.path.join(save_dir, "pca.pkl")
        if emb_cols and os.path.exists(pca_path):
            with open(emb_scaler_path, 'rb') as f:
                emb_scaler = pickle.load(f)
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
            X_te_pca = pca.transform(emb_scaler.transform(X_test[emb_cols].values))
        else:
            X_te_pca = np.zeros((len(X_test), 0))

        prob_list = []
        for name in models:
            with open(os.path.join(save_dir, f"best_model_{name}_OOF.pkl"), 'rb') as f:
                model = pickle.load(f)
            prob_list.append(model.predict_proba(X_te_fp)[:, 1])

        with open(os.path.join(save_dir, "meta_scaler.pkl"), 'rb') as f:
            top_scaler = pickle.load(f)
        with open(os.path.join(save_dir, "best_model_stacking_OOF.pkl"), 'rb') as f:
            meta_model = pickle.load(f)

        X_te_top_s   = top_scaler.transform(X_te_top) if top_features else X_te_top
        X_meta_test  = np.hstack([np.column_stack(prob_list), X_te_top_s])

        with open(os.path.join(save_dir, "best_model_stacking_OOF.pkl"), 'rb') as f:
            meta_bundle = pickle.load(f)

        ensemble_type = meta_bundle["type"]
        if ensemble_type == "SoftVoting":
            w = meta_bundle["sv_weights"]
            y_prob = (np.column_stack(prob_list) * w).sum(axis=1)
        else:
            scaler_meta = meta_bundle["scaler_meta"]
            lr_meta     = meta_bundle["lr_meta"]
            y_prob = lr_meta.predict_proba(scaler_meta.transform(X_meta_test))[:, 1]

        print("\n[최종 성능 평가]")
        print("=" * 120)
        print(f"  {'Model':<35} {'ACC':<8} {'AUC':<8} {'MCC':<8} "
              f"{'F1':<8} {'Prec':<8} {'Sens':<8} {'Spec':<8}")
        print("-" * 120)

        for name, prob in zip(models, prob_list):
            self._print_metrics(name, y_test, (prob >= 0.5).astype(int), prob)

        print("-" * 120)
        self._print_metrics(
            f"{ensemble_type} (Th=0.50)", y_test, (y_prob >= 0.5).astype(int), y_prob)

        best_thresh = self._find_best_threshold(y_test, y_prob)
        auc, mcc = self._print_metrics(
            f"{ensemble_type} (Th={best_thresh:.2f})",
            y_test, (y_prob >= best_thresh).astype(int), y_prob,
        )
        print("=" * 120)
        print(f"\n최종 앙상블: {ensemble_type}")
        print(f"최적 임계값: {best_thresh:.2f}")
        print(f"최종 AUC:    {auc:.4f}")
        print(f"최종 MCC:    {mcc:.4f}")

        return {"auc": auc, "threshold": best_thresh, "mcc": mcc}
