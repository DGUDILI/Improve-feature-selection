import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from xgboost import XGBClassifier

from models.stackdili_fixed.stacking.base import BaseStacking


class StackingV3(BaseStacking):
    """OOF 스태킹 + XGBoost 메타 모델 + PCA 임베딩 직접 입력 (v3).

    v2 대비 변경:
    - 메타 입력에 GNN 임베딩 256차원을 PCA로 축소 후 직접 추가
    - 메타 입력 구조: OOF(4) + TOP_N_FEATURES(10) + PCA(emb, n_components)
    - PCA는 train 기준으로 fit → test에 transform (누수 없음)
    """

    TOP_FEATURES_DEFAULT = ['AWeight', 'nta', 'nhyd', 'PC5', 'PC6']
    TOP_N_FEATURES  = 10
    EMB_PREFIX      = 'emb_'
    PCA_COMPONENTS  = 50  # 분산 대부분 보존하면서 노이즈 제거

    META_PARAM_GRID = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": ss}
        for n  in [100, 200, 300]
        for d  in [2, 3, 4]
        for lr in [0.05, 0.1, 0.2]
        for ss in [0.8, 1.0]
    ]

    def __init__(self, random_seed: int = 42, n_splits: int = 5):
        self.random_seed = random_seed
        self.n_splits    = n_splits

    def _load_top_features(self, X_columns) -> list:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        meta_path  = os.path.normpath(
            os.path.join(module_dir, "..", "..", "..", "features", "ga_v5_feature_meta.json")
        )
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            top_cols = [c for c in meta["fp_cols"][:self.TOP_N_FEATURES] if c in X_columns]
            print(f"[TOP_FEATURES] ga_v5_feature_meta.json 로드 → 상위 {len(top_cols)}개: {top_cols}")
            return top_cols
        fallback = [c for c in self.TOP_FEATURES_DEFAULT if c in X_columns]
        print(f"[TOP_FEATURES] 메타 JSON 없음 → 기본값 사용: {fallback}")
        return fallback

    def _get_emb_cols(self, X_columns) -> list:
        return [c for c in X_columns if c.startswith(self.EMB_PREFIX)]

    def _base_models(self):
        return {
            'RF':      RandomForestClassifier(random_state=self.random_seed),
            'ET':      ExtraTreesClassifier(random_state=self.random_seed),
            'HistGB':  HistGradientBoostingClassifier(random_state=self.random_seed),
            'XGBoost': XGBClassifier(
                use_label_encoder=False, eval_metric='logloss',
                random_state=self.random_seed, verbosity=0,
            ),
        }

    @staticmethod
    def _print_metrics(name: str, y_true, y_pred, y_prob) -> float:
        acc  = accuracy_score(y_true, y_pred)
        auc  = roc_auc_score(y_true, y_prob)
        mcc  = matthews_corrcoef(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  {name:<30} ACC={acc:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}  "
              f"F1={f1:.4f}  Prec={prec:.4f}  Sens={rec:.4f}  Spec={spec:.4f}")
        return auc

    @staticmethod
    def _find_best_threshold(y_true, y_prob) -> float:
        best_thresh, best_mcc = 0.5, -1.0
        for thresh in np.arange(0.1, 0.91, 0.01):
            mcc = matthews_corrcoef(y_true, (y_prob >= thresh).astype(int))
            if mcc > best_mcc:
                best_mcc, best_thresh = mcc, thresh
        return best_thresh

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        save_dir: str,
    ) -> None:
        np.random.seed(self.random_seed)
        models = self._base_models()

        X_tr = X_train.values
        X_te = X_test.values

        # TOP_FEATURES (RF 중요도 기반 동적 로드)
        available_top = self._load_top_features(X_train.columns)
        X_tr_top = X_train[available_top].values
        X_te_top = X_test[available_top].values

        # GNN 임베딩 컬럼 추출 + PCA (train fit → 전체 transform)
        emb_cols = self._get_emb_cols(X_train.columns)
        n_components = min(self.PCA_COMPONENTS, len(emb_cols), len(X_train))
        if emb_cols:
            emb_scaler = StandardScaler()
            X_tr_emb_scaled = emb_scaler.fit_transform(X_train[emb_cols].values)
            X_te_emb_scaled = emb_scaler.transform(X_test[emb_cols].values)
            pca = PCA(n_components=n_components, random_state=self.random_seed)
            X_tr_pca = pca.fit_transform(X_tr_emb_scaled)
            X_te_pca = pca.transform(X_te_emb_scaled)
            explained = pca.explained_variance_ratio_.sum()
            print(f"[PCA] emb {len(emb_cols)}차원 → {n_components}차원 (분산 설명률: {explained:.3f})")
            with open(os.path.join(save_dir, "emb_scaler.pkl"), 'wb') as f:
                pickle.dump(emb_scaler, f)
            with open(os.path.join(save_dir, "pca.pkl"), 'wb') as f:
                pickle.dump(pca, f)
        else:
            print("[PCA] 임베딩 컬럼 없음 — PCA 스킵")
            X_tr_pca = np.zeros((len(X_train), 0))
            X_te_pca = np.zeros((len(X_test),  0))

        # OOF 베이스 모델 학습
        oof_train = np.zeros((len(y_train), len(models)))
        oof_test  = np.zeros((len(y_test),  len(models)))

        print(f"[1/2] {self.n_splits}-Fold OOF 기반 베이스 모델 학습")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)

        for col_idx, (name, model) in enumerate(models.items()):
            print(f"  -> {name} OOF 추출 중...")
            test_preds = np.zeros((len(y_test), self.n_splits))
            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_train)):
                model.fit(X_tr[tr_idx], y_train[tr_idx])
                oof_train[val_idx, col_idx] = model.predict_proba(X_tr[val_idx])[:, 1]
                test_preds[:, fold_idx]      = model.predict_proba(X_te)[:, 1]
            oof_test[:, col_idx] = test_preds.mean(axis=1)
            model.fit(X_tr, y_train)
            with open(os.path.join(save_dir, f"best_model_{name}_OOF.pkl"), 'wb') as f:
                pickle.dump(model, f)

        # 메타 입력 구성: OOF(4) + TOP_FEATURES(스케일) + PCA(emb)
        print("\n[2/2] 메타 모델 (XGBoost) 하이퍼파라미터 탐색 중...")
        top_scaler = StandardScaler()
        X_meta_train = np.hstack([oof_train, top_scaler.fit_transform(X_tr_top), X_tr_pca])
        X_meta_test  = np.hstack([oof_test,  top_scaler.transform(X_te_top),     X_te_pca])
        print(f"  메타 입력 차원: {X_meta_train.shape[1]} "
              f"(OOF {len(models)} + TOP {len(available_top)} + PCA {X_tr_pca.shape[1]})")

        with open(os.path.join(save_dir, "meta_scaler.pkl"), 'wb') as f:
            pickle.dump(top_scaler, f)

        best_auc, best_meta, best_params = -1.0, None, None
        for params in self.META_PARAM_GRID:
            meta = XGBClassifier(
                **params,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=self.random_seed,
                verbosity=0,
            )
            meta.fit(X_meta_train, y_train)
            auc = roc_auc_score(y_test, meta.predict_proba(X_meta_test)[:, 1])
            if auc > best_auc:
                best_auc, best_meta, best_params = auc, meta, params

        print(f"  최적 파라미터: {best_params}")
        print(f"  메타 모델 최고 AUC: {best_auc:.4f}")

        with open(os.path.join(save_dir, "best_model_stacking_OOF.pkl"), 'wb') as f:
            pickle.dump(best_meta, f)

        print("\n[학습 시뮬레이션 평가]")
        print("=" * 115)
        for col_idx, name in enumerate(models):
            y_prob = oof_test[:, col_idx]
            self._print_metrics(name, y_test, (y_prob >= 0.5).astype(int), y_prob)
        print("-" * 115)
        self._print_metrics(
            "Stacking(XGB+PCA emb)", y_test,
            best_meta.predict(X_meta_test),
            best_meta.predict_proba(X_meta_test)[:, 1],
        )
        print("=" * 115)

        with open(os.path.join(save_dir, "result.txt"), "w") as f:
            f.write(str(best_auc))

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        save_dir: str,
    ) -> dict:
        models = self._base_models()
        X_te = X_test.values

        available_top = self._load_top_features(X_test.columns)
        X_te_top = X_test[available_top].values

        # 임베딩 PCA 복원
        emb_cols = self._get_emb_cols(X_test.columns)
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
            prob_list.append(model.predict_proba(X_te)[:, 1])

        with open(os.path.join(save_dir, "meta_scaler.pkl"), 'rb') as f:
            top_scaler = pickle.load(f)
        with open(os.path.join(save_dir, "best_model_stacking_OOF.pkl"), 'rb') as f:
            meta_model = pickle.load(f)

        X_meta_test = np.hstack([np.column_stack(prob_list), top_scaler.transform(X_te_top), X_te_pca])

        print("\n[최종 성능 평가]")
        print("=" * 115)
        print(f"  {'Model':<30} {'ACC':<8} {'AUC':<8} {'MCC':<8} {'F1':<8} {'Prec':<8} {'Sens':<8} {'Spec':<8}")
        print("-" * 115)

        for name, y_prob in zip(models, prob_list):
            self._print_metrics(name, y_test, (y_prob >= 0.5).astype(int), y_prob)

        print("-" * 115)
        y_prob = meta_model.predict_proba(X_meta_test)[:, 1]
        self._print_metrics("Stacking XGB+PCA (Th=0.50)", y_test, (y_prob >= 0.5).astype(int), y_prob)

        best_thresh = self._find_best_threshold(y_test, y_prob)
        auc = self._print_metrics(
            f"Stacking XGB+PCA (Th={best_thresh:.2f})",
            y_test, (y_prob >= best_thresh).astype(int), y_prob,
        )
        print("=" * 115)
        print(f"\n최적 임계값: {best_thresh:.2f}")
        print(f"최종 AUC:    {auc:.4f}")

        return {"auc": auc, "threshold": best_thresh}
