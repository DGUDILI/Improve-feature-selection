import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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


class StackingV1(BaseStacking):
    """OOF 스태킹 + LogisticRegression 메타 모델 + 상위 피처 힌트 (v1)."""

    TOP_FEATURES = ['AWeight', 'nta', 'nhyd', 'PC5', 'PC6']

    def __init__(self, random_seed: int = 42, n_splits: int = 5):
        self.random_seed = random_seed
        self.n_splits    = n_splits

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
        print(f"  {name:<22} ACC={acc:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}  "
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

        available_top = [f for f in self.TOP_FEATURES if f in X_train.columns]
        X_tr_top = X_train[available_top].values
        X_te_top = X_test[available_top].values

        oof_train = np.zeros((len(y_train), len(models)))
        oof_test  = np.zeros((len(y_test),  len(models)))

        print(f"[1/2] {self.n_splits}-Fold OOF 기반 베이스 모델 학습")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)

        for col_idx, (name, model) in enumerate(models.items()):
            print(f"  -> {name} OOF 추출 중...")
            test_preds = np.zeros((len(y_test), self.n_splits))
            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_train)):
                model.fit(X_tr[tr_idx], y_train[tr_idx])
                oof_train[val_idx, col_idx]  = model.predict_proba(X_tr[val_idx])[:, 1]
                test_preds[:, fold_idx]       = model.predict_proba(X_te)[:, 1]
            oof_test[:, col_idx] = test_preds.mean(axis=1)
            model.fit(X_tr, y_train)
            with open(os.path.join(save_dir, f"best_model_{name}_OOF.pkl"), 'wb') as f:
                pickle.dump(model, f)

        print("\n[2/2] 메타 모델 (LogisticRegression + 피처 힌트) 학습")
        scaler = StandardScaler()
        X_meta_train = np.hstack([oof_train, scaler.fit_transform(X_tr_top)])
        X_meta_test  = np.hstack([oof_test,  scaler.transform(X_te_top)])

        meta_model = LogisticRegression(max_iter=1000, random_state=self.random_seed)
        meta_model.fit(X_meta_train, y_train)

        with open(os.path.join(save_dir, "meta_scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(save_dir, "best_model_stacking_OOF.pkl"), 'wb') as f:
            pickle.dump(meta_model, f)

        print("\n[학습 시뮬레이션 평가]")
        print("=" * 100)
        for col_idx, name in enumerate(models):
            y_prob = oof_test[:, col_idx]
            self._print_metrics(name, y_test, (y_prob >= 0.5).astype(int), y_prob)
        print("-" * 100)
        auc = self._print_metrics(
            "Stacking(LR+Feat)", y_test,
            meta_model.predict(X_meta_test),
            meta_model.predict_proba(X_meta_test)[:, 1],
        )
        print("=" * 100)

        weights = meta_model.coef_[0]
        meta_feature_names = list(models.keys()) + available_top
        print("\n[💡 메타 모델 가중치 (힌트 피처 포함)]")
        for name, weight in zip(meta_feature_names, weights):
            print(f"  - {name}: {weight:.4f}")

        with open(os.path.join(save_dir, "result.txt"), "w") as f:
            f.write(str(auc))

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        save_dir: str,
    ) -> dict:
        models = self._base_models()
        X_te = X_test.values

        available_top = [f for f in self.TOP_FEATURES if f in X_test.columns]
        X_te_top = X_test[available_top].values

        prob_list = []
        for name in models:
            with open(os.path.join(save_dir, f"best_model_{name}_OOF.pkl"), 'rb') as f:
                model = pickle.load(f)
            prob_list.append(model.predict_proba(X_te)[:, 1])

        with open(os.path.join(save_dir, "meta_scaler.pkl"), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(save_dir, "best_model_stacking_OOF.pkl"), 'rb') as f:
            meta_model = pickle.load(f)

        X_meta_test = np.hstack([np.column_stack(prob_list), scaler.transform(X_te_top)])

        print("\n[최종 성능 평가]")
        print("=" * 110)
        print(f"  {'Model':<22} {'ACC':<8} {'AUC':<8} {'MCC':<8} {'F1':<8} {'Prec':<8} {'Sens':<8} {'Spec':<8}")
        print("-" * 110)

        for name, y_prob in zip(models, prob_list):
            self._print_metrics(name, y_test, (y_prob >= 0.5).astype(int), y_prob)

        print("-" * 110)
        y_prob = meta_model.predict_proba(X_meta_test)[:, 1]
        self._print_metrics("Stacking (Th=0.50)", y_test, (y_prob >= 0.5).astype(int), y_prob)

        best_thresh = self._find_best_threshold(y_test, y_prob)
        auc = self._print_metrics(
            f"Stacking (Th={best_thresh:.2f})",
            y_test, (y_prob >= best_thresh).astype(int), y_prob,
        )
        print("=" * 110)
        print(f"\n최적 임계값: {best_thresh:.2f}")
        print(f"최종 AUC:    {auc:.4f}")

        return {"auc": auc, "threshold": best_thresh}
