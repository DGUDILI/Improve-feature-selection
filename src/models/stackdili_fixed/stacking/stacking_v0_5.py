import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from models.stackdili_fixed.stacking.base import BaseStacking


class StackingV05(BaseStacking):
    """OOF 스태킹 v0.5: LR/SVC(스케일링) + RF/XGB/LGBM + ExtraTrees 메타 모델."""

    SCALE_MODELS = {'LR', 'SVC'}

    def __init__(self, random_seed: int = 42, n_splits: int = 5):
        self.random_seed = random_seed
        self.n_splits    = n_splits

    def _base_models(self):
        return {
            'LR':   LogisticRegression(random_state=self.random_seed, max_iter=1000, C=0.1, solver="saga"),
            'SVC':  SVC(random_state=self.random_seed, probability=True, C=1.0, kernel='rbf'),
            'RF':   RandomForestClassifier(n_estimators=300, random_state=self.random_seed),
            'XGB':  XGBClassifier(n_estimators=300, random_state=self.random_seed, eval_metric="logloss", verbosity=0),
            'LGBM': LGBMClassifier(n_estimators=300, random_state=self.random_seed, verbose=-1),
        }

    @staticmethod
    def _print_metrics(name, y_true, y_pred, y_prob) -> float:
        acc  = accuracy_score(y_true, y_pred)
        auc  = roc_auc_score(y_true, y_prob)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, *_ = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  {name:<22} ACC={acc:.4f}  AUC={auc:.4f}  "
              f"F1={f1:.4f}  Prec={prec:.4f}  Sens={rec:.4f}  Spec={spec:.4f}")
        return auc

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        save_dir: str,
    ) -> None:
        np.random.seed(self.random_seed)

        X_tr = X_train.values
        X_te = X_test.values

        # LR/SVC용 스케일링
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        with open(os.path.join(save_dir, "v05_scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)

        models = self._base_models()
        oof_train = np.zeros((len(y_train), len(models)))
        oof_test  = np.zeros((len(y_test),  len(models)))

        print(f"[1/2] {self.n_splits}-Fold OOF 기반 베이스 모델 학습")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)

        for col_idx, (name, model) in enumerate(models.items()):
            print(f"  -> {name} OOF 추출 중...")
            X_fold = X_tr_scaled if name in self.SCALE_MODELS else X_tr
            X_test_fold = X_te_scaled if name in self.SCALE_MODELS else X_te

            test_preds = np.zeros((len(y_test), self.n_splits))
            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_fold, y_train)):
                model.fit(X_fold[tr_idx], y_train[tr_idx])
                oof_train[val_idx, col_idx] = model.predict_proba(X_fold[val_idx])[:, 1]
                test_preds[:, fold_idx]     = model.predict_proba(X_test_fold)[:, 1]

            oof_test[:, col_idx] = test_preds.mean(axis=1)
            model.fit(X_fold, y_train)
            with open(os.path.join(save_dir, f"best_model_{name}_OOF.pkl"), 'wb') as f:
                pickle.dump(model, f)

        print("\n[2/2] 메타 모델 (ExtraTreesClassifier) 학습")
        meta_model = ExtraTreesClassifier(n_estimators=300, random_state=self.random_seed)
        meta_model.fit(oof_train, y_train)

        with open(os.path.join(save_dir, "best_model_Stacking_OOF.pkl"), 'wb') as f:
            pickle.dump(meta_model, f)

        np.save(os.path.join(save_dir, "v05_oof_test.npy"), oof_test)

        print("\n[학습 시뮬레이션 평가]")
        print("=" * 100)
        for col_idx, name in enumerate(models):
            y_prob = oof_test[:, col_idx]
            self._print_metrics(name, y_test, (y_prob >= 0.5).astype(int), y_prob)
        print("-" * 100)
        auc = self._print_metrics(
            "Stacking(ExtraTrees)", y_test,
            meta_model.predict(oof_test),
            meta_model.predict_proba(oof_test)[:, 1],
        )
        print("=" * 100)

        with open(os.path.join(save_dir, "result.txt"), "w") as f:
            f.write(str(auc))

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        save_dir: str,
    ) -> dict:
        oof_test = np.load(os.path.join(save_dir, "v05_oof_test.npy"))

        with open(os.path.join(save_dir, "best_model_Stacking_OOF.pkl"), 'rb') as f:
            meta_model = pickle.load(f)

        y_prob = meta_model.predict_proba(oof_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        print("\n[최종 성능 평가 - OOF Stacking v0.5]")
        print("=" * 70)
        auc = self._print_metrics("Stacking(ExtraTrees)", y_test, y_pred, y_prob)
        print("=" * 70)

        return {"auc": auc, "threshold": 0.5}
