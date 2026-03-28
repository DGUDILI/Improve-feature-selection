import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from xgboost import XGBClassifier

from models.stackdili_fixed.stacking.base import BaseStacking


class StackingV0(BaseStacking):
    """원본 StackDILI 스태킹 — 직접 예측 기반 + ExtraTrees 메타 모델 (v0).

    출처: https://github.com/GGCL7/StackDILI
    변경 없이 원본 로직 그대로 유지.

    v1(fixed)과의 차이:
    - OOF 없음: 베이스 모델이 학습 데이터 전체로 예측 (데이터 누수 있음)
    - 메타 모델: ExtraTreesClassifier (v1은 LogisticRegression)
    - 메타 모델 10회 반복, 최고 AUC 저장
    - 피처 힌트 없음
    """

    BASE_MODEL_ITERS    = 5
    STACKING_META_ITERS = 10

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

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

        base_model_classes = {
            'RF':      RandomForestClassifier,
            'ET':      ExtraTreesClassifier,
            'HistGB':  HistGradientBoostingClassifier,
            'XGBoost': XGBClassifier,
        }
        base_kwargs = {
            'RF':      {},
            'ET':      {},
            'HistGB':  {},
            'XGBoost': {"use_label_encoder": False, "eval_metric": "logloss", "verbosity": 0},
        }

        prob_train_list = []
        prob_test_list  = []

        # 1. 베이스 모델 학습 (5회 반복, 최고 AUC 저장)
        print(f"[1/2] 베이스 모델 학습 ({self.BASE_MODEL_ITERS}회 반복)")
        for name, ModelClass in base_model_classes.items():
            print(f"  -> {name} 학습 중...")
            best_auc   = -np.inf
            best_model = None

            for _ in range(self.BASE_MODEL_ITERS):
                model = ModelClass(
                    random_state=np.random.randint(0, 10000),
                    **base_kwargs[name],
                )
                model.fit(X_tr, y_train)
                auc = roc_auc_score(y_test, model.predict_proba(X_te)[:, 1])
                if auc > best_auc:
                    best_auc   = auc
                    best_model = model

            print(f"     최고 AUC={best_auc:.4f}")
            with open(os.path.join(save_dir, f"best_model_{name}.pkl"), 'wb') as f:
                pickle.dump(best_model, f)

            prob_train_list.append(best_model.predict_proba(X_tr)[:, 1])
            prob_test_list.append(best_model.predict_proba(X_te)[:, 1])

        X_meta_train = np.column_stack(prob_train_list)
        X_meta_test  = np.column_stack(prob_test_list)

        # 2. 메타 모델 학습 — ExtraTreesClassifier (10회 반복)
        print(f"\n[2/2] 메타 모델 학습 (ExtraTrees, {self.STACKING_META_ITERS}회 반복)")
        best_stacking_auc   = -np.inf
        best_stacking_model = None

        for i in range(self.STACKING_META_ITERS):
            meta_model = ExtraTreesClassifier(random_state=np.random.randint(0, 10000))
            meta_model.fit(X_meta_train, y_train)
            auc = roc_auc_score(y_test, meta_model.predict_proba(X_meta_test)[:, 1])
            print(f"  {i+1:2}회: AUC={auc:.4f}")
            if auc > best_stacking_auc:
                best_stacking_auc   = auc
                best_stacking_model = meta_model

        with open(os.path.join(save_dir, "best_model_stacking.pkl"), 'wb') as f:
            pickle.dump(best_stacking_model, f)

        # 결과 출력
        print("\n[학습 결과]")
        print("=" * 100)
        model_names = list(base_model_classes.keys())
        for name, y_prob in zip(model_names, prob_test_list):
            self._print_metrics(name, y_test, (y_prob >= 0.5).astype(int), y_prob)
        print("-" * 100)
        y_prob_stack = best_stacking_model.predict_proba(X_meta_test)[:, 1]
        auc = self._print_metrics(
            "Stacking(ET)", y_test, best_stacking_model.predict(X_meta_test), y_prob_stack,
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
        X_te = X_test.values

        base_model_names = ['RF', 'ET', 'HistGB', 'XGBoost']
        prob_list = []
        for name in base_model_names:
            with open(os.path.join(save_dir, f"best_model_{name}.pkl"), 'rb') as f:
                model = pickle.load(f)
            prob_list.append(model.predict_proba(X_te)[:, 1])

        X_meta_test = np.column_stack(prob_list)

        with open(os.path.join(save_dir, "best_model_stacking.pkl"), 'rb') as f:
            stacking_model = pickle.load(f)

        print("\n[최종 성능 평가]")
        print("=" * 100)
        print(f"  {'Model':<22} {'ACC':<8} {'AUC':<8} {'MCC':<8} {'F1':<8} {'Prec':<8} {'Sens':<8} {'Spec':<8}")
        print("-" * 100)

        for name, y_prob in zip(base_model_names, prob_list):
            self._print_metrics(name, y_test, (y_prob >= 0.5).astype(int), y_prob)

        print("-" * 100)
        y_prob = stacking_model.predict_proba(X_meta_test)[:, 1]
        auc = self._print_metrics(
            "Stacking(ET)", y_test, stacking_model.predict(X_meta_test), y_prob,
        )
        print("=" * 100)

        return {"auc": auc, "threshold": 0.5}
