import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from xgboost import XGBClassifier

from models.stackdili_fixed.ga.base import BaseGA


class GAv4(BaseGA):
    """XGBoost L1/L2 정규화 기반 피처 선택 (v4).

    트리 기반 모델에 Elastic Net 철학(L1+L2 패널티)을 주입합니다.
    - reg_alpha (L1): 리프 가중치 희소화 → 사용 피처 수 감소
    - reg_lambda (L2): 리프 가중치 축소 → 상관 피처 안정적 처리
    - scale_pos_weight: DILI 불균형 데이터 자동 보정

    CV로 최적 (reg_alpha, reg_lambda) 조합을 탐색한 뒤
    전체 데이터로 재학습하여 feature_importances_ > 0 인 피처를 선택합니다.
    선형 Elastic Net과 달리 비선형 관계를 포착합니다.
    """

    def __init__(
        self,
        reg_alphas:   list = [0.1, 1.0, 10.0],
        reg_lambdas:  list = [1.0, 10.0, 100.0],
        cv_folds:     int  = 5,
        n_estimators: int  = 300,
        min_features: int  = 10,
        random_seed:  int  = 42,
        n_jobs:       int  = -1,
    ):
        self.reg_alphas   = reg_alphas
        self.reg_lambdas  = reg_lambdas
        self.cv_folds     = cv_folds
        self.n_estimators = n_estimators
        self.min_features = min_features
        self.random_seed  = random_seed
        self.n_jobs       = n_jobs

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _make_xgb(self, reg_alpha, reg_lambda, scale_pos_weight):
        return XGBClassifier(
            n_estimators=self.n_estimators,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=self.random_seed,
            n_jobs=self.n_jobs,
            verbosity=0,
        )

    def _search_best_params(self, X_vals, y_vals, scale_pos_weight):
        """(reg_alpha, reg_lambda) 후보 × CV → 최적 조합 반환."""
        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
        )

        best_mcc    = -np.inf
        best_params = (self.reg_alphas[0], self.reg_lambdas[0])

        for alpha in self.reg_alphas:
            for lam in self.reg_lambdas:
                mcc_scores = []
                model = self._make_xgb(alpha, lam, scale_pos_weight)

                for tr_idx, val_idx in skf.split(X_vals, y_vals):
                    model.fit(X_vals[tr_idx], y_vals[tr_idx])
                    y_pred = model.predict(X_vals[val_idx])
                    mcc_scores.append(matthews_corrcoef(y_vals[val_idx], y_pred))

                mean_mcc = float(np.mean(mcc_scores))
                print(
                    f"  alpha={alpha:6.1f}, lambda={lam:7.1f}"
                    f" → MCC={mean_mcc:.4f}"
                )

                if mean_mcc > best_mcc:
                    best_mcc    = mean_mcc
                    best_params = (alpha, lam)

        return best_params, best_mcc

    # ------------------------------------------------------------------
    # BaseGA 인터페이스
    # ------------------------------------------------------------------

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        X_vals = X.values
        y_vals = y.values

        # 1. 불균형 보정 가중치
        n_neg = int((y_vals == 0).sum())
        n_pos = int((y_vals == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"[XGB-RegNet] scale_pos_weight={scale_pos_weight:.4f} (불균형 보정)")

        # 2. CV 기반 최적 (reg_alpha, reg_lambda) 탐색
        n_combos = len(self.reg_alphas) * len(self.reg_lambdas)
        print(
            f"[XGB-RegNet] (reg_alpha, reg_lambda) 탐색 중..."
            f" ({n_combos}조합 × {self.cv_folds}-fold CV)"
        )
        (best_alpha, best_lambda), best_mcc = self._search_best_params(
            X_vals, y_vals, scale_pos_weight
        )
        print(
            f"[XGB-RegNet] 최적: reg_alpha={best_alpha}, reg_lambda={best_lambda}"
            f" (MCC={best_mcc:.4f})"
        )

        # 3. 전체 데이터로 재학습
        print("[XGB-RegNet] 전체 데이터로 재학습 중...")
        final_model = self._make_xgb(best_alpha, best_lambda, scale_pos_weight)
        final_model.fit(X_vals, y_vals)

        # 4. feature_importances_ > 0 피처 추출 (importance 내림차순)
        importances = final_model.feature_importances_
        nonzero_mask = importances > 0
        n_nonzero = int(nonzero_mask.sum())
        print(
            f"[XGB-RegNet] feature_importances_ > 0: {n_nonzero}개 / {X.shape[1]}개 피처"
        )

        # importance 내림차순으로 정렬된 컬럼명
        sorted_idx = np.argsort(importances)[::-1]
        selected_idx = [i for i in sorted_idx if importances[i] > 0]

        # 5. min_features fallback
        if len(selected_idx) < self.min_features:
            print(
                f"[XGB-RegNet] 선택 피처 {len(selected_idx)}개 < "
                f"min_features({self.min_features}), "
                f"importance 상위 {self.min_features}개로 fallback"
            )
            selected_idx = sorted_idx[: self.min_features].tolist()

        selected_cols = X.columns[selected_idx].tolist()
        print(f"[XGB-RegNet] 최종 선택: {len(selected_cols)}개 피처")
        return selected_cols
