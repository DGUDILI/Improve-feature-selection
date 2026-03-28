import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from mrmr import mrmr_classif

from models.stackdili_fixed.ga.base import BaseGA


class GAv3(BaseGA):
    """VT + MRMR 피처 선택 (v3).

    두 단계로 구성됩니다.
    - VarianceThreshold: 분산이 낮은 상수성 피처 제거
    - MRMR: 타겟 관련성 최대화 + 피처 간 중복 최소화로 Top-K 선택

    Boruta 없이 MRMR만 사용하므로 v1보다 빠르고,
    분자 지문의 중복 피처를 효과적으로 제거합니다.
    """

    def __init__(
        self,
        n_features: int = 128,
        var_threshold: float = 0.01,
        random_seed: int = 42,
    ):
        self.n_features    = n_features
        self.var_threshold = var_threshold
        self.random_seed   = random_seed

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        # 1. Variance Threshold
        print(f"[VT] 분산 임계값({self.var_threshold}) 적용 중... (입력: {X.shape[1]}개)")
        selector = VarianceThreshold(threshold=self.var_threshold)
        selector.fit(X)
        kept_cols = X.columns[selector.get_support()].tolist()
        X_filtered = X[kept_cols]
        print(f"[VT] 완료: {X_filtered.shape[1]}개 피처 유지")

        # 2. MRMR
        k = min(self.n_features, X_filtered.shape[1])
        print(f"\n[MRMR] Top-{k} 피처 선택 중...")
        selected = mrmr_classif(X=X_filtered, y=y, K=k)
        print(f"[MRMR] 완료: {len(selected)}개 선택")

        return selected
