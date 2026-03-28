import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from boruta import BorutaPy
from mrmr import mrmr_classif

from models.stackdili_fixed.ga.base import BaseGA


class GAv1(BaseGA):
    """MRMR + Boruta 앙상블 피처 선택 (v1).

    두 가지 상호 보완적 방법을 결합합니다.
    - MRMR: 타겟 관련성을 최대화하면서 피처 간 중복을 최소화
    - Boruta: Shadow Feature 대비 통계적으로 유의한 피처만 채택

    앙상블 모드:
    - 'intersection': 두 방법 모두 선택한 피처 (보수적)
    - 'union': 어느 한 방법이라도 선택한 피처 (포용적)
    교집합이 min_features보다 작으면 자동으로 합집합으로 fallback.
    """

    def __init__(
        self,
        n_mrmr_features: int = 128,
        boruta_max_iter: int = 100,
        boruta_perc: int = 75,
        ensemble_mode: str = "intersection",
        min_features: int = 10,
        var_threshold: float = 0.01,
        random_seed: int = 42,
        n_jobs: int = -1,
    ):
        self.n_mrmr_features = n_mrmr_features
        self.boruta_max_iter = boruta_max_iter
        self.boruta_perc     = boruta_perc
        self.ensemble_mode   = ensemble_mode
        self.min_features    = min_features
        self.var_threshold   = var_threshold
        self.random_seed     = random_seed
        self.n_jobs          = n_jobs

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _apply_variance_threshold(self, X: pd.DataFrame) -> pd.DataFrame:
        selector = VarianceThreshold(threshold=self.var_threshold)
        selector.fit(X)
        kept_cols = X.columns[selector.get_support()].tolist()
        return X[kept_cols]

    def _run_mrmr(self, X: pd.DataFrame, y: pd.Series) -> list:
        k = min(self.n_mrmr_features, X.shape[1])
        selected = mrmr_classif(X=X, y=y, K=k)
        return selected  # 중요도 내림차순 정렬된 컬럼명 리스트

    def _run_boruta(self, X: pd.DataFrame, y: pd.Series) -> list:
        rf = RandomForestClassifier(
            n_jobs=self.n_jobs,
            class_weight="balanced",
            random_state=self.random_seed,
        )
        boruta = BorutaPy(
            estimator=rf,
            n_estimators="auto",
            perc=self.boruta_perc,
            max_iter=self.boruta_max_iter,
            random_state=self.random_seed,
            verbose=1,
        )
        boruta.fit(X.values, y.values)
        selected = X.columns[boruta.support_].tolist()
        return selected

    def _ensemble(self, mrmr_cols: list, boruta_cols: list) -> list:
        mrmr_set   = set(mrmr_cols)
        boruta_set = set(boruta_cols)

        intersection = mrmr_set & boruta_set
        union        = mrmr_set | boruta_set

        if self.ensemble_mode == "union":
            result = union
        else:  # intersection (기본값)
            if len(intersection) >= self.min_features:
                result = intersection
            else:
                print(
                    f"[앙상블] 교집합 {len(intersection)}개 < min_features({self.min_features}), "
                    "합집합으로 fallback"
                )
                result = union

        # MRMR 중요도 순서 유지 (mrmr_cols 순서 기준, 나머지는 뒤에 추가)
        ordered = [c for c in mrmr_cols if c in result]
        ordered += [c for c in boruta_cols if c in result and c not in set(ordered)]
        return ordered

    # ------------------------------------------------------------------
    # BaseGA 인터페이스
    # ------------------------------------------------------------------

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        print(f"[VT] 분산 임계값({self.var_threshold}) 적용 중... (입력: {X.shape[1]}개)")
        X_filtered = self._apply_variance_threshold(X)
        print(f"[VT] 완료: {X_filtered.shape[1]}개 피처 유지")

        print(f"\n[MRMR] Top-{self.n_mrmr_features} 피처 선택 중...")
        mrmr_cols = self._run_mrmr(X_filtered, y)
        print(f"[MRMR] 완료: {len(mrmr_cols)}개 선택")

        print(f"\n[Boruta] Shadow Feature 검정 중 (max_iter={self.boruta_max_iter})...")
        boruta_cols = self._run_boruta(X_filtered, y)
        print(f"[Boruta] 완료: {len(boruta_cols)}개 선택")

        print(f"\n[앙상블] mode='{self.ensemble_mode}'")
        selected = self._ensemble(mrmr_cols, boruta_cols)
        print(f"[앙상블] 최종 선택: {len(selected)}개 피처")

        return selected
