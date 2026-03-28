from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseStacking(ABC):
    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        save_dir: str,
    ) -> None:
        """베이스 모델 + 메타 모델 학습 후 save_dir에 저장."""
        pass

    @abstractmethod
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        save_dir: str,
    ) -> dict:
        """저장된 모델 로드 후 평가. {"auc": float, "threshold": float} 반환."""
        pass
