from abc import ABC, abstractmethod
import pandas as pd


class BaseGA(ABC):
    @abstractmethod
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        """피처 선택 실행. 선택된 컬럼명 리스트를 반환."""
        pass
