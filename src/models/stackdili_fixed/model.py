import os
import shutil
import subprocess
import pandas as pd

from typing import Optional
from models.stackdili_fixed.ga.base import BaseGA
from models.stackdili_fixed.stacking.base import BaseStacking


class Model:
    """데이터 정제 → (선택) GA → Stacking 파이프라인 조립기."""

    def __init__(
        self,
        stacking: BaseStacking,
        ga: Optional[BaseGA] = None,
        stacking_version: str = "unknown",
        ga_version: Optional[str] = None,
    ):
        # src/models/stackdili_fixed/ 기준으로 프로젝트 루트 계산
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.ga               = ga
        self.stacking         = stacking
        self.stacking_version = stacking_version
        self.ga_version       = ga_version

    def _restore_features(self, features_path: str) -> None:
        """Feature_raw.csv → Feature.csv 복원. raw 파일이 없으면 지금 Feature.csv를 백업한다."""
        features_dir = os.path.dirname(features_path)
        raw_path = os.path.join(features_dir, "Feature_raw.csv")
        if os.path.exists(raw_path):
            shutil.copy2(raw_path, features_path)
            print("[원본 복원] Feature_raw.csv → Feature.csv")
        elif os.path.exists(features_path):
            shutil.copy2(features_path, raw_path)
            print("[최초 백업] Feature.csv → Feature_raw.csv (이후 실행부터 자동 복원)")

    def _build_save_dir(self, clean: bool) -> str:
        dir_name = f"stacking_{self.stacking_version}"
        if self.ga_version:
            dir_name += f"_ga_{self.ga_version}"
        if clean:
            dir_name += "_clean"
        return os.path.join(
            self.project_root, "src", "models", "stackdili_fixed", "Model", dir_name
        )

    def run(self, clean: bool = False):
        features_path = os.path.join(self.project_root, "src", "features", "Feature.csv")

        # 매 실행 시 원본 자동 복원
        self._restore_features(features_path)

        # 1. 데이터 정제 (Train-Test 중복 제거, 선택 사항)
        if clean:
            print("[1/3] 데이터 정제")
            clean_script = os.path.join(self.project_root, "src", "preprocessing", "make_clean_data.py")
            subprocess.run(["python", clean_script], check=True, text=True)
            # 정제 결과(Feature_cleaned.csv)를 Feature.csv에 덮어써서 이후 단계에서 사용
            cleaned_csv = os.path.join(self.project_root, "src", "features", "Feature_cleaned.csv")
            shutil.copy2(cleaned_csv, features_path)

        # 2. GA 피처 선택 (선택 사항)
        if self.ga is not None:
            print("[GA] 피처 선택 중...")
            raw = pd.read_csv(features_path)
            train_raw   = raw[raw['ref'] != 'DILIrank']
            X_train_raw = train_raw.drop(['SMILES', 'Label', 'ref'], axis=1)
            y_train_raw = train_raw['Label']
            selected_cols = self.ga.select_features(X_train_raw, y_train_raw)
            # GA가 Feature.csv를 직접 수정한 경우(예: GAv5 임베딩 추가)에만 재로드
            if not all(c in raw.columns for c in selected_cols):
                raw = pd.read_csv(features_path)
            raw[['SMILES', 'Label', 'ref'] + selected_cols].to_csv(features_path, index=False)

        # 3. 스태킹 학습 및 평가
        ga_label = f"GA {self.ga_version}" if self.ga_version else "GA 없음"
        print(f"[3/3] 스태킹 학습 및 평가  |  Stacking {self.stacking_version}  |  {ga_label}")
        cleaned = pd.read_csv(features_path)

        train = cleaned[cleaned['ref'] != 'DILIrank']
        test  = cleaned[cleaned['ref'] == 'DILIrank']

        X_train = train.drop(['SMILES', 'Label', 'ref'], axis=1)
        y_train = train['Label'].values
        X_test  = test.drop(['SMILES', 'Label', 'ref'], axis=1)
        y_test  = test['Label'].values

        save_dir = self._build_save_dir(clean)
        os.makedirs(save_dir, exist_ok=True)
        self.stacking.fit(X_train, y_train, X_test, y_test, save_dir)
        self.stacking.evaluate(X_test, y_test, save_dir)

    def predict(self, _):
        return None
