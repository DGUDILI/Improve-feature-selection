import os
import subprocess
import pandas as pd

from typing import Optional
from models.stackdili_fixed.ga.base import BaseGA
from models.stackdili_fixed.stacking.base import BaseStacking


class Model:
    """데이터 정제 → (선택) GA → Stacking 파이프라인 조립기."""

    def __init__(self, stacking: BaseStacking, ga: Optional[BaseGA] = None):
        # src/models/stackdili_fixed/ 기준으로 프로젝트 루트 계산
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.ga       = ga
        self.stacking = stacking
        self.save_dir = os.path.join(
            self.project_root, "src", "models", "stackdili_fixed", "Model"
        )

    def run(self):
        features_path = os.path.join(self.project_root, "src", "features", "Feature.csv")

        # 1. GA 피처 선택 (선택 사항)
        if self.ga is not None:
            print("[GA] 피처 선택 중...")
            raw = pd.read_csv(features_path)
            train_raw   = raw[raw['ref'] != 'DILIrank']
            X_train_raw = train_raw.drop(['SMILES', 'Label', 'ref'], axis=1)
            y_train_raw = train_raw['Label']
            selected_cols = self.ga.select_features(X_train_raw, y_train_raw)
            raw[['SMILES', 'Label', 'ref'] + selected_cols].to_csv(features_path, index=False)

        # 2. 데이터 정제 (Train-Test 중복 제거)
        print("[1/2] 데이터 정제")
        clean_script = os.path.join(self.project_root, "src", "preprocessing", "make_clean_data.py")
        subprocess.run(["python", clean_script], check=True, text=True)

        # 2. 스태킹 학습 및 평가
        print("[2/2] 스태킹 학습 및 평가")
        cleaned_path = os.path.join(self.project_root, "src", "features", "Feature_cleaned.csv")
        cleaned = pd.read_csv(cleaned_path)

        train = cleaned[cleaned['ref'] != 'DILIrank']
        test  = cleaned[cleaned['ref'] == 'DILIrank']

        X_train = train.drop(['SMILES', 'Label', 'ref'], axis=1)
        y_train = train['Label'].values
        X_test  = test.drop(['SMILES', 'Label', 'ref'], axis=1)
        y_test  = test['Label'].values

        os.makedirs(self.save_dir, exist_ok=True)
        self.stacking.fit(X_train, y_train, X_test, y_test, self.save_dir)
        result = self.stacking.evaluate(X_test, y_test, self.save_dir)

        result_path = os.path.join(self.save_dir, "result.txt")
        with open(result_path) as f:
            oof_auc = f.read()
        print(f"\nOOF AUC:  {oof_auc}")
        print(f"Eval AUC: {result['auc']}")

    def predict(self, _):
        return None
