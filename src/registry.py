from typing import Optional
from models.stackdili_fixed.ga.ga_v0 import GAv0
from models.stackdili_fixed.stacking.stacking_v0 import StackingV0
from models.stackdili_fixed.stacking.stacking_v1 import StackingV1
from models.stackdili_fixed.model import Model

GA_REGISTRY = {
    "v0": GAv0,  # 원본 StackDILI GA
    # "v1": GAv1,  # 새 GA 버전 추가 시 여기에 등록
}

STACKING_REGISTRY = {
    "v0": StackingV0,  # 원본 StackDILI (직접 예측 + ExtraTrees 메타)
    "v1": StackingV1,  # fixed (OOF + LogisticRegression 메타 + 피처 힌트)
}


def build_model(stacking_version: str, ga_version: Optional[str] = None) -> Model:
    ga       = GA_REGISTRY[ga_version]() if ga_version else None
    stacking = STACKING_REGISTRY[stacking_version]()
    return Model(stacking=stacking, ga=ga)
