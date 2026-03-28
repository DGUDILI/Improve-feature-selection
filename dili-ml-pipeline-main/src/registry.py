from typing import Optional
from models.stackdili_fixed.ga.ga_v0 import GAv0
from models.stackdili_fixed.ga.ga_v1 import GAv1
from models.stackdili_fixed.ga.ga_v2 import GAv2
from models.stackdili_fixed.ga.ga_v3 import GAv3
from models.stackdili_fixed.ga.ga_v4 import GAv4
from models.stackdili_fixed.ga.ga_v5 import GAv5
from models.stackdili_fixed.stacking.stacking_v0 import StackingV0
from models.stackdili_fixed.stacking.stacking_v1 import StackingV1
from models.stackdili_fixed.model import Model

GA_REGISTRY = {
    "v0": GAv0,  # 원본 StackDILI GA — DEAP 기반 유전 알고리즘, 단일 목적 (Accuracy)
    "v1": GAv1,  # MRMR + Boruta 앙상블
    "v2": GAv2,  # NSGA-II — MCC + 피처 수 동시 최적화, 파레토 프론트
    "v3": GAv3,  # VT + MRMR — 빠른 중복 제거 기반 피처 선택
    "v4": GAv4,  # XGBoost L1/L2 — 비선형 Elastic Net 효과, CV로 최적 정규화 탐색
    "v5": GAv5,  # 이미지 아키텍처 — VT+RF Top-128(Path A) + GCN+CrossAttention 24-dim(Path B)
}

STACKING_REGISTRY = {
    "v0": StackingV0,  # 원본 StackDILI (직접 예측 + ExtraTrees 메타)
    "v1": StackingV1,  # fixed (OOF + LogisticRegression 메타 + 피처 힌트)
}


def build_model(stacking_version: str, ga_version: Optional[str] = None) -> Model:
    ga       = GA_REGISTRY[ga_version]() if ga_version else None
    stacking = STACKING_REGISTRY[stacking_version]()
    return Model(stacking=stacking, ga=ga)
