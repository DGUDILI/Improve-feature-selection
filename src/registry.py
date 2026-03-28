from typing import Optional
from models.stackdili_fixed.model import Model


def _load_ga(version: str):
    """요청된 GA 버전만 import."""
    if version == "v0":
        from models.stackdili_fixed.ga.ga_v0 import GAv0
        return GAv0
    if version == "v1":
        from models.stackdili_fixed.ga.ga_v1 import GAv1
        return GAv1
    if version == "v4":
        from models.stackdili_fixed.ga.ga_v4 import GAv4
        return GAv4
    if version == "v5":
        from models.stackdili_fixed.ga.ga_v5 import GAv5
        return GAv5
    raise KeyError(f"GA 버전 '{version}'이 존재하지 않습니다. 가능한 버전: {list(GA_REGISTRY)}")


def _load_stacking(version: str):
    """요청된 Stacking 버전만 import."""
    if version == "v0":
        from models.stackdili_fixed.stacking.stacking_v0 import StackingV0
        return StackingV0
    if version == "v0.5":
        from models.stackdili_fixed.stacking.stacking_v0_5 import StackingV05
        return StackingV05
    if version == "v1":
        from models.stackdili_fixed.stacking.stacking_v1 import StackingV1
        return StackingV1
    raise KeyError(f"Stacking 버전 '{version}'이 존재하지 않습니다. 가능한 버전: {list(STACKING_REGISTRY)}")


# train.py의 choices= 에 사용하기 위한 키 목록 (import 없이 반환)
GA_REGISTRY = {
    "v0": None,
    "v1": None,
    "v4": None,
    "v5": None,
}

STACKING_REGISTRY = {
    "v0":   None,
    "v0.5": None,
    "v1":   None,
}


def build_model(stacking_version: str, ga_version: Optional[str] = None) -> Model:
    ga       = _load_ga(ga_version)() if ga_version else None
    stacking = _load_stacking(stacking_version)()
    return Model(stacking=stacking, ga=ga, stacking_version=stacking_version, ga_version=ga_version)
