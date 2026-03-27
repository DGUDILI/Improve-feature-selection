# 실행 가이드

---

## 목차

1. [환경 세팅 (Docker)](#환경-세팅)
2. [실행 방법](#실행-방법)
3. [현재 버전 목록](#현재-버전-목록)
4. [파이프라인 흐름](#파이프라인-흐름)
5. [새 버전 추가하기](#새-버전-추가하기)

---

## 환경 세팅

> **Windows / Mac 공통** — Docker만 설치되어 있으면 됩니다.

### 1. Docker Desktop 설치

- [Docker Desktop 다운로드](https://www.docker.com/products/docker-desktop/)

### 2. 이미지 빌드

```bash
./run.sh build
```

Windows PowerShell에서는:
```powershell
bash run.sh build
```

> 첫 빌드는 conda 환경 생성 때문에 5~10분 걸릴 수 있습니다.

### 3. 환경 확인

```bash
./run.sh env-test
```

---

## 실행 방법

### 기본 실행 (GA 없이 Stacking v1)

```bash
./run.sh run
```

### 버전 지정 실행

```bash
./run.sh run [stacking_version] [ga_version]

# 예시
./run.sh run v0        # 원본 StackDILI stacking (GA 없이)
./run.sh run v1        # fixed stacking (GA 없이)
./run.sh run v0 v0     # 원본 GA + 원본 stacking
./run.sh run v1 v0     # 원본 GA + fixed stacking
```

> GA는 선택 사항입니다. 생략하면 `Feature.csv`의 전체 피처로 바로 Stacking을 실행합니다.
> GA는 피처 수에 따라 수십 분 이상 걸릴 수 있습니다.

### 컨테이너 쉘 접속 (직접 실험할 때)

```bash
./run.sh shell

# 컨테이너 안에서
conda run -n dili_ml_pipeline_env python src/train.py --stacking v1 --ga v0
```

---

## 현재 버전 목록

### GA

| 버전 | 파일 | 설명 |
|------|------|------|
| `v0` | `ga/ga_v0.py` | 원본 StackDILI GA — DEAP 기반, RF 5-fold CV 피트니스 |

### Stacking

| 버전 | 파일 | 설명 |
|------|------|------|
| `v0` | `stacking/stacking_v0.py` | 원본 StackDILI — 직접 예측 기반, ExtraTrees 메타 모델 |
| `v1` | `stacking/stacking_v1.py` | fixed — OOF 기반, LogisticRegression 메타 모델 + 피처 힌트 |

**v0 vs v1 차이:**

| | v0 (원본) | v1 (fixed) |
|---|---|---|
| 예측 방식 | 직접 예측 (데이터 누수 있음) | OOF (누수 없음) |
| 메타 모델 | ExtraTreesClassifier | LogisticRegression |
| 반복 횟수 | 베이스 5회 + 메타 10회 | 1회 (결정론적) |
| 피처 힌트 | 없음 | TOP 5 피처 추가 |
| 임계값 최적화 | 없음 | 있음 (MCC 기준) |

---

## 파이프라인 흐름

```
[전처리팀] Feature.py
     ↓ src/features/Feature.csv 생성
     ↓
[GA팀, 선택] ga_v0.py → select_features()
     ↓ 선택된 피처로 Feature.csv 덮어쓰기
     ↓
[전처리팀] make_clean_data.py
     ↓ src/features/Feature_cleaned.csv 생성 (Train-Test 중복 제거)
     ↓
[스태킹팀] stacking_vN.py → fit() → evaluate()
     ↓ src/models/stackdili_fixed/Model/ 에 pkl 저장
     ↓
결과 출력 (ACC / AUC / MCC / F1 / Prec / Sens / Spec)
OOF AUC / Eval AUC
```

---

## 새 버전 추가하기

### GA 새 버전 추가 (예: ga_v1.py)

**1. 파일 생성** `src/models/stackdili_fixed/ga/ga_v1.py`

```python
from models.stackdili_fixed.ga.base import BaseGA
import pandas as pd

class GAv1(BaseGA):
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        # 새 피처 선택 로직 작성
        # 반드시 선택된 컬럼명 리스트를 반환해야 함
        selected_cols = [...]
        return selected_cols
```

**2. registry.py에 등록** `src/registry.py`

```python
from models.stackdili_fixed.ga.ga_v0 import GAv0
from models.stackdili_fixed.ga.ga_v1 import GAv1  # 추가

GA_REGISTRY = {
    "v0": GAv0,
    "v1": GAv1,  # 추가
}
```

**3. 실행**

```bash
./run.sh run v1 v1
```

---

### Stacking 새 버전 추가 (예: stacking_v2.py)

**1. 파일 생성** `src/models/stackdili_fixed/stacking/stacking_v2.py`

```python
from models.stackdili_fixed.stacking.base import BaseStacking
import pandas as pd
import numpy as np

class StackingV2(BaseStacking):
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_test: pd.DataFrame, y_test: np.ndarray,
            save_dir: str) -> None:
        # 학습 로직 작성
        # 모델은 save_dir에 pkl로 저장
        pass

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray,
                 save_dir: str) -> dict:
        # 평가 로직 작성
        # 반드시 {"auc": float, "threshold": float} 딕셔너리 반환
        return {"auc": ..., "threshold": ...}
```

**2. registry.py에 등록** `src/registry.py`

```python
from models.stackdili_fixed.stacking.stacking_v1 import StackingV1
from models.stackdili_fixed.stacking.stacking_v2 import StackingV2  # 추가

STACKING_REGISTRY = {
    "v0": StackingV0,
    "v1": StackingV1,
    "v2": StackingV2,  # 추가
}
```

**3. 실행**

```bash
./run.sh run v2
```

---

## 인터페이스 규칙

새 버전을 만들 때 반드시 지켜야 하는 규칙입니다.

### BaseGA (`src/models/stackdili_fixed/ga/base.py`)

```python
def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
    ...
    return selected_cols  # 반드시 컬럼명 문자열 리스트 반환
```

### BaseStacking (`src/models/stackdili_fixed/stacking/base.py`)

```python
def fit(self, X_train, y_train, X_test, y_test, save_dir) -> None:
    # save_dir에 pkl 저장

def evaluate(self, X_test, y_test, save_dir) -> dict:
    return {"auc": float, "threshold": float}  # 이 형식 유지
```
