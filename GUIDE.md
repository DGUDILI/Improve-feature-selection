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

```bash
./run.sh run [stacking_version] [ga_version] [clean]
```

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `stacking_version` | Stacking 버전 (`v0`, `v0.5`, `v1`, `v3`, `v4`) | `v1` |
| `ga_version` | GA 버전 (`v0`, `v1`, `v4`, `v5`). 생략하면 GA 없이 실행 | 생략 |
| `clean` | `clean`을 입력하면 Train-Test 중복 제거 실행 | 생략 |

### 예시

```bash
./run.sh run                      # Stacking v1, GA 없음, 정제 없음
./run.sh run v1                   # Stacking v1, GA 없음, 정제 없음
./run.sh run v1 v4                # Stacking v1 + GA v4, 정제 없음
./run.sh run v1 v4 clean          # Stacking v1 + GA v4 + 데이터 정제
./run.sh run v1 "" clean          # Stacking v1, GA 없음 + 데이터 정제
./run.sh run v0.5 v1              # Stacking v0.5 + GA v1, 정제 없음
./run.sh run v4 v5                # Stacking v4 + GA v5 (권장: AUC 0.9731, MCC 0.8719)
```

> - GA를 생략하면 `Feature.csv`의 전체 피처로 바로 Stacking을 실행합니다.
> - GA는 피처 수에 따라 수십 분 이상 걸릴 수 있습니다.
> - 데이터 정제(`clean`)는 Train-Test 중복 제거가 필요한 경우에만 사용하세요.

### 결과 저장 위치

버전 조합별로 별도 디렉토리에 저장되므로 결과가 덮어써지지 않습니다.

```
src/models/stackdili_fixed/Model/
├── stacking_v1/                     # ./run.sh run v1
├── stacking_v1_clean/               # ./run.sh run v1 "" clean
├── stacking_v1_ga_v4/               # ./run.sh run v1 v4
├── stacking_v1_ga_v4_clean/         # ./run.sh run v1 v4 clean
└── stacking_v0.5_ga_v1_clean/       # ./run.sh run v0.5 v1 clean
```

### 컨테이너 쉘 접속 (직접 실험할 때)

```bash
./run.sh shell

# 컨테이너 안에서
conda run -n dili_ml_pipeline_env python src/train.py --stacking v1 --ga v4 --clean
```

---

## 현재 버전 목록

### GA

| 버전 | 파일 | 설명 |
|------|------|------|
| `v0` | `ga/ga_v0.py` | 원본 StackDILI GA — DEAP 기반, RF 5-fold CV 피트니스 |
| `v1` | `ga/ga_v1.py` | MRMR + Boruta 앙상블 (분산 필터링 → MRMR → Boruta 교집합/합집합) |
| `v4` | `ga/ga_v4.py` | XGBoost L1/L2 정규화 — CV로 최적 reg_alpha/reg_lambda 탐색, 중요도 > 0 피처 선택 |
| `v5` | `ga/ga_v5.py` | 듀얼 패스 — Path A: VT+RF Top-128, Path B: SMILES→GCN→CrossAttention → 256-dim 임베딩 추가. epochs=100, batch=64, CosineAnnealingLR, weighted CrossEntropyLoss, Early Stopping(patience=12) |

### Stacking

| 버전 | 파일 | 설명 |
|------|------|------|
| `v0` | `stacking/stacking_v0.py` | 원본 StackDILI — 직접 예측 기반, ExtraTrees 메타 모델 |
| `v0.5` | `stacking/stacking_v0_5.py` | OOF 기반, LR/SVC(스케일) + ExtraTrees 메타 모델 |
| `v1` | `stacking/stacking_v1.py` | OOF 기반, LogisticRegression 메타 모델 + 피처 힌트 + MCC 임계값 최적화 |
| `v3` | `stacking/stacking_v3.py` | OOF 기반, XGBoost 메타 + GNN 임베딩 PCA(50차원) 직접 입력 |
| `v4` | `stacking/stacking_v4.py` | **권장** OOF 기반, 베이스 모델은 fp_cols만 사용(임베딩 제외), 소프트 보팅 + LR 중 자동 선택, AUC+MCC result.txt 저장 |

**Stacking 버전 비교:**

| | `v0` | `v0.5` | `v1` | `v3` | `v4` |
|---|---|---|---|---|---|
| 예측 방식 | 직접 예측 (누수) | OOF 5-fold | OOF 5-fold | OOF 5-fold | OOF 5-fold |
| 베이스 모델 입력 | 전체 피처 | 전체 피처 | 전체 피처 | 전체 피처 | **fp_cols만** |
| 메타 모델 | ExtraTrees | ExtraTrees | LogisticRegression | XGBoost | **소프트 보팅 or LR** |
| 임베딩 처리 | - | - | - | PCA → 메타 입력 | 베이스 모델이 간접 반영 |
| 임계값 최적화 | 없음 | 없음 | 있음 | 있음 | 있음 (MCC 기준) |
| result.txt | AUC | AUC | AUC | AUC | **AUC + MCC** |

**GA v5 조합별 실험 결과 (AUC 기준):**

| 조합 | AUC | MCC | 비고 |
|------|-----|-----|------|
| `v0 + ga_v5` | 0.9731 | - | 데이터 누수 있음 |
| `v1 + ga_v5` | 0.9515 | - | 임베딩이 베이스 모델 방해 |
| `v3 + ga_v5_clean` | 0.9515 | - | |
| **`v4 + ga_v5`** | **0.9731** | **0.8719** | **권장: 누수 없이 최고 성능** |

---

## 파이프라인 흐름

```
[전처리팀] Feature.py
     ↓ src/features/Feature.csv 생성
     ↓
[자동] Feature_raw.csv 백업
     ↓ 최초 실행 시 Feature.csv → Feature_raw.csv 자동 백업
     ↓ 이후 매 실행 전 Feature_raw.csv → Feature.csv 자동 복원
     ↓
[데이터 정제, 선택] make_clean_data.py  ← --clean 옵션 지정 시 실행
     ↓ Train-Test 중복 제거
     ↓
[GA, 선택] ga_vN.py → select_features()  ← --ga 옵션 지정 시 실행
     ↓ 선택된 피처로 Feature.csv 덮어쓰기
     ↓
[스태킹] stacking_vN.py → fit() → evaluate()
     ↓ src/models/stackdili_fixed/Model/stacking_{sv}_ga_{gv}/ 에 pkl 저장
     ↓
결과 출력 (OOF AUC / Eval AUC)
```

### Feature.csv 보호 방식

- **최초 실행 시**: `Feature.csv`를 `Feature_raw.csv`로 자동 백업합니다.
- **이후 매 실행 전**: `Feature_raw.csv` → `Feature.csv`로 자동 복원합니다.
- GA가 `Feature.csv`를 덮어써도 다음 실행 전 원본으로 되돌아갑니다.

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
