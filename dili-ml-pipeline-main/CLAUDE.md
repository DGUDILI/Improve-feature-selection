# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A machine learning pipeline for DILI (Drug-Induced Liver Injury) prediction, improving the original StackDILI model with optional GA-based feature selection and ensemble stacking. Built for Python 3.10, running in Docker + Conda.

## Commands

All commands use `run.sh` as the entry point (Docker wrapper):

```bash
./run.sh build          # Build Docker image
./run.sh env-test       # Validate conda environment
./run.sh run            # Run pipeline (default: stacking=v1, no GA)
./run.sh run v1         # Stacking v1, no GA
./run.sh run v1 v0      # Stacking v1 + GA v0
./run.sh shell          # Interactive shell inside container
```

Inside the container, the pipeline can be run directly:
```bash
conda run -n dili_ml_pipeline_env python src/train.py --stacking v1 --ga v0
```

There is no test suite. Environment validation is done via `src/env_test.py`.

## Architecture

### Three-Stage Pipeline

**Entry point**: `src/train.py` → `src/registry.py` → `src/models/stackdili_fixed/model.py`

The `Model` class in `model.py` orchestrates:
1. **원본 복원**: `Feature_raw.csv`가 존재하면 매 실행 시 `Feature.csv`로 자동 복사 (GA 덮어쓰기 방지)
2. **Feature extraction** (pre-computed, reads `src/features/Feature.csv`)
3. **GA feature selection** (optional) — `ga/ga_vN.py` 실행, `Feature.csv` 덮어쓰기. GA 실행 후 Feature.csv를 재로드하여 필터링 (GAv5처럼 GA가 Feature.csv를 직접 수정하는 경우 대응)
4. **Data cleaning** — calls `src/preprocessing/make_clean_data.py`, produces `Feature_cleaned.csv` (removes Train-Test duplicates via canonical SMILES normalization using RDKit)
5. **Stacking** — trains ensemble, saves models to `src/models/stackdili_fixed/Model/`

### Registry Pattern

`src/registry.py` maps version strings to classes:
```python
GA_REGISTRY = {"v0": GAv0, "v1": GAv1, ..., "v5": GAv5}
STACKING_REGISTRY = {"v0": StackingV0, "v1": StackingV1}
```

To add a new version: implement the class extending `BaseGA` or `BaseStacking`, then register it here.

### Stacking Versions

| | v0 (original) | v1 (improved) |
|---|---|---|
| Data leakage | Yes (direct prediction) | No (OOF-based) |
| Meta model | ExtraTreesClassifier (10 iters) | LogisticRegression (1 iter) |
| Threshold | Fixed | Optimized via MCC |

Both use 4 base models: RandomForest, ExtraTrees, HistGradientBoosting, XGBoost.

### Data Flow

```
data/Dataset.csv → Feature.py → Feature.csv
                                     ↓ (optional GA)
                               Feature.csv (filtered / embedding 추가)
                                     ↓
                          make_clean_data.py → Feature_cleaned.csv
                                                      ↓
                                             Stacking → Model/ + console metrics
```

**Train/test split**: determined by the `ref` column in `Feature.csv` — rows where `ref == 'DILIrank'` are the test set.

### Feature 원본 보존

- `src/features/Feature_raw.csv`: 원본 피처 파일 (절대 덮어쓰지 않음)
- `src/features/Feature.csv`: GA 실행 시 덮어써지는 작업 파일
- `model.py`의 `_restore_features()`가 매 실행 전 `Feature_raw.csv → Feature.csv` 자동 복사

### Interface Contracts

`BaseGA` (`ga/base.py`):
```python
def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:  # column names
```

`BaseStacking` (`stacking/base.py`):
```python
def fit(self, X_train, y_train, X_test, y_test, save_dir) -> None
def evaluate(self, X_test, y_test, save_dir) -> dict  # {"auc": float, "threshold": float}
```

### Model Persistence

Trained models are saved as `.pkl` files under `src/models/stackdili_fixed/Model/` (git-ignored). Key files: `best_model_{name}_OOF.pkl`, `meta_model.pkl`, `threshold.pkl`, `result.txt`.

---

## GA 버전 목록

| 버전 | 파일 | 방법 | Eval AUC | MCC | 선택 피처 수 |
|------|------|------|----------|-----|------------|
| v0 | `ga/ga_v0.py` | DEAP GA — RF 5-fold Accuracy, 단일 목적 | 0.8896 | 0.6404 | 119개 |
| v1 | `ga/ga_v1.py` | MRMR + Boruta 앙상블 (perc=75) | 0.8917 | 0.6354 | 33개 |
| v2 | `ga/ga_v2.py` | NSGA-II — MCC + 피처 수 파레토 최적화 | 0.8874 | 0.6320 | 83개 |
| v3 | `ga/ga_v3.py` | VT + MRMR Top-128 | 0.8937 | 0.6421 | 128개 |
| v4 | `ga/ga_v4.py` | XGBoost L1/L2 정규화 — CV 최적 탐색 | 0.8895 | 0.6320 | 118개 |
| v5 | `ga/ga_v5.py` | 이미지 아키텍처 — VT+RF(Path A) + GCN+CrossAttention 24-dim(Path B) + 256-dim 임베딩 스태킹 연결 | **0.9329** | **0.7547** | 384개 (RF 128 + emb 256) |

### GA 버전별 주요 특징

**v1 (MRMR + Boruta)**
- `boruta_perc=75` (기본값): Boruta 34개 Confirmed → MRMR 교집합 33개 선택
- `boruta_perc=100` 시: Boruta 2개만 Confirmed → 합집합 fallback → 128개 (v3와 동일 결과)
- 의존성: `mrmr-selection`, `Boruta`

**v2 (NSGA-II)**
- `pop_size=30`, `n_generations=10`, `cv_folds=3` (속도 최적화)
- 원래 설정(100/50/5)은 41분+ 소요로 실용성 없음
- DEAP `tools.selNSGA2` 사용, `FitnessV2Multi`/`IndividualV2` 고유 이름으로 GAv0 충돌 방지

**v3 (VT + MRMR)**
- 가장 빠름, 추가 의존성 없음 (mrmr-selection만 필요)
- v1 perc=100과 동일 결과 산출 (둘 다 MRMR Top-128 최종 선택)

**v4 (XGBoost L1/L2)**
- `reg_alphas=[0.1,1.0,10.0]`, `reg_lambdas=[1.0,10.0,100.0]` 9조합 × CV
- `scale_pos_weight` 자동 계산으로 DILI 불균형 대응
- `feature_importances_ > 0` 기준 선택, `min_features=10` fallback

**v5 (이미지 아키텍처)**
- Path A: VT(0.01) → RF Top-128 → `fp_vec` (Projection Block 없음)
- Path B: SMILES → TwoLayerGCN(25→64→128) → NodeFPCrossAttention → GraphAttentionReadout(Q/K/V=24-dim)
- 패딩 기반 미니배치 (`batch_size=32`, `max_atoms=492`): 샘플별 루프 대비 ~32배 빠름
- 256-dim 임베딩을 `emb_000~emb_255`로 Feature.csv에 추가 → Stacking 직접 입력
- `Feature_raw.csv`에서 train+test 전체 SMILES 읽어 임베딩 생성
- 모듈: `ga/modules/atom_features.py`, `gcn.py`, `attention.py`
- 의존성: `torch`, `rdkit` (기존 설치됨)

---

## 의존성 (docker/environment.yml pip 섹션)

```yaml
pip:
  - torch-geometric
  - deap          # GA v0, v2
  - mrmr-selection  # GA v1, v3
  - Boruta        # GA v1
```

`torch`, `rdkit`, `xgboost`, `scikit-learn`은 conda 채널에서 설치.

---

## 주요 설계 결정 및 주의사항

- **Feature.csv 덮어쓰기**: 모든 GA 버전이 `Feature.csv`를 덮어씀. `Feature_raw.csv`를 반드시 유지할 것
- **GAv5 Feature.csv 직접 수정**: GAv5는 `select_features()` 내부에서 Feature.csv를 직접 저장함. `model.py`는 GA 실행 후 Feature.csv를 재로드하여 이를 처리
- **GAv2 DEAP 충돌**: `creator.FitnessV2Multi`, `creator.IndividualV2` 고유 이름 사용 필수 (GAv0의 `FitnessMax`, `Individual`과 충돌 방지)
- **GAv5 NaN 방지**: `GraphAttentionReadout`에서 A1/A2 각각 독립적으로 마스킹 후 `nan_to_num` 적용 (`-scores`에 `+inf` 발생 방지)
- **iFeatureOmegaCLI 미설치**: `Feature.py` 재실행 불가. 원본 피처 재생성 필요 시 `Feature_raw.csv` 직접 제공
