# dili-ml-pipeline
### Senior-Project_3

Dongguk University Capston Design Class 2, Group 3 Repository

# DILI ML Pipeline

StackDILI 개선 모델 개발 및 실험 파이프라인

자세한 실행 방법 및 가이드는 [GUIDE.md](GUIDE.md)를 참고하세요.

---

## 팀 역할 분담

| 역할 | 담당 파일 |
|------|-----------|
| 데이터 전처리 | `src/features/`, `src/preprocessing/` |
| GA + 베이스 모델 | `src/models/stackdili_fixed/ga/`, `src/models/stackdili_fixed/base_models/` |
| 스태킹 + 결과 분석 | `src/models/stackdili_fixed/stacking/` |
| XAI | `src/xai/` |

---

## 프로젝트 구조

```
dili_ml_pipeline/
├── run.sh                          # 실행 진입점
├── Dockerfile
├── docker-compose.yml
├── docker/
│   └── environment.yml             # conda 환경 정의
│
├── data/
│   └── Dataset.csv                 # 원본 데이터
│
├── src/
│   ├── train.py                    # 학습 진입점
│   ├── registry.py                 # GA / Stacking 버전 등록
│   ├── env_test.py                 # 환경 확인용
│   │
│   ├── features/
│   │   ├── Feature.py              # 피처 추출 (iFeatureOmegaCLI)
│   │   └── Feature.csv             # 추출된 피처 (Feature.py 결과물)
│   │
│   ├── preprocessing/
│   │   └── make_clean_data.py      # Train-Test 중복 제거
│   │
│   └── models/
│       └── stackdili_fixed/
│           ├── model.py            # 조립기: GA → 정제 → Stacking 순서 관리
│           │
│           ├── ga/
│           │   ├── base.py         # GA 인터페이스 (BaseGA)
│           │   └── ga_v1.py        # GA 구현 v1 (DEAP 기반)
│           │
│           ├── stacking/
│           │   ├── base.py         # Stacking 인터페이스 (BaseStacking)
│           │   └── stacking_v1.py  # Stacking 구현 v1 (OOF + LR 메타)
│           │
│           ├── base_models/
│           │   └── ML_model.py     # 단독 베이스 모델 실험용 (파이프라인 외)
│           │
│           └── Model/              # 학습된 모델 저장 (자동 생성, git 제외)
│
└── outputs/
    ├── docs/
    └── models/
```
