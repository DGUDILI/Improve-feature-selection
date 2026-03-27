# import os
# import warnings
# warnings.filterwarnings("ignore")

# import numpy as np
# import pickle
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     accuracy_score, roc_auc_score, matthews_corrcoef,
#     precision_score, recall_score, f1_score, confusion_matrix
# )

# RANDOM_SEED = 42
# N_SPLITS    = 5
# np.random.seed(RANDOM_SEED)

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH  = os.path.join(SCRIPT_DIR, "../Feature/Feature_cleaned.csv")
# SAVE_PATH  = os.path.join(SCRIPT_DIR, "../Model")
# os.makedirs(SAVE_PATH, exist_ok=True)

# data = pd.read_csv(DATA_PATH)
# train_data = data[data['ref'] != 'DILIrank']
# test_data  = data[data['ref'] == 'DILIrank']

# X_train = train_data.drop(['SMILES', 'Label', 'ref'], axis=1).values
# y_train = train_data['Label'].values
# X_test  = test_data.drop(['SMILES', 'Label', 'ref'], axis=1).values
# y_test  = test_data['Label'].values

# models = {
#     'RF':      RandomForestClassifier(random_state=RANDOM_SEED),
#     'ET':      ExtraTreesClassifier(random_state=RANDOM_SEED),
#     'HistGB':  HistGradientBoostingClassifier(random_state=RANDOM_SEED),
#     'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED, verbosity=0),
# }

# def print_metrics(name, y_true, y_pred, y_prob):
#     acc  = accuracy_score(y_true, y_pred)
#     auc  = roc_auc_score(y_true, y_prob)
#     mcc  = matthews_corrcoef(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, zero_division=0)
#     rec  = recall_score(y_true, y_pred, zero_division=0)
#     f1   = f1_score(y_true, y_pred, zero_division=0)
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     spec = tn / (tn + fp) if (tn + fp) > 0 else 0
#     print(f"  {name:<15} ACC={acc:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}  "
#           f"F1={f1:.4f}  Prec={prec:.4f}  Sens={rec:.4f}  Spec={spec:.4f}")
#     return auc

# # OOF 결과를 저장할 배열
# oof_train = np.zeros((len(y_train), len(models)))
# oof_test  = np.zeros((len(y_test),  len(models)))

# # K-Fold OOF 예측 (데이터 누수 방지 핵심 로직)
# print(f"[1/3] {N_SPLITS}-Fold OOF 기반 베이스 모델 학습 및 예측")
# skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

# for col_idx, (name, model) in enumerate(models.items()):
#     print(f"  -> {name} OOF 추출 중...")
#     test_preds = np.zeros((len(y_test), N_SPLITS))

#     for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
#         X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
#         X_fold_val,   y_fold_val   = X_train[val_idx],   y_train[val_idx]

#         model.fit(X_fold_train, y_fold_train)
#         oof_train[val_idx, col_idx] = model.predict_proba(X_fold_val)[:, 1]
#         test_preds[:, fold_idx]     = model.predict_proba(X_test)[:, 1]

#     # Test 예측은 fold별 평균 사용
#     oof_test[:, col_idx] = test_preds.mean(axis=1)

#     # 전체 데이터로 최종 베이스 모델 저장
#     model.fit(X_train, y_train)
#     with open(os.path.join(SAVE_PATH, f"best_model_{name}_OOF.pkl"), 'wb') as f:
#         pickle.dump(model, f)

# # ==============================================================================
# # 여기서부터 복사해서 기존 코드의 [2/3] 부분부터 끝까지 덮어쓰기 하세요!
# # ==============================================================================

# # 메타 모델 (Logistic Regression) 학습
# print("\n[2/3] 메타 모델 (LogisticRegression) 학습")
# # 복잡한 트리 대신, 4개 모델의 최적 가중치만 찾는 로지스틱 회귀 사용!
# stacking_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
# stacking_model.fit(oof_train, y_train)

# with open(os.path.join(SAVE_PATH, "best_model_stacking_OOF.pkl"), 'wb') as f:
#     pickle.dump(stacking_model, f)

# # 최종 평가
# print("\n[3/3] 최종 성능 평가")
# print("=" * 95)
# print(f"  {'Model':<15} {'ACC':<8} {'AUC':<8} {'MCC':<8} {'F1':<8} {'Prec':<8} {'Sens':<8} {'Spec':<8}")
# print("-" * 95)

# # 1. 베이스 모델들 성능 출력
# for col_idx, name in enumerate(models.keys()):
#     y_prob_base = oof_test[:, col_idx]
#     y_pred_base = (y_prob_base >= 0.5).astype(int)
#     print_metrics(name, y_test, y_pred_base, y_prob_base)

# print("-" * 95)

# # 2. 로지스틱 회귀를 이용한 Stacking 성능
# y_pred_stack = stacking_model.predict(oof_test)
# y_prob_stack = stacking_model.predict_proba(oof_test)[:, 1]
# auc_stack = print_metrics("Stacking(LR)", y_test, y_pred_stack, y_prob_stack)

# # 3. 단순 평균(Soft Voting) 성능 비교 (보너스)
# y_prob_soft = oof_test.mean(axis=1)
# y_pred_soft = (y_prob_soft >= 0.5).astype(int)
# auc_soft = print_metrics("Soft Voting", y_test, y_pred_soft, y_prob_soft)

# print("=" * 95)

# # 가중치 확인 (어떤 베이스 모델의 말을 가장 신뢰했는지 구경하기)
# weights = stacking_model.coef_[0]
# print("\n[💡 메타 모델 가중치 (어떤 모델을 제일 믿었을까?)]")
# for name, weight in zip(models.keys(), weights):
#     print(f"  - {name}: {weight:.4f}")

# # result.txt 저장 (가장 높은 성능을 기록한 모델의 AUC 저장)
# final_best_auc = max(auc_stack, auc_soft)
# result_path = os.path.join(SCRIPT_DIR, "../result.txt")
# with open(result_path, "w") as f:
#     f.write(str(final_best_auc))

# # Train과 Test 중복 화합물 검사 (치팅 검사)
# original_data = pd.read_csv(os.path.join(SCRIPT_DIR, "../Feature/Feature.csv"))
# train_smiles = set(original_data[original_data['ref'] != 'DILIrank']['SMILES'])
# test_smiles  = set(original_data[original_data['ref'] == 'DILIrank']['SMILES'])
# overlap = train_smiles.intersection(test_smiles)

# print(f"\nTrain 데이터 화합물 개수: {len(train_smiles)}")
# print(f"Test 데이터 화합물 개수: {len(test_smiles)}")
# print(f"Train과 Test에 중복으로 존재하는 화합물 개수: {len(overlap)}개")
# if len(test_smiles) > 0:
#     print(f"Test 셋 중 {len(overlap) / len(test_smiles) * 100:.2f}%가 이미 Train 셋에 있는 데이터입니다")

# if __name__ == "__main__":
#     pass


import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix
)

RANDOM_SEED = 42
N_SPLITS    = 5
np.random.seed(RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "../../features/Feature_cleaned.csv")
SAVE_PATH  = os.path.join(SCRIPT_DIR, "../Model")
os.makedirs(SAVE_PATH, exist_ok=True)

data = pd.read_csv(DATA_PATH)
train_data = data[data['ref'] != 'DILIrank']
test_data  = data[data['ref'] == 'DILIrank']

X_train_df = train_data.drop(['SMILES', 'Label', 'ref'], axis=1)
X_test_df  = test_data.drop(['SMILES', 'Label', 'ref'], axis=1)

X_train = X_train_df.values
y_train = train_data['Label'].values
X_test  = X_test_df.values
y_test  = test_data['Label'].values

# 💡 [핵심 작전 B] 메타 모델에게 힌트로 줄 Top 5 원본 피처 추출
top_features = ['AWeight', 'nta', 'nhyd', 'PC5', 'PC6']
X_train_top = X_train_df[top_features].values
X_test_top  = X_test_df[top_features].values

# 다시 든든한 4개의 트리 모델로 복귀
models = {
    'RF':      RandomForestClassifier(random_state=RANDOM_SEED),
    'ET':      ExtraTreesClassifier(random_state=RANDOM_SEED),
    'HistGB':  HistGradientBoostingClassifier(random_state=RANDOM_SEED),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED, verbosity=0),
}

def print_metrics(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    mcc  = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"  {name:<15} ACC={acc:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}  "
          f"F1={f1:.4f}  Prec={prec:.4f}  Sens={rec:.4f}  Spec={spec:.4f}")
    return auc

oof_train = np.zeros((len(y_train), len(models)))
oof_test  = np.zeros((len(y_test),  len(models)))

print(f"[1/3] {N_SPLITS}-Fold OOF 기반 베이스 모델 학습 및 예측")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

for col_idx, (name, model) in enumerate(models.items()):
    print(f"  -> {name} OOF 추출 중...")
    test_preds = np.zeros((len(y_test), N_SPLITS))

    # 트리 모델들은 스케일링이 필요 없으므로 X_train 그대로 사용
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
        X_fold_val,   y_fold_val   = X_train[val_idx],   y_train[val_idx]

        model.fit(X_fold_train, y_fold_train)
        oof_train[val_idx, col_idx] = model.predict_proba(X_fold_val)[:, 1]
        test_preds[:, fold_idx]     = model.predict_proba(X_test)[:, 1]

    oof_test[:, col_idx] = test_preds.mean(axis=1)

    model.fit(X_train, y_train)
    with open(os.path.join(SAVE_PATH, f"best_model_{name}_OOF.pkl"), 'wb') as f:
        pickle.dump(model, f)

# 💡 [핵심 작전 B] 원본 힌트 피처 스케일링 후 예측값과 병합!
print("\n[2/3] 메타 모델 (LogisticRegression + 피처 힌트) 학습")
meta_scaler = StandardScaler()
X_train_top_scaled = meta_scaler.fit_transform(X_train_top)
X_test_top_scaled  = meta_scaler.transform(X_test_top)

# 평가 파일에서 쓸 수 있도록 스케일러 저장
with open(os.path.join(SAVE_PATH, "meta_scaler.pkl"), 'wb') as f:
    pickle.dump(meta_scaler, f)

# 4개 모델 확률값 + 5개 원본 피처 = 총 9개 피처 완성!
X_meta_train = np.hstack([oof_train, X_train_top_scaled])
X_meta_test  = np.hstack([oof_test, X_test_top_scaled])

stacking_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
stacking_model.fit(X_meta_train, y_train)

with open(os.path.join(SAVE_PATH, "best_model_stacking_OOF.pkl"), 'wb') as f:
    pickle.dump(stacking_model, f)

print("\n[3/3] 학습 시뮬레이션 평가")
print("=" * 95)
for col_idx, name in enumerate(models.keys()):
    y_prob_base = oof_test[:, col_idx]
    y_pred_base = (y_prob_base >= 0.5).astype(int)
    print_metrics(name, y_test, y_pred_base, y_prob_base)

print("-" * 95)
y_pred_stack = stacking_model.predict(X_meta_test)
y_prob_stack = stacking_model.predict_proba(X_meta_test)[:, 1]
auc_stack = print_metrics("Stacking(LR+Feat)", y_test, y_pred_stack, y_prob_stack)
print("=" * 95)

# 가중치 확인: 메타 모델이 모델들의 예측값을 더 믿었을까, 아니면 원본 피처를 더 믿었을까?
weights = stacking_model.coef_[0]
meta_features = list(models.keys()) + top_features
print("\n[💡 메타 모델 가중치 (힌트 피처 포함)]")
for name, weight in zip(meta_features, weights):
    print(f"  - {name}: {weight:.4f}")

final_best_auc = auc_stack
result_path = os.path.join(SCRIPT_DIR, "../result.txt")
with open(result_path, "w") as f:
    f.write(str(final_best_auc))

if __name__ == "__main__":
    pass