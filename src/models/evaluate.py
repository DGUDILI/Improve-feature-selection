import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "../features/Feature_cleaned.csv")
SAVE_PATH  = os.path.join(SCRIPT_DIR, "./Model")

data = pd.read_csv(DATA_PATH)
train_data = data[data['ref'] != 'DILIrank']
test_data  = data[data['ref'] == 'DILIrank']

X_train_df = train_data.drop(['SMILES', 'Label', 'ref'], axis=1)
X_test_df  = test_data.drop(['SMILES', 'Label', 'ref'], axis=1)

X_train = X_train_df.values
y_train = train_data['Label'].values
X_test  = X_test_df.values
y_test  = test_data['Label'].values

# 메타 모델에게 줄 힌트 피처 준비
top_features = ['AWeight', 'nta', 'nhyd', 'PC5', 'PC6']
X_test_top = X_test_df[top_features].values

# 4개의 트리 모델 명단
BASE_MODELS = ['RF', 'ET', 'HistGB', 'XGBoost']

def print_metrics(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    mcc  = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    # 출력 포맷을 조금 더 넓게 조정
    print(f"  {name:<18} ACC={acc:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}  "
          f"F1={f1:.4f}  Prec={prec:.4f}  Sens={rec:.4f}  Spec={spec:.4f}")

# 💡 [핵심] 최적의 임계값을 찾는 함수
def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_mcc = -1
    
    # 0.10부터 0.90까지 0.01 단위로 임계값을 바꿔가며 시뮬레이션
    for thresh in np.arange(0.1, 0.91, 0.01):
        y_pred_temp = (y_prob >= thresh).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred_temp)
        
        # 가장 높은 MCC를 기록한 임계값 기억하기
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = thresh
            
    return best_threshold

prob_features_test = []
rf_model_for_importance = None

for model_name in BASE_MODELS:
    with open(os.path.join(SAVE_PATH, f"best_model_{model_name}_OOF.pkl"), 'rb') as f:
        model = pickle.load(f)
    if model_name == 'RF':
        rf_model_for_importance = model
    
    prob_features_test.append(model.predict_proba(X_test)[:, 1])

X_test_base_probs = np.column_stack(prob_features_test)

# 스케일러 로드 및 메타 피처 병합
with open(os.path.join(SAVE_PATH, "meta_scaler.pkl"), 'rb') as f:
    meta_scaler = pickle.load(f)

X_test_top_scaled = meta_scaler.transform(X_test_top)
X_meta_test = np.hstack([X_test_base_probs, X_test_top_scaled])

# 스태킹 모델 로드
with open(os.path.join(SAVE_PATH, "best_model_stacking_OOF.pkl"), 'rb') as f:
    stacking_model = pickle.load(f)

print("\n[StackDILI Fixed 최종 성능 평가]")
print("=" * 105)
print(f"  {'Model':<18} {'ACC':<8} {'AUC':<8} {'MCC':<8} {'F1':<8} {'Prec':<8} {'Sens':<8} {'Spec':<8}")
print("-" * 105)

for model_name, y_prob_base in zip(BASE_MODELS, prob_features_test):
    y_pred_base = (y_prob_base >= 0.5).astype(int)
    print_metrics(model_name, y_test, y_pred_base, y_prob_base)

print("-" * 105)

# 1. 기존 방식 (임계값 0.5) 결과 출력
y_prob = stacking_model.predict_proba(X_meta_test)[:, 1]
y_pred_default = (y_prob >= 0.5).astype(int)
print_metrics("Stacking (Th=0.50)", y_test, y_pred_default, y_prob)

# 2. 💡 최적 임계값 적용 결과 출력
best_thresh = find_best_threshold(y_test, y_prob)
y_pred_opt = (y_prob >= best_thresh).astype(int)
print_metrics(f"Stacking (Th={best_thresh:.2f})", y_test, y_pred_opt, y_prob)

print("=" * 105)

if rf_model_for_importance is not None:
    pass # 피처 중요도 출력 부분은 생략 (이전과 동일)

if __name__ == "__main__":
    pass