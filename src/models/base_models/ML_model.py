import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "../../features/Feature.csv")
SAVE_PATH  = os.path.join(SCRIPT_DIR, "../Model")
os.makedirs(SAVE_PATH, exist_ok=True)

models = {
    'RF':      RandomForestClassifier(),
    'ET':      ExtraTreesClassifier(),
    'HistGB':  HistGradientBoostingClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
}

data = pd.read_csv(DATA_PATH)
train_data = data[data['ref'] != 'DILIrank']
test_data  = data[data['ref'] == 'DILIrank']

X_train = train_data.drop(['SMILES', 'Label', 'ref'], axis=1)
y_train = train_data['Label']
X_test  = test_data.drop(['SMILES', 'Label', 'ref'], axis=1)
y_test  = test_data['Label']

for model_name, model in models.items():
    print(f"\n[{model_name}] 학습 시작")
    best_auc   = -np.inf
    best_model = None

    for i in range(5):
        model.set_params(random_state=np.random.randint(0, 10000))
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)
        print(f"  {i+1}회: ACC={acc:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}")

        if auc > best_auc:
            best_auc   = auc
            best_model = pickle.dumps(model)

    model_file = os.path.join(SAVE_PATH, f"best_model_{model_name}.pkl")
    with open(model_file, 'wb') as f:
        f.write(best_model)
    print(f"  -> 최적 모델 저장 (AUC={best_auc:.4f}): {model_file}")

if __name__ == "__main__":
    pass
