import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 프로젝트 루트(/app)에서 실행되는 것을 기준으로 경로 설정
data_path = "src/features/Feature.csv"
save_path_combined = "src/features/Feature.csv"

data = pd.read_csv(data_path)

train_data = data[data['ref'] != 'DILIrank']
test_data  = data[data['ref'] == 'DILIrank']

X_train = train_data.drop(['SMILES', 'Label', 'ref'], axis=1)
y_train = train_data['Label']
X_test  = test_data.drop(['SMILES', 'Label', 'ref'], axis=1)
y_test  = test_data['Label']

assert list(X_train.columns) == list(X_test.columns), "Train/Test 피처 컬럼이 일치하지 않습니다."

N_GENERATIONS = 20
POP_SIZE      = 50
P_CROSSOVER   = 0.8
P_MUTATION    = 0.1


def evaluate(individual):
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    if len(selected_features) == 0:
        return 0.0,
    X_train_selected = X_train.iloc[:, selected_features]
    model = RandomForestClassifier(random_state=RANDOM_SEED)
    scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
    return np.mean(scores),


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X_train.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate",     tools.cxTwoPoint)
toolbox.register("mutate",   tools.mutFlipBit, indpb=0.05)
toolbox.register("select",   tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

population = toolbox.population(n=POP_SIZE)

algorithms.eaSimple(
    population, toolbox,
    cxpb=P_CROSSOVER, mutpb=P_MUTATION,
    ngen=N_GENERATIONS, verbose=True
)

best_individual   = tools.selBest(population, k=1)[0]
selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]
selected_columns  = X_train.columns[selected_features]

X_train_selected = X_train[selected_columns]
X_test_selected  = X_test[selected_columns]

model = RandomForestClassifier(random_state=RANDOM_SEED)
model.fit(X_train_selected, y_train)

y_pred   = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"선택된 피처 수: {len(selected_columns)}")
print(f"테스트 정확도: {accuracy:.4f}")

train_data_selected = train_data[['SMILES', 'Label', 'ref'] + selected_columns.tolist()]
test_data_selected  = test_data[['SMILES', 'Label', 'ref'] + selected_columns.tolist()]
combined_data = pd.concat([train_data_selected, test_data_selected], axis=0)

combined_data.to_csv(save_path_combined, index=False)
print(f"GA 선택 피처 데이터셋 저장 완료: {save_path_combined}")


if __name__ == "__main__":
    pass
