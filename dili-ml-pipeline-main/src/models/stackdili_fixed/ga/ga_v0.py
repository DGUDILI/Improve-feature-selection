import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from deap import base, creator, tools, algorithms

from models.stackdili_fixed.ga.base import BaseGA


class GAv0(BaseGA):
    """원본 StackDILI GA — DEAP 기반 유전 알고리즘 피처 선택 (v0).

    출처: https://github.com/GGCL7/StackDILI
    변경 없이 원본 로직 그대로 유지.
    """

    def __init__(
        self,
        n_generations: int = 20,
        pop_size: int = 50,
        p_crossover: float = 0.8,
        p_mutation: float = 0.1,
        random_seed: int = 42,
    ):
        self.n_generations = n_generations
        self.pop_size      = pop_size
        self.p_crossover   = p_crossover
        self.p_mutation    = p_mutation
        self.random_seed   = random_seed

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        np.random.seed(self.random_seed)
        X_vals = X.values
        y_vals = y.values

        def _evaluate(individual):
            selected = [i for i, bit in enumerate(individual) if bit == 1]
            if len(selected) == 0:
                return 0.0,
            scores = cross_val_score(
                RandomForestClassifier(random_state=self.random_seed),
                X_vals[:, selected], y_vals,
                cv=5, scoring='accuracy',
            )
            return np.mean(scores),

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool",  np.random.randint, 2)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_bool, n=X.shape[1])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate",       tools.cxTwoPoint)
        toolbox.register("mutate",     tools.mutFlipBit, indpb=0.05)
        toolbox.register("select",     tools.selTournament, tournsize=3)
        toolbox.register("evaluate",   _evaluate)

        population = toolbox.population(n=self.pop_size)
        algorithms.eaSimple(
            population, toolbox,
            cxpb=self.p_crossover, mutpb=self.p_mutation,
            ngen=self.n_generations, verbose=True,
        )

        best = tools.selBest(population, k=1)[0]
        selected_cols = X.columns[[i for i, bit in enumerate(best) if bit == 1]].tolist()

        print(f"GA 선택된 피처 수: {len(selected_cols)}")
        return selected_cols
