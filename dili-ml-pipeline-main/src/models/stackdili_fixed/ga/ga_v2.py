import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from deap import base, creator, tools, algorithms

from models.stackdili_fixed.ga.base import BaseGA


class GAv2(BaseGA):
    """NSGA-II 기반 다목적 피처 선택 (v2).

    두 개의 목적을 동시에 최적화합니다.
    - f1: MCC 최대화 (불균형 DILI 데이터에 적합한 피트니스 척도)
    - f2: 선택 피처 수 최소화 (모델 간결성)

    NSGA-II가 파레토 프론트를 도출하면, pareto_select 전략으로 최종 해를 1개 선택합니다.
    - 'best_mcc'  : 파레토 프론트 내 MCC 최고 해
    - 'balanced'  : MCC - feat_penalty * feature_ratio 가중합 최고 해
    """

    def __init__(
        self,
        pop_size: int = 30,
        n_generations: int = 10,
        p_crossover: float = 0.9,
        p_mutation: float = 0.1,
        cv_folds: int = 3,
        pareto_select: str = "best_mcc",
        feat_penalty: float = 0.1,
        random_seed: int = 42,
        n_jobs: int = -1,
    ):
        self.pop_size       = pop_size
        self.n_generations  = n_generations
        self.p_crossover    = p_crossover
        self.p_mutation     = p_mutation
        self.cv_folds       = cv_folds
        self.pareto_select  = pareto_select
        self.feat_penalty   = feat_penalty
        self.random_seed    = random_seed
        self.n_jobs         = n_jobs

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _make_evaluate(self, X_vals, y_vals):
        """클로저로 평가 함수 생성 (DEAP toolbox에 등록용)."""
        cv_folds     = self.cv_folds
        random_seed  = self.random_seed
        n_jobs       = self.n_jobs
        n_features   = X_vals.shape[1]

        def _evaluate(individual):
            selected_idx = [i for i, bit in enumerate(individual) if bit == 1]

            if len(selected_idx) == 0:
                return 0.0, 1.0  # 최악값: -MCC=0 최소화 불가, 피처 비율=1

            rf = RandomForestClassifier(
                class_weight="balanced",
                random_state=random_seed,
                n_jobs=n_jobs,
            )
            skf = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=random_seed
            )
            X_sub = X_vals[:, selected_idx]

            mcc_scores = []
            for tr_idx, val_idx in skf.split(X_sub, y_vals):
                rf.fit(X_sub[tr_idx], y_vals[tr_idx])
                y_pred = rf.predict(X_sub[val_idx])
                mcc_scores.append(matthews_corrcoef(y_vals[val_idx], y_pred))

            mean_mcc      = float(np.mean(mcc_scores))
            feature_ratio = len(selected_idx) / n_features

            # DEAP는 최소화 기준: (-MCC, feature_ratio) 모두 최소화
            return -mean_mcc, feature_ratio

        return _evaluate

    def _build_toolbox(self, n_features, evaluate_fn):
        """DEAP toolbox 구성. GAv0과 충돌 방지를 위해 고유 이름 사용."""
        if not hasattr(creator, "FitnessV2Multi"):
            creator.create("FitnessV2Multi", base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, "IndividualV2"):
            creator.create("IndividualV2", list, fitness=creator.FitnessV2Multi)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool",   random.randint, 0, 1)
        toolbox.register(
            "individual", tools.initRepeat,
            creator.IndividualV2, toolbox.attr_bool, n=n_features,
        )
        toolbox.register("population",  tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate",        tools.cxUniform,  indpb=0.5)
        toolbox.register("mutate",      tools.mutFlipBit, indpb=0.05)
        toolbox.register("select",      tools.selNSGA2)
        toolbox.register("evaluate",    evaluate_fn)
        return toolbox

    def _run_nsga2(self, toolbox):
        """NSGA-II 메인 루프."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # 초기 모집단 생성 및 평가
        pop = toolbox.population(n=self.pop_size)
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        # NSGA-II 선택으로 초기 정렬
        pop = toolbox.select(pop, len(pop))

        for gen in range(1, self.n_generations + 1):
            # 자손 생성
            offspring = algorithms.varAnd(
                pop, toolbox,
                cxpb=self.p_crossover,
                mutpb=self.p_mutation,
            )

            # 미평가 자손만 피트니스 계산 (비용 절감)
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)

            # (μ + λ) NSGA-II 선택
            pop = toolbox.select(pop + offspring, self.pop_size)

            # 진행 로그
            pareto_front = tools.sortNondominated(
                pop, len(pop), first_front_only=True
            )[0]
            mcc_vals  = [-ind.fitness.values[0] for ind in pareto_front]
            feat_vals = [
                sum(ind) for ind in pareto_front
            ]
            print(
                f"[NSGA-II] 세대 {gen:3d}/{self.n_generations} | "
                f"파레토: {len(pareto_front):3d}개 | "
                f"최고 MCC: {max(mcc_vals):.4f} | "
                f"최소 피처: {min(feat_vals):3d}개"
            )

        return pop

    def _select_from_pareto(self, population, col_names):
        """파레토 프론트에서 전략에 따라 최종 해 1개 선택."""
        pareto_front = tools.sortNondominated(
            population, len(population), first_front_only=True
        )[0]

        n_total = len(col_names)
        mcc_vals   = [-ind.fitness.values[0] for ind in pareto_front]
        feat_counts = [sum(ind) for ind in pareto_front]

        print(f"\n[파레토 프론트 요약]")
        print(
            f"  총 {len(pareto_front)}개 해 | "
            f"MCC 범위: {min(mcc_vals):.4f} ~ {max(mcc_vals):.4f} | "
            f"피처 수 범위: {min(feat_counts)} ~ {max(feat_counts)}개"
        )

        if self.pareto_select == "balanced":
            scores = [
                mcc - self.feat_penalty * (fc / n_total)
                for mcc, fc in zip(mcc_vals, feat_counts)
            ]
            best_ind = pareto_front[int(np.argmax(scores))]
        else:  # "best_mcc" (기본값)
            best_ind = pareto_front[int(np.argmax(mcc_vals))]

        selected_cols = [
            col_names[i] for i, bit in enumerate(best_ind) if bit == 1
        ]
        best_mcc  = -best_ind.fitness.values[0]
        print(
            f"[최종 선택] 전략='{self.pareto_select}' → "
            f"MCC={best_mcc:.4f}, 피처 수={len(selected_cols)}개"
        )
        return selected_cols

    # ------------------------------------------------------------------
    # BaseGA 인터페이스
    # ------------------------------------------------------------------

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        X_vals    = X.values
        y_vals    = y.values
        col_names = X.columns.tolist()

        print(f"[NSGA-II] 초기화 | 모집단={self.pop_size} | 세대={self.n_generations} | 피처={X.shape[1]}개")
        print(f"[NSGA-II] 목적함수: f1=-MCC(최소화), f2=피처비율(최소화)")

        evaluate_fn = self._make_evaluate(X_vals, y_vals)
        toolbox     = self._build_toolbox(X.shape[1], evaluate_fn)
        population  = self._run_nsga2(toolbox)
        selected    = self._select_from_pareto(population, col_names)

        print(f"\nNSGA-II 선택된 피처 수: {len(selected)}")
        return selected
