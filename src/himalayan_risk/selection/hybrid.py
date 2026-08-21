"""Sequential FOA-GWO hybrid (Algorithm 1).

Taking the union or intersection of separately obtained subsets was
rejected early, since neither operation preserves the interaction
structure that made a subset work. Instead both searches are chained
inside a single loop:

* FOA phase - local seeding scatters candidates across the space.
* GWO phase - the population is drawn toward the alpha, beta and delta
  leaders, with the control parameter `a` decaying linearly from 2 to 0
  so exploration dominates early iterations and exploitation dominates
  later ones.
"""

from __future__ import annotations

import logging
from typing import Callable, List

import numpy as np
import pandas as pd

from ..modelling import mask_fitness

LOGGER = logging.getLogger(__name__)

Evaluator = Callable[[np.ndarray], float]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def FOA_GWO_hybrid_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    evaluator: Evaluator,
    pop_size: int = 20,
    max_iter: int = 30,
    size_penalty: float = 0.02,
) -> List[str]:
    n = X_train.shape[1]

    population = np.random.randint(0, 2, (pop_size, n))
    scores = np.array([mask_fitness(ind, evaluator, n, size_penalty) for ind in population])

    best = population[np.argmax(scores)].copy()
    best_score = float(np.max(scores))
    LOGGER.info("FOA-GWO initial fitness %.5f", best_score)

    for iteration in range(max_iter):

        # ---- FOA phase: explore -------------------------------------
        seeds = []
        for tree in population:
            seed = tree.copy()
            index = np.random.randint(n)
            seed[index] = 1 - seed[index]
            seeds.append(seed)

        combined = np.vstack((population, np.array(seeds)))
        combined_scores = np.array(
            [mask_fitness(ind, evaluator, n, size_penalty) for ind in combined]
        )

        survivors = np.argsort(combined_scores)[-pop_size:]
        population = combined[survivors]
        scores = combined_scores[survivors]

        # ---- GWO phase: exploit -------------------------------------
        order = np.argsort(scores)
        alpha = population[order[-1]]
        beta = population[order[-2]]
        delta = population[order[-3]]

        a = 2 - iteration * (2 / max_iter)

        for i in range(pop_size):
            wolf = population[i].copy()

            positions = []
            for leader in (alpha, beta, delta):
                r1, r2 = np.random.rand(2)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - wolf)
                positions.append(leader - A * D)

            new_position = sum(positions) / 3.0
            population[i] = (_sigmoid(new_position) > np.random.rand(n)).astype(int)

        scores = np.array(
            [mask_fitness(ind, evaluator, n, size_penalty) for ind in population]
        )

        if np.max(scores) > best_score:
            best_score = float(np.max(scores))
            best = population[np.argmax(scores)].copy()

        LOGGER.info(
            "FOA-GWO iter %d/%d | a=%.3f | fitness %.5f | features %d",
            iteration + 1, max_iter, a, best_score, int(best.sum()),
        )

    LOGGER.info("FOA-GWO complete | fitness %.5f | features %d", best_score, int(best.sum()))
    return list(X_train.columns[np.where(best == 1)[0]])
