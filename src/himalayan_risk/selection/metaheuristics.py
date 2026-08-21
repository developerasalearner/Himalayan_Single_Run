"""Population-based feature selection.

Each optimiser searches the same space: binary masks of length n, where
a one retains the corresponding feature. Every mask is scored by the
shared fitness of Equation 5 - a recall-weighted composite minus a levy
per retained feature - using the small XGBoost surrogate built in
`modelling.make_evaluator`.

These are faithful reimplementations of the original notebook's
optimisers, with two changes: the random streams are seeded (see
`seeds.set_seeds`), and per-iteration logging replaced bare prints.
"""

from __future__ import annotations

import logging
import random
from typing import Callable, List

import numpy as np
import pandas as pd

from ..modelling import mask_fitness

LOGGER = logging.getLogger(__name__)

Evaluator = Callable[[np.ndarray], float]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _to_features(X: pd.DataFrame, mask: np.ndarray) -> List[str]:
    return list(X.columns[np.where(mask == 1)[0]])


def GA_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    evaluator: Evaluator,
    pop_size: int = 20,
    generations: int = 30,
    size_penalty: float = 0.02,
    mutation_rate: float = 0.02,
) -> List[str]:
    """Genetic algorithm: single-point crossover with elitism."""
    n = X_train.shape[1]
    population = np.random.randint(0, 2, (pop_size, n))
    scores = np.array([mask_fitness(ind, evaluator, n, size_penalty) for ind in population])

    best = population[np.argmax(scores)].copy()
    best_score = float(np.max(scores))

    for generation in range(generations):
        offspring = [best.copy()]

        while len(offspring) < pop_size:
            p1, p2 = random.sample(list(population), 2)
            point = np.random.randint(1, max(n - 1, 2))
            child = np.concatenate([p1[:point], p2[point:]])
            flips = np.random.rand(n) < mutation_rate
            child[flips] = 1 - child[flips]
            offspring.append(child)

        population = np.array(offspring)
        scores = np.array([mask_fitness(ind, evaluator, n, size_penalty) for ind in population])

        if np.max(scores) > best_score:
            best_score = float(np.max(scores))
            best = population[np.argmax(scores)].copy()

        LOGGER.info("GA gen %d/%d | fitness %.4f | features %d",
                    generation + 1, generations, best_score, int(best.sum()))

    return _to_features(X_train, best)


def PSO_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    evaluator: Evaluator,
    swarm_size: int = 20,
    max_iter: int = 30,
    size_penalty: float = 0.02,
    inertia: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
) -> List[str]:
    """Particle swarm with a sigmoid map from velocity to bit probability.

    The continuous velocity update maps awkwardly onto a binary mask,
    which is why this optimiser tends to underperform here.
    """
    n = X_train.shape[1]
    particles = np.random.randint(0, 2, (swarm_size, n))
    velocities = np.random.uniform(-1, 1, (swarm_size, n))

    pbest = particles.copy()
    pbest_scores = np.array([mask_fitness(p, evaluator, n, size_penalty) for p in particles])

    gbest = pbest[np.argmax(pbest_scores)].copy()
    gbest_score = float(np.max(pbest_scores))

    for iteration in range(max_iter):
        for i in range(swarm_size):
            r1, r2 = np.random.rand(n), np.random.rand(n)
            velocities[i] = (
                inertia * velocities[i]
                + c1 * r1 * (pbest[i] - particles[i])
                + c2 * r2 * (gbest - particles[i])
            )
            particles[i] = (_sigmoid(velocities[i]) > np.random.rand(n)).astype(int)

            score = mask_fitness(particles[i], evaluator, n, size_penalty)
            if score > pbest_scores[i]:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = score
                if score > gbest_score:
                    gbest = particles[i].copy()
                    gbest_score = float(score)

        LOGGER.info("PSO iter %d/%d | fitness %.4f | features %d",
                    iteration + 1, max_iter, gbest_score, int(gbest.sum()))

    return _to_features(X_train, gbest)


def GWO_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    evaluator: Evaluator,
    pack_size: int = 20,
    max_iter: int = 30,
    size_penalty: float = 0.02,
) -> List[str]:
    """Grey wolf optimiser, standalone.

    The pack is redrawn each iteration from the alpha, beta and delta
    leaders. This converges fast but explores little, which is the
    weakness the hybrid addresses.
    """
    n = X_train.shape[1]
    wolves = np.random.randint(0, 2, (pack_size, n))
    scores = np.array([mask_fitness(w, evaluator, n, size_penalty) for w in wolves])

    best = wolves[np.argmax(scores)].copy()
    best_score = float(np.max(scores))

    for iteration in range(max_iter):
        order = np.argsort(scores)
        alpha, beta, delta = wolves[order[-1]], wolves[order[-2]], wolves[order[-3]]

        centroid = (alpha + beta + delta) / 3.0
        probability = _sigmoid(centroid)

        for i in range(pack_size):
            wolves[i] = (probability > np.random.rand(n)).astype(int)

        scores = np.array([mask_fitness(w, evaluator, n, size_penalty) for w in wolves])
        if np.max(scores) > best_score:
            best_score = float(np.max(scores))
            best = wolves[np.argmax(scores)].copy()

        LOGGER.info("GWO iter %d/%d | fitness %.4f | features %d",
                    iteration + 1, max_iter, best_score, int(best.sum()))

    return _to_features(X_train, best)


def EVO_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    evaluator: Evaluator,
    pop_size: int = 20,
    max_iter: int = 30,
    size_penalty: float = 0.02,
) -> List[str]:
    """Evolutionary variant driven by the population centroid.

    This optimiser is retained because its failure is informative: it
    tends to collapse onto a tiny mask whose classifier labels nearly
    everything positive, scoring well on the recall and compactness terms
    of the fitness while being useless.
    """
    n = X_train.shape[1]
    population = np.random.randint(0, 2, (pop_size, n))
    scores = np.array([mask_fitness(p, evaluator, n, size_penalty) for p in population])

    best = population[np.argmax(scores)].copy()
    best_score = float(np.max(scores))

    for iteration in range(max_iter):
        centre = np.mean(population, axis=0)

        for i in range(pop_size):
            neighbour = population[random.randint(0, pop_size - 1)]
            tau = np.random.rand(4)

            candidate = population[i] + (tau[0] * best - tau[1] * centre)
            candidate = candidate + (tau[2] * best - tau[3] * neighbour)
            candidate = (_sigmoid(candidate) > 0.5).astype(int)

            score = mask_fitness(candidate, evaluator, n, size_penalty)
            if score > scores[i]:
                population[i] = candidate
                scores[i] = score
                if score > best_score:
                    best = candidate.copy()
                    best_score = float(score)

        LOGGER.info("EVO iter %d/%d | fitness %.4f | features %d",
                    iteration + 1, max_iter, best_score, int(best.sum()))

    return _to_features(X_train, best)


def FOA_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    evaluator: Evaluator,
    pop_size: int = 20,
    max_iter: int = 30,
    size_penalty: float = 0.02,
) -> List[str]:
    """Forest optimisation algorithm: seed, pool, then keep the fittest."""
    n = X_train.shape[1]
    forest = np.random.randint(0, 2, (pop_size, n))
    scores = np.array([mask_fitness(f, evaluator, n, size_penalty) for f in forest])

    best = forest[np.argmax(scores)].copy()
    best_score = float(np.max(scores))

    for iteration in range(max_iter):
        seeds = []
        for tree in forest:
            seed = tree.copy()
            index = np.random.randint(n)
            seed[index] = 1 - seed[index]
            seeds.append(seed)

        combined = np.vstack((forest, np.array(seeds)))
        combined_scores = np.array(
            [mask_fitness(c, evaluator, n, size_penalty) for c in combined]
        )

        survivors = np.argsort(combined_scores)[-pop_size:]
        forest = combined[survivors]
        scores = combined_scores[survivors]

        if np.max(scores) > best_score:
            best_score = float(np.max(scores))
            best = forest[np.argmax(scores)].copy()

        LOGGER.info("FOA iter %d/%d | fitness %.4f | features %d",
                    iteration + 1, max_iter, best_score, int(best.sum()))

    return _to_features(X_train, best)


OPTIMIZERS = {
    "GA": GA_feature_selection,
    "PSO": PSO_feature_selection,
    "GWO": GWO_feature_selection,
    "EVO": EVO_feature_selection,
    "FOA": FOA_feature_selection,
}
