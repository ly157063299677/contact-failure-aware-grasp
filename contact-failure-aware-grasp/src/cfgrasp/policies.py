"""Subset policies. All indices refer to Thumb, Index, Middle, Ring, Little."""
import numpy as np
from .subsets import best_subset, expansions, utilities


class LearnedPolicy:
    def __init__(self, probability, deviation, **weights):
        self.probability = np.asarray(probability)
        self.utility = utilities(probability, deviation, **weights)

    def initial(self):
        return best_subset(range(1, 32), self.utility, self.probability)

    def expand(self, mask):
        candidates = expansions(mask)
        return best_subset(candidates, self.utility, self.probability) if candidates else None


class RandomPolicy:
    def __init__(self, rng):
        self.rng = rng

    def initial(self):
        return int(self.rng.integers(1, 32))

    def expand(self, mask):
        candidates = expansions(mask)
        return int(self.rng.choice(candidates)) if candidates else None


class OrderedPolicy:
    def __init__(self, order, initial_count=2):
        if sorted(order) != list(range(5)) or not 1 <= initial_count <= 5:
            raise ValueError("Invalid five-finger order or initial_count")
        self.order, self.initial_count = list(order), initial_count

    def initial(self):
        return sum(1 << finger for finger in self.order[:self.initial_count])

    def expand(self, mask):
        for finger in self.order:
            if not mask & (1 << finger):
                return mask | (1 << finger)
        return None


def contact_policy(diagnostics):
    """Features documented in docs/features.md; unavailable speed ranks as zero."""
    diagnostics = np.asarray(diagnostics)
    missing = diagnostics[:, 1]
    speed = np.where(diagnostics[:, 11] > 0, diagnostics[:, 10], 0.0)
    return OrderedPolicy(sorted(range(5), key=lambda i: (-missing[i], -speed[i], i)))


def singleton_policy(probability, deviation):
    score = np.asarray(probability) - 0.5 * np.asarray(deviation)
    if score.shape == (31,):
        score = score[[0, 1, 3, 7, 15]]
    if score.shape != (5,) or not np.isfinite(score).all():
        raise ValueError("Top-k requires finite singleton scores in anatomical finger order")
    return OrderedPolicy(sorted(range(5), key=lambda i: (-score[i], i)))
