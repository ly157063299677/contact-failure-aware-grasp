"""Five-finger subset enumeration and the two comparison families."""
from itertools import combinations
import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "little")
SUBSET_IDS = np.arange(1, 32, dtype=np.int64)
# The integer order is an explicit implementation choice for otherwise tied subsets.
MASKS = ((SUBSET_IDS[:, None] >> np.arange(5)) & 1).astype(np.float32)


def mask_to_bits(subset_id: int) -> np.ndarray:
    if not 1 <= int(subset_id) <= 31:
        raise ValueError("subset_id must be an integer in [1,31]")
    return MASKS[int(subset_id) - 1].astype(bool).copy()


def expansions(subset_id: int) -> list[int]:
    mask_to_bits(subset_id)
    return [subset_id | (1 << finger) for finger in range(5)
            if not subset_id & (1 << finger)]


def pair_indices() -> tuple[np.ndarray, np.ndarray]:
    """Return zero-based [110,2] equal-size and [75,2] expansion indices."""
    equal = [(a - 1, b - 1) for a, b in combinations(range(1, 32), 2)
             if a.bit_count() == b.bit_count()]
    expand = [(a - 1, b - 1) for a in range(1, 32) for b in expansions(a)]
    return np.asarray(equal, dtype=np.int64), np.asarray(expand, dtype=np.int64)


def utilities(success, deviation, alpha=1.0, beta=0.5, gamma=0.1):
    """Eq. (4), with deviation in original, NOT regression-normalized units."""
    return alpha * np.asarray(success) - beta * np.asarray(deviation) - gamma * MASKS.sum(-1) / 5


def best_subset(candidates, utility, probability) -> int:
    candidates = list(candidates)
    if not candidates:
        raise ValueError("No candidate subset")
    utility, probability = np.asarray(utility), np.asarray(probability)
    if utility.shape != (31,) or probability.shape != (31,):
        raise ValueError("Expected one score for each of the 31 subsets")
    if not np.isfinite(utility).all() or not np.isfinite(probability).all():
        raise ValueError("Non-finite prediction; do not silently select a subset")
    return min(candidates, key=lambda sid: (-utility[sid - 1], -probability[sid - 1], sid))
