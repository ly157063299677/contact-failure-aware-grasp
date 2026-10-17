"""State-wise soft outcome and utility-comparison supervision (Eqs. 10--12)."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .subsets import MASKS, pair_indices


def estimate_deviation_scale(dataset: Iterable[Mapping[str, Any]]) -> float:
    """Population std of every finite *training* deviation, floored at 1e-3.

    Use only the training partition. Failed validations with a finite deviation
    are included; targets are never conditioned on success. Welford merging
    avoids loading the complete training-label array into memory. The reference
    implementation chooses population std (ddof=0), which the paper does not
    otherwise specify. Absence of any finite training target is an input error.
    """
    count = 0
    mean = 0.0
    second_moment = 0.0
    for record in dataset:
        values = torch.as_tensor(record["deviation"], dtype=torch.float64).detach().cpu()
        values = values[torch.isfinite(values)]
        n = values.numel()
        if n == 0:
            continue
        batch_mean = float(values.mean())
        batch_second = float(((values - batch_mean) ** 2).sum())
        delta = batch_mean - mean
        total = count + n
        second_moment += batch_second + delta * delta * count * n / total
        mean += delta * n / total
        count = total
    if count == 0:
        raise ValueError("Training partition has no finite configuration-deviation targets")
    return max((max(second_moment, 0.0) / count) ** 0.5, 1e-3)


class OutcomeComparisonLoss(nn.Module):
    """Predict absolute outcomes and relative utility from the same two heads.

    Pair indices always refer to two subsets of the same state. Equal-size and
    expansion families each receive one-half weight, even if their retained
    pair counts differ. An empty family contributes zero. No success threshold
    is applied to the empirical-frequency targets.
    """

    def __init__(self, sd: float, lambda_d: float = 1.0, lambda_pair: float = 0.5,
                 temperature: float = 0.1, alpha: float = 1.0,
                 beta: float = 0.5, gamma: float = 0.1) -> None:
        super().__init__()
        if not all(math.isfinite(value) for value in
                   (sd, lambda_d, lambda_pair, temperature, alpha, beta, gamma)):
            raise ValueError("Loss scales and weights must be finite")
        if not (sd >= 1e-3 and temperature > 0):
            raise ValueError("sd must be at least 1e-3 and temperature must be positive")
        if alpha <= 0 or beta < 0 or gamma < 0 or lambda_d < 0 or lambda_pair < 0:
            raise ValueError("Utility and loss weights must satisfy the paper's sign constraints")
        self.sd = float(sd)
        self.lambda_d, self.lambda_pair = float(lambda_d), float(lambda_pair)
        self.temperature = float(temperature)
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        equal, expand = pair_indices()
        self.register_buffer("equal_pairs", torch.as_tensor(equal.copy(), dtype=torch.long))
        self.register_buffer("expansion_pairs", torch.as_tensor(expand.copy(), dtype=torch.long))
        self.register_buffer("group_fraction", torch.as_tensor(MASKS.sum(axis=1) / 5, dtype=torch.float32))
        if self.equal_pairs.shape != (110, 2) or self.expansion_pairs.shape != (75, 2):
            raise ValueError("Expected 110 equal-size and 75 one-finger expansion pairs")

    def _pair_loss(self, predicted: Tensor, observed: Tensor, valid: Tensor,
                   pairs: Tensor) -> tuple[Tensor, Tensor]:
        first, second = pairs[:, 0], pairs[:, 1]
        retained = valid[:, first] & valid[:, second]
        count = retained.sum()
        if not bool(count):
            return predicted.sum() * 0.0, count
        prediction_logits = ((predicted[:, first] - predicted[:, second]) /
                             self.temperature)[retained]
        # Safe observed utilities have already replaced unavailable values.
        target = (((observed[:, first] - observed[:, second]) /
                   self.temperature)[retained]).sigmoid().detach()
        return F.binary_cross_entropy_with_logits(prediction_logits, target), count

    def forward(self, prediction: Mapping[str, Tensor], success: Tensor,
                deviation: Tensor) -> dict[str, Tensor]:
        logits = prediction["success_logits"]
        p = prediction["success"]
        d = prediction["deviation"]
        if logits.ndim != 2 or logits.shape[1] != 31:
            raise ValueError("Predictions must have shape [states, 31]")
        if any(value.shape != logits.shape for value in (p, d, success, deviation)):
            raise ValueError("Prediction and label shapes must match")
        # Labels remain observations: gradients must never enter the targets.
        success, deviation = success.detach(), deviation.detach()
        valid_success = torch.isfinite(success)
        valid_deviation = torch.isfinite(deviation)
        if bool(((success[valid_success] < 0) | (success[valid_success] > 1)).any()):
            raise ValueError("Finite success targets must be empirical frequencies in [0,1]")
        if bool((deviation[valid_deviation] < 0).any()):
            raise ValueError("Finite normalized deviations must be nonnegative")
        n_success, n_deviation = valid_success.sum(), valid_deviation.sum()
        # Index BEFORE subtraction / BCE. Multiplying NaN losses by zero would
        # still contaminate gradients from the regression branch.
        bce = (F.binary_cross_entropy_with_logits(logits[valid_success], success[valid_success])
               if bool(n_success) else logits.sum() * 0.0)
        huber = (F.huber_loss(
            (d[valid_deviation] - deviation[valid_deviation]) / self.sd,
            torch.zeros_like(d[valid_deviation]), delta=1.0,
        ) if bool(n_deviation) else d.sum() * 0.0)
        predicted_utility = self.alpha * p - self.beta * d - self.gamma * self.group_fraction
        safe_success = torch.where(valid_success, success, torch.zeros_like(success))
        safe_deviation = torch.where(valid_deviation, deviation, torch.zeros_like(deviation))
        observed_utility = (self.alpha * safe_success - self.beta * safe_deviation -
                            self.gamma * self.group_fraction).detach()
        valid_pair = valid_success & valid_deviation
        equal, n_equal = self._pair_loss(
            predicted_utility, observed_utility, valid_pair, self.equal_pairs
        )
        expansion, n_expansion = self._pair_loss(
            predicted_utility, observed_utility, valid_pair, self.expansion_pairs
        )
        comparison = 0.5 * equal + 0.5 * expansion
        outcome = bce + self.lambda_d * huber
        total = outcome + self.lambda_pair * comparison
        return {
            "loss": total, "outcome": outcome, "success_bce": bce,
            "deviation_huber": huber, "comparison": comparison,
            "equal_pair": equal, "expansion_pair": expansion,
            "n_success": n_success, "n_deviation": n_deviation,
            "n_equal_pair": n_equal, "n_expansion_pair": n_expansion,
        }


def loss_for_batch(prediction: Mapping[str, Tensor], batch: Mapping[str, Tensor],
                   criterion: OutcomeComparisonLoss) -> dict[str, Tensor]:
    return criterion(prediction, batch["success"], batch["deviation"])
