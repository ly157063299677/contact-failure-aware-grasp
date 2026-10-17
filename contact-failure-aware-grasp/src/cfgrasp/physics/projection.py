"""Equation (13): actuator permissions are not kinematic joint locks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Any

import numpy as np

from .base import Deadline, NumericalFailure, RefinementOutcome


@dataclass
class ActuatorProjector:
    initial: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    finger_groups: tuple[tuple[int, ...], ...]
    mask: int

    def __post_init__(self) -> None:
        self.initial = np.asarray(self.initial, dtype=float).copy()
        self.lower = np.asarray(self.lower, dtype=float).copy()
        self.upper = np.asarray(self.upper, dtype=float).copy()
        if self.initial.ndim != 1 or self.initial.shape != self.lower.shape or self.initial.shape != self.upper.shape:
            raise ValueError("actuator targets and limits must be same-length vectors")
        if len(self.finger_groups) != 5 or not 1 <= self.mask <= 31:
            raise ValueError("expected five finger groups and a nonempty 5-bit mask")
        indices = [j for group in self.finger_groups for j in group]
        if len(indices) != len(set(indices)) or any(j < 0 or j >= len(self.initial) for j in indices):
            raise ValueError("finger actuator groups must be disjoint valid indices")
        if (not np.isfinite(self.initial).all() or not np.isfinite(self.lower).all()
                or not np.isfinite(self.upper).all() or np.any(self.lower > self.upper)):
            raise ValueError("invalid actuator targets/limits")
        if np.any(self.initial < self.lower) or np.any(self.initial > self.upper):
            raise ValueError("initial targets must already satisfy actuator limits")
        self.selected = np.array([j for i, group in enumerate(self.finger_groups)
                                  if self.mask & (1 << i) for j in group], dtype=int)

    def __call__(self, proposed_targets: np.ndarray) -> np.ndarray:
        proposed_targets = np.asarray(proposed_targets, dtype=float)
        if proposed_targets.shape != self.initial.shape:
            raise ValueError("joint-to-actuator mapping returned the wrong shape")
        if not np.isfinite(proposed_targets).all():
            raise NumericalFailure("nonfinite proposed actuator targets")
        projected = self.initial.copy()  # unselected and non-finger targets stay fixed
        projected[self.selected] = np.clip(proposed_targets[self.selected],
                                           self.lower[self.selected], self.upper[self.selected])
        return projected


class NativeADMM(Protocol):
    """Implement by instrumenting native GraspADMM, retaining its QP/ADMM math.

    outer_iteration MUST route every proposed hand target (including substeps)
    through commit_targets before any physics step. It must never apply a target
    directly. It may apply correction forces to any finger. Its return value is
    False only for an identified terminal solver failure. Unexpected errors escape.
    """
    def outer_iteration(self, commit_targets: Callable[[np.ndarray], None], deadline: Deadline) -> bool: ...
    def clear_correction_forces(self) -> None: ...


class ProjectedADMMRunner:
    """Adapter helper; this wrapper does not replace the native physical objective.

    The supplied commit callback writes projected actuator targets AND reasserts
    the saved palm target. Supply an explicit joint-to-actuator mapping upstream;
    slicing generalized joint coordinates is not a valid universal mapping.
    """
    def __init__(self, native: NativeADMM, projector: ActuatorProjector,
                 write_targets_and_fixed_palm: Callable[[np.ndarray], None]):
        self.native = native
        self.projector = projector
        self.write = write_targets_and_fixed_palm

    def run(self, iterations: int, deadline: Deadline) -> RefinementOutcome:
        completed = 0

        def commit(proposed: np.ndarray) -> None:
            deadline.check()
            self.write(self.projector(proposed))
            deadline.check()

        try:
            for _ in range(iterations):
                deadline.check()
                ok = self.native.outer_iteration(commit, deadline)
                deadline.check()
                if not ok:
                    return RefinementOutcome("solver_failure", completed)
                completed += 1
            return RefinementOutcome("ok", completed)
        except NumericalFailure as exc:
            return RefinementOutcome("numerical_failure", completed, str(exc))
        finally:
            # Cleanup always happens, including on timeout; it remains wall-clock charged.
            self.native.clear_correction_forces()
