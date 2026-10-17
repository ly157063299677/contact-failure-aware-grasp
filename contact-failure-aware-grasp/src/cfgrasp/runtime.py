"""Deadline-aware subset expansion and full-hand continuation (Tables S3-S4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from .physics.base import Deadline, DeadlineExceeded, GraspBackend, NumericalFailure, OnlineConfig, SimulationConfig
from .physics.validation import Candidate, ValidationResult, perturbation_bank, prepare_candidate, repeated_validation, validation_suite


class SubsetPolicy(Protocol):
    def initial(self) -> int: ...
    def expand(self, current_mask: int) -> int | None: ...


PolicyFactory = Callable[[Any, Any, Deadline], SubsetPolicy]


@dataclass
class AttemptRecord:
    mask: int
    status: str = "started"
    iterations: int = 0
    optimization_seconds: float = 0.0
    settling_seconds: float = 0.0
    validation_seconds: float = 0.0
    directional: list[int] = field(default_factory=lambda: [-1] * 6)
    deviation: float | None = None
    message: str = ""


@dataclass
class RuntimeResult:
    accepted: Candidate | None
    selected_mask: int | None
    attempts: list[AttemptRecord]
    elapsed_seconds: float
    budget_exhausted: bool
    status: str

    @property
    def candidate(self) -> Candidate | None:
        return self.accepted

    def to_record(self) -> dict[str, Any]:
        """JSON-compatible metrics; opaque simulator snapshots are never serialized."""
        return {
            "accepted": self.accepted is not None,
            "online_seconds": self.elapsed_seconds,
            "deviation": None if self.accepted is None else self.accepted.deviation,
            "selected_groups": None if self.selected_mask is None else self.selected_mask.bit_count(),
            "selected_mask": self.selected_mask,
            "budget_exhausted": self.budget_exhausted,
            "attempts": len(self.attempts),
            "optimization_seconds": sum(item.optimization_seconds for item in self.attempts),
            "validation_seconds": sum(item.validation_seconds for item in self.attempts),
            "status": self.status,
            "attempt_records": [vars(item).copy() for item in self.attempts],
        }


def _checked_mask(mask: int) -> int:
    if isinstance(mask, bool) or not isinstance(mask, (int, np.integer)) or not 1 <= mask <= 31:
        raise ValueError("policy must return a nonempty integer five-finger mask")
    return int(mask)


def _checkpoint_validation(backend: GraspBackend, original: Any, record: AttemptRecord,
                           config: SimulationConfig, rng: np.random.Generator,
                           deadline: Deadline) -> Candidate | None:
    started = deadline.clock()
    try:
        try:
            prepared = prepare_candidate(backend, original, config, deadline)
        except NumericalFailure as exc:
            record.status, record.message = "numerical_failure", str(exc)
            return None
    finally:
        record.settling_seconds = deadline.clock() - started
    if prepared.candidate is None:
        record.status, record.message = prepared.status, prepared.message
        return None
    candidate = prepared.candidate
    record.deviation = candidate.deviation
    started = deadline.clock()
    try:
        perturbation = deadline.call(perturbation_bank, rng, 1, config)[0]
        suite = validation_suite(candidate.environment, candidate.snapshot, perturbation, config, deadline)
        record.directional = suite.directional.tolist()
        record.status = ("accepted" if suite.passed else "deadline" if suite.deadline_exhausted
                         else "numerical_failure" if suite.status == "numerical_failure" else "validation_failure")
        deadline.check()
    finally:
        record.validation_seconds = deadline.clock() - started
    return candidate if suite.passed else None


def run_subset_search(backend: GraspBackend, policy_factory: PolicyFactory,
                      simulation: SimulationConfig = SimulationConfig(),
                      online: OnlineConfig = OnlineConfig(), *,
                      rng: np.random.Generator | None = None,
                      synchronize: Callable[[], None] | None = None,
                      deadline: Deadline | None = None) -> RuntimeResult:
    """Cache policy predictions once, then restart one-finger expansions from z0.

    policy_factory(features, diagnostics, deadline) computes/caches all needed
    predictions; proposed/MLP policies must cache all 31 utilities. It is executed
    once inside the deadline. synchronize must be torch.cuda.synchronize (for the
    inference device) when inference uses CUDA; calls before and after inference
    include GPU work and transfers in the same wall-clock measurement.
    """
    deadline = deadline or Deadline(online.budget_seconds)
    rng = rng if rng is not None else np.random.default_rng()
    attempts: list[AttemptRecord] = []
    accepted = None
    selected = None
    original = None
    restored_on_exit = False
    status = "no_accepted_output"
    try:
        # The timer already runs: capture, diagnosis, feature extraction and restore
        # are charged, although asset initialization happens before this function.
        original = deadline.call(backend.snapshot)
        diagnostics = deadline.call(backend.diagnose, simulation, deadline)
        deadline.call(backend.restore, original)
        features = deadline.call(backend.build_features, diagnostics, deadline)
        if synchronize is not None:
            deadline.call(synchronize)
        try:
            policy = deadline.call(policy_factory, features, diagnostics, deadline)
        finally:
            if synchronize is not None:
                synchronize()
        deadline.check()
        mask = _checked_mask(deadline.call(policy.initial))
        while True:
            record = AttemptRecord(mask)
            attempts.append(record)
            started = deadline.clock()
            try:
                deadline.call(backend.restore, original)
                deadline.call(backend.begin_refinement, mask, original, deadline)
                outcome = deadline.call(backend.refine, simulation.iterations, deadline)
                record.iterations, record.status, record.message = outcome.iterations, outcome.status, outcome.message
            except NumericalFailure as exc:
                record.status, record.message = "numerical_failure", str(exc)
            finally:
                backend.clear_correction_forces()
                record.optimization_seconds = deadline.clock() - started
            deadline.check()
            if record.status == "ok":
                candidate = _checkpoint_validation(backend, original, record, simulation, rng, deadline)
                if candidate is not None:
                    accepted, selected, status = candidate, mask, "accepted"
                    break
            if mask == 31:
                break  # full hand is attempted at most once in subset search
            next_mask = deadline.call(policy.expand, mask)
            if next_mask is None:
                break
            next_mask = _checked_mask(next_mask)
            if mask & next_mask != mask or next_mask.bit_count() != mask.bit_count() + 1:
                raise ValueError("expansion must add exactly one previously unselected finger")
            mask = next_mask
    except DeadlineExceeded:
        status = "deadline"
        if attempts:
            attempts[-1].status = "deadline"
    except NumericalFailure as exc:
        status = "diagnosis_or_transfer_numerical_failure"
        if attempts:
            attempts[-1].status, attempts[-1].message = "numerical_failure", str(exc)
    finally:
        # Never leave optimizer forces active, even after a failed internal call.
        try:
            backend.clear_correction_forces()
        finally:
            if original is not None and (accepted is None or deadline.exhausted):
                backend.restore(original)
                restored_on_exit = True
    elapsed = deadline.elapsed
    exhausted = deadline.seconds is not None and elapsed >= deadline.seconds
    if exhausted:
        # This branch also handles a clock boundary crossed immediately after
        # cleanup. Restore before reporting the rejected late output.
        if accepted is not None and not restored_on_exit:
            backend.restore(original)
            elapsed = deadline.elapsed
        accepted, selected, status = None, None, "deadline"
        if attempts:
            attempts[-1].status = "deadline"
    return RuntimeResult(accepted, selected, attempts, elapsed, exhausted, status)


def run_full_hand(backend: GraspBackend, simulation: SimulationConfig = SimulationConfig(),
                  online: OnlineConfig = OnlineConfig(), *, rng: np.random.Generator | None = None,
                  deadline: Deadline | None = None) -> RuntimeResult:
    """Continue the SAME native solver in blocks of 50 outer iterations.

    Validation always uses detached copies. Contact/dual/optimizer state is never
    reinitialized between segments. Terminal numerical failure ends this baseline.
    """
    deadline = deadline or Deadline(online.budget_seconds)
    rng = rng if rng is not None else np.random.default_rng()
    attempts: list[AttemptRecord] = []
    accepted = None
    original = None
    restored_on_exit = False
    status = "no_accepted_output"
    try:
        original = deadline.call(backend.snapshot)
        deadline.call(backend.diagnose, simulation, deadline)
        deadline.call(backend.restore, original)
        deadline.call(backend.begin_refinement, 31, original, deadline)
        while True:
            deadline.check()
            record = AttemptRecord(31)
            attempts.append(record)
            started = deadline.clock()
            try:
                outcome = deadline.call(backend.refine, simulation.iterations, deadline)
                record.status, record.iterations, record.message = outcome.status, outcome.iterations, outcome.message
            except NumericalFailure as exc:
                record.status, record.message = "numerical_failure", str(exc)
            finally:
                backend.clear_correction_forces()
                record.optimization_seconds = deadline.clock() - started
            deadline.check()
            if record.status != "ok":
                break
            accepted = _checkpoint_validation(backend, original, record, simulation, rng, deadline)
            if accepted is not None:
                status = "accepted"
                break
    except DeadlineExceeded:
        status = "deadline"
        if attempts:
            attempts[-1].status = "deadline"
    except NumericalFailure as exc:
        status = "diagnosis_or_transfer_numerical_failure"
        if attempts:
            attempts[-1].status, attempts[-1].message = "numerical_failure", str(exc)
    finally:
        try:
            backend.clear_correction_forces()
        finally:
            if original is not None and (accepted is None or deadline.exhausted):
                backend.restore(original)
                restored_on_exit = True
    elapsed = deadline.elapsed
    exhausted = deadline.seconds is not None and elapsed >= deadline.seconds
    if exhausted:
        if accepted is not None and not restored_on_exit:
            backend.restore(original)
            elapsed = deadline.elapsed
        accepted, status = None, "deadline"
        if attempts:
            attempts[-1].status = "deadline"
    return RuntimeResult(accepted, 31 if accepted is not None else None,
                         attempts, elapsed, exhausted, status)


def independent_final_validation(result: RuntimeResult, simulation: SimulationConfig,
                                 rng: np.random.Generator) -> list[ValidationResult]:
    """Five fresh suites on saved accepted state; excluded from online latency.

    Use an RNG stream independent of offline banks and online validation. No
    accepted output gives no trials; its repair credit is zero at aggregation.
    """
    if result.accepted is None:
        return []
    return repeated_validation(result.accepted, perturbation_bank(rng, simulation.repeats, simulation), simulation)
