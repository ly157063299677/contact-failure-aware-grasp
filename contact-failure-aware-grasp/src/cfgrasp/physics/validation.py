"""Free-object validation, shared perturbation banks and candidate settling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import Deadline, DeadlineExceeded, GraspBackend, NumericalFailure, SimulationConfig, ValidationEnvironment

DIRECTIONS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
DIRECTION_NAMES = ("+x", "-x", "+y", "-y", "+z", "-z")


def step_count(seconds: float, timestep: float) -> int:
    count = round(seconds / timestep)
    if not np.isclose(count * timestep, seconds, rtol=0, atol=1e-10):
        raise ValueError("simulation duration must be an integer multiple of timestep")
    return count


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError("orientations must be wxyz quaternions")
    norms = np.linalg.norm(first), np.linalg.norm(second)
    if not np.isfinite(first).all() or not np.isfinite(second).all() or min(norms) <= 0:
        raise NumericalFailure("invalid object quaternion")
    dot = abs(float(np.dot(first / norms[0], second / norms[1])))
    return float(np.degrees(2 * np.arccos(np.clip(dot, 0, 1))))


def normalized_deviation(q: np.ndarray, q0: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    q, q0, lower, upper = map(lambda value: np.asarray(value, float), (q, q0, lower, upper))
    if q.ndim != 1 or q.size == 0 or not (q.shape == q0.shape == lower.shape == upper.shape):
        raise ValueError("all finger-joint vectors must have the same nonzero size")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper <= lower):
        raise ValueError("joint ranges must be finite and strictly positive")
    if not np.isfinite(q).all() or not np.isfinite(q0).all():
        return float("nan")
    return float(np.mean(np.abs(q - q0) / (upper - lower)))


def perturbation_bank(rng: np.random.Generator, repeats: int, config: SimulationConfig) -> np.ndarray:
    """Draw once per initial state for offline comparisons; reuse for all 31 masks."""
    return np.concatenate((rng.uniform(-config.velocity_linear, config.velocity_linear, (repeats, 3)),
                           rng.uniform(-config.velocity_angular, config.velocity_angular, (repeats, 3))), axis=1)


def _inside_limits(env: ValidationEnvironment, initial_pose: tuple[np.ndarray, np.ndarray], config: SimulationConfig) -> bool:
    position, quaternion = env.object_pose()
    if not np.isfinite(position).all():
        raise NumericalFailure("nonfinite object position")
    return (np.linalg.norm(position - initial_pose[0]) <= config.translation_threshold
            and rotation_distance_deg(initial_pose[1], quaternion) <= config.rotation_threshold_deg)


def settle(env: ValidationEnvironment, config: SimulationConfig, deadline: Deadline) -> bool:
    """No artificial correction forces; actuator and palm targets are held fixed.

    The manuscript does not specify separate settling displacement thresholds; the
    configured validation translation/rotation limits are used explicitly here.
    """
    if not np.isclose(env.timestep, config.timestep):
        raise ValueError("validation environment timestep differs from protocol")
    deadline.call(env.clear_forces)
    origin = deadline.call(env.object_pose)
    for _ in range(step_count(config.settle_seconds, config.timestep)):
        deadline.call(env.step)
        if not deadline.call(_inside_limits, env, origin, config):
            return False
    return True


@dataclass
class ValidationResult:
    directional: np.ndarray  # int8[6]: 1 pass, 0 fail, -1 unexecuted/interrupted
    deadline_exhausted: bool = False
    status: str = "complete"
    elapsed_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return bool(np.all(self.directional == 1) and not self.deadline_exhausted)


def validation_suite(env: ValidationEnvironment, prevalidation_state: Any,
                     perturbation: np.ndarray, config: SimulationConfig,
                     deadline: Deadline | None = None) -> ValidationResult:
    """One six-direction suite, returning the environment to its saved state.

    Each direction restores the SAME perturbed snapshot. Forces are world-axis
    loads of magnitude m*g at the object COM, without built-in gravity. A zero
    perturbation implements the initial screen. A partially executed direction
    remains -1, never becomes a fabricated zero or a pass.
    """
    deadline = deadline or Deadline()
    started = deadline.clock()
    perturbation = np.asarray(perturbation, dtype=float)
    if perturbation.shape != (6,) or not np.isfinite(perturbation).all():
        raise ValueError("velocity perturbation must be a finite six-vector")
    if not np.isclose(env.timestep, config.timestep) or env.object_mass <= 0:
        raise ValueError("invalid environment timestep or object mass")
    results = np.full(6, -1, dtype=np.int8)
    result = ValidationResult(results)
    try:
        deadline.call(env.restore, prevalidation_state)
        deadline.call(env.clear_forces)
        deadline.call(env.set_gravity_enabled, False)
        deadline.call(env.set_velocity_delta, perturbation[:3], perturbation[3:])
        perturbed = deadline.call(env.snapshot)
        origin = deadline.call(env.object_pose)
        for index, direction in enumerate(DIRECTIONS):
            deadline.call(env.restore, perturbed)
            deadline.call(env.clear_forces)
            deadline.call(env.set_object_force, direction * env.object_mass * config.gravity)
            passed = True
            try:
                for _ in range(step_count(config.validation_seconds, config.timestep)):
                    deadline.call(env.step)
                    if not deadline.call(_inside_limits, env, origin, config):
                        passed = False
                        break
            except NumericalFailure:
                # A numerical interruption is not an observed physical failure.
                # Keep -1, but try the remaining independent restored directions.
                result.status = "numerical_failure"
                continue
            results[index] = int(passed)
    except DeadlineExceeded:
        result.deadline_exhausted = True
        result.status = "deadline"
    except NumericalFailure:
        result.status = "numerical_failure"
    finally:
        # Restore even on timeout. This cleanup contributes to elapsed wall time.
        env.clear_forces()
        env.restore(prevalidation_state)
    if deadline.exhausted:
        result.deadline_exhausted = True
        result.status = "deadline"
    result.elapsed_seconds = deadline.clock() - started
    return result


@dataclass
class Candidate:
    environment: ValidationEnvironment
    snapshot: Any
    configuration: np.ndarray
    deviation: float


@dataclass
class CandidateOutcome:
    status: str
    candidate: Candidate | None
    iterations: int = 0
    optimization_seconds: float = 0.0
    settling_seconds: float = 0.0
    message: str = ""


def prepare_candidate(backend: GraspBackend, original_state: Any, config: SimulationConfig,
                      deadline: Deadline) -> CandidateOutcome:
    """Copy a force-free optimized state to the independent free-object model."""
    started = deadline.clock()
    deadline.call(backend.clear_correction_forces)
    refined_state = deadline.call(backend.snapshot)
    env = deadline.call(backend.validation_copy, refined_state, deadline)
    try:
        if not settle(env, config, deadline):
            return CandidateOutcome("settling_failure", None, settling_seconds=deadline.clock() - started)
    except NumericalFailure as exc:
        return CandidateOutcome("settling_failure", None, settling_seconds=deadline.clock() - started, message=str(exc))
    state = deadline.call(env.snapshot)
    q = deadline.call(env.finger_configuration)
    q0 = deadline.call(backend.finger_configuration, original_state)
    lower, upper = deadline.call(backend.joint_limits)
    deviation = deadline.call(normalized_deviation, q, q0, lower, upper)
    if not np.isfinite(deviation):
        return CandidateOutcome("numerical_failure", None, settling_seconds=deadline.clock() - started)
    return CandidateOutcome("ok", Candidate(env, state, q.copy(), deviation),
                            settling_seconds=deadline.clock() - started)


def refine_candidate(backend: GraspBackend, original_state: Any, mask: int,
                     config: SimulationConfig, deadline: Deadline | None = None) -> CandidateOutcome:
    """Independent fixed-budget subset attempt; usable directly by offline labeling.

    Timeout is propagated; explicitly identified numerical failure returns a
    failed outcome. Programming/configuration errors are never hidden.
    """
    deadline = deadline or Deadline()
    started = deadline.clock()
    try:
        deadline.call(backend.restore, original_state)
        deadline.call(backend.begin_refinement, mask, original_state, deadline)
        outcome = deadline.call(backend.refine, config.iterations, deadline)
    except NumericalFailure as exc:
        return CandidateOutcome("numerical_failure", None, optimization_seconds=deadline.clock() - started,
                                message=str(exc))
    finally:
        backend.clear_correction_forces()
    optimization_seconds = deadline.clock() - started
    deadline.check()
    if outcome.status != "ok":
        return CandidateOutcome(outcome.status, None, outcome.iterations, optimization_seconds, message=outcome.message)
    prepared = prepare_candidate(backend, original_state, config, deadline)
    prepared.iterations = outcome.iterations
    prepared.optimization_seconds = optimization_seconds
    return prepared


def repeated_validation(candidate: Candidate, perturbations: np.ndarray, config: SimulationConfig,
                        deadline: Deadline | None = None) -> list[ValidationResult]:
    deadline = deadline or Deadline()
    return [validation_suite(candidate.environment, candidate.snapshot, perturbation, config, deadline)
            for perturbation in perturbations]
