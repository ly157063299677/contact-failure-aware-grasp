"""Backend contracts and timing shared by offline labeling and online refinement.

No physical solver is substituted here. A project-specific adapter must bind the
native GraspADMM objective, its assets and its joint-to-actuator mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Protocol, TypeVar

import numpy as np

T = TypeVar("T")


class DeadlineExceeded(TimeoutError):
    """An online operation completed too late, or cannot be started in budget."""


class NumericalFailure(RuntimeError):
    """An explicitly identified physics/solver numerical failure, not a code bug."""


class Deadline:
    def __init__(self, seconds: float | None = None, *, clock: Callable[[], float] = perf_counter):
        if seconds is not None and (not np.isfinite(seconds) or seconds <= 0):
            raise ValueError("deadline seconds must be finite and positive")
        self.clock = clock
        self.started = clock()
        self.seconds = seconds

    @property
    def elapsed(self) -> float:
        return self.clock() - self.started

    @property
    def exhausted(self) -> bool:
        return self.seconds is not None and self.elapsed >= self.seconds

    def check(self) -> None:
        if self.exhausted:
            raise DeadlineExceeded("online wall-clock deadline reached")

    def call(self, function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        self.check()
        value = function(*args, **kwargs)
        self.check()
        return value


@dataclass(frozen=True)
class SimulationConfig:
    timestep: float = 0.004
    diagnosis_seconds: float = 0.2
    settle_seconds: float = 0.1
    validation_seconds: float = 2.0
    translation_threshold: float = 0.05
    rotation_threshold_deg: float = 15.0
    gravity: float = 9.81
    iterations: int = 50
    repeats: int = 5
    velocity_linear: float = 0.01
    velocity_angular: float = 0.05

    def __post_init__(self) -> None:
        for name in ("timestep", "diagnosis_seconds", "settle_seconds", "validation_seconds",
                     "translation_threshold", "rotation_threshold_deg", "gravity"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.iterations <= 0 or self.repeats <= 0:
            raise ValueError("iterations and repeats must be positive")
        if (not np.isfinite(self.velocity_linear) or not np.isfinite(self.velocity_angular)
                or self.velocity_linear < 0 or self.velocity_angular < 0):
            raise ValueError("velocity perturbation bounds must be nonnegative")


@dataclass(frozen=True)
class OnlineConfig:
    budget_seconds: float = 5.0


@dataclass(frozen=True)
class RefinementOutcome:
    status: str  # 'ok' or an adapter-specific explicitly identified terminal failure
    iterations: int
    message: str = ""


class ValidationEnvironment(Protocol):
    """A detached FREE-object environment, with controls/palm targets held fixed.

    snapshot/restore must include controls, velocities, activation, warm starts,
    applied forces, mocap/equality state, plugin/user state and gravity settings.
    Object pose uses world position and a normalized wxyz quaternion. Perturbation
    velocities are component-wise in world coordinates. Finger coordinates include
    every finger joint, even when its actuator was not selected.
    """
    timestep: float
    object_mass: float

    def snapshot(self) -> Any: ...
    def restore(self, snapshot: Any) -> None: ...
    def object_pose(self) -> tuple[np.ndarray, np.ndarray]: ...
    def finger_configuration(self) -> np.ndarray: ...
    def set_velocity_delta(self, linear: np.ndarray, angular: np.ndarray) -> None: ...
    def set_gravity_enabled(self, enabled: bool) -> None: ...
    def set_object_force(self, force_world_at_com: np.ndarray) -> None: ...
    def clear_forces(self) -> None: ...
    def step(self) -> None: ...


class GraspBackend(Protocol):
    """Asset-specific adapter required from the experiment authors.

    snapshot/restore include ALL mutable simulation and optimizer state (contact
    variables, duals and native optimizer accumulators). validation_copy must own
    independent model/data and must not mutate this backend or its solver state.
    diagnose must retain initial targets and implement the 50-step contact rollout;
    build_features produces the input schema documented in docs/data_schema.md.
    Both methods must check deadline between internal work units.

    begin_refinement resets contact/dual variables, stores original u0/palm targets,
    disables native recovery and squeezing, and installs Eq. (13) projection at
    EVERY hand target update, including internal simulation substeps. Correction
    forces and contact data remain available for ALL fingers.
    """
    def snapshot(self) -> Any: ...
    def restore(self, snapshot: Any) -> None: ...
    def diagnose(self, config: SimulationConfig, deadline: Deadline) -> Any: ...
    def build_features(self, diagnostics: Any, deadline: Deadline) -> Any: ...
    def finger_configuration(self, snapshot: Any) -> np.ndarray: ...
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]: ...
    def begin_refinement(self, mask: int, reference_snapshot: Any, deadline: Deadline) -> None: ...
    def refine(self, iterations: int, deadline: Deadline) -> RefinementOutcome: ...
    def clear_correction_forces(self) -> None: ...
    def validation_copy(self, snapshot: Any, deadline: Deadline) -> ValidationEnvironment: ...
