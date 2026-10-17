"""Independent subset optimization and repeated physical validation (Eq. 5)."""
from __future__ import annotations
import hashlib
import numpy as np
from .data import validate_features
from .physics import (Deadline, SimulationConfig, perturbation_bank,
                      refine_candidate, repeated_validation, validation_suite)
from .subsets import MASKS


def state_rng(seed: int, state_id: str, stream: str) -> np.random.Generator:
    """Stable independent streams; independent of manifest ordering and Python hash salt."""
    digest = hashlib.sha256(f"{seed}\0{state_id}\0{stream}".encode("utf-8")).digest()
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest, dtype="<u4")))


def generate_state_labels(backend, config: SimulationConfig, rng):
    if config.repeats != 5:
        raise ValueError("The release data schema and manuscript protocol require five repetitions")
    original = backend.snapshot()
    deadline = Deadline()  # Offline label generation is NOT the online deadline.
    try:
        screen_env = backend.validation_copy(original, deadline)
        screen = validation_suite(screen_env, screen_env.snapshot(), np.zeros(6), config)
        if screen.status != "complete":
            raise RuntimeError("Initial screening was incomplete; do not invent the failed-state flag")
        backend.restore(original)
        diagnostics = backend.diagnose(config, deadline)
        backend.restore(original)
        features = backend.build_features(diagnostics, deadline)
        validate_features(features)
        # Bank shared by ALL subsets from this same input state, but not by online/final evaluation.
        bank = perturbation_bank(rng, config.repeats, config)
        directional = np.full((31, config.repeats, 6), -1, dtype=np.int8)
        success = np.zeros(31, dtype=np.float32)
        deviation = np.full(31, np.nan, dtype=np.float32)
        repetition = np.full((31, config.repeats), -1, dtype=np.int8)
        status, validation_status, iterations = [], [], []
        for sid in range(1, 32):
            outcome = refine_candidate(backend, original, sid, config)
            status.append(outcome.status)
            iterations.append(outcome.iterations)
            if outcome.candidate is None:
                # Terminal optimization/settling failures get repair target zero.
                # All unexecuted direction/repetition entries remain -1.
                validation_status.append(["not_executed"] * config.repeats)
                continue
            deviation[sid - 1] = outcome.candidate.deviation
            results = repeated_validation(outcome.candidate, bank, config)
            validation_status.append([result.status for result in results])
            for repeat, result in enumerate(results):
                if result.deadline_exhausted:
                    raise RuntimeError("Unexpected offline deadline interruption")
                directional[sid - 1, repeat] = result.directional
                # Numerical validation failure is an unsuccessful suite, with its status retained.
                repetition[sid - 1, repeat] = int(result.passed)
            success[sid - 1] = repetition[sid - 1].mean()
        return features, {"masks": MASKS.copy(), "success": success, "deviation": deviation,
                          "directional": directional, "repetition": repetition,
                          "solver_status": np.asarray(status), "validation_status": np.asarray(validation_status),
                          "iterations": np.asarray(iterations), "perturbation_bank": bank,
                          "initial_screen": screen.directional}, not screen.passed
    finally:
        backend.clear_correction_forces()
        backend.restore(original)
