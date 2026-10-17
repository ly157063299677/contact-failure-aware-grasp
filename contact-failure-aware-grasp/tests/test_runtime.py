"""Protocol tests with deterministic fakes. These are NOT physical grasp results."""
from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from cfgrasp.physics import ActuatorProjector, Deadline, NumericalFailure, OnlineConfig, RefinementOutcome, SimulationConfig, validation_suite
from cfgrasp.physics.validation import normalized_deviation, rotation_distance_deg
from cfgrasp.runtime import run_full_hand, run_subset_search


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class ProtocolEnvironment:
    """Synthetic state machine used only to test ordering/restore/deadline logic."""
    timestep = 0.004
    object_mass = 2.0

    def __init__(self, passes=True, clock=None, step_cost=0):
        self.passes = passes
        self.clock, self.step_cost = clock, step_cost
        self.state = {"position": np.zeros(3), "velocity": np.zeros(6), "force": np.zeros(3),
                      "gravity": True, "q": np.ones(5) * 0.02}
        self.direction_starts = []
        self.loads = []

    def snapshot(self):
        return deepcopy(self.state)

    def restore(self, snapshot):
        self.state = deepcopy(snapshot)

    def object_pose(self):
        return self.state["position"].copy(), np.array([1., 0., 0., 0.])

    def finger_configuration(self):
        return self.state["q"].copy()

    def set_velocity_delta(self, linear, angular):
        self.state["velocity"] += np.concatenate((linear, angular))

    def set_gravity_enabled(self, enabled):
        self.state["gravity"] = enabled

    def set_object_force(self, force):
        self.state["force"] = force.copy()
        self.loads.append(force.copy())
        self.direction_starts.append(deepcopy(self.state))

    def clear_forces(self):
        self.state["force"][:] = 0

    def step(self):
        if self.clock:
            self.clock.now += self.step_cost
        if np.any(self.state["force"]) and not self.passes:
            self.state["position"][0] = 0.1


class ProtocolBackend:
    def __init__(self):
        self.state = {"q": np.zeros(5), "segment": 0}
        self.begins = []
        self.segment_history = []
        self.force_clears = 0
        self.features_at = None

    def snapshot(self):
        return deepcopy(self.state)

    def restore(self, snapshot):
        self.state = deepcopy(snapshot)

    def diagnose(self, config, deadline):
        self.state["q"][:] = 99
        return {"diagnostics": True}

    def build_features(self, diagnostics, deadline):
        self.features_at = self.state["q"].copy()
        return {"features": True}

    def finger_configuration(self, snapshot):
        return snapshot["q"].copy()

    def joint_limits(self):
        return -np.ones(5), np.ones(5)

    def begin_refinement(self, mask, reference_snapshot, deadline):
        self.begins.append((mask, self.state["q"].copy()))
        self.state["mask"] = mask
        self.state["segment"] = 0

    def refine(self, iterations, deadline):
        self.state["segment"] += 1
        self.segment_history.append(self.state["segment"])
        self.state["q"] += 0.02
        return RefinementOutcome("ok", iterations)

    def clear_correction_forces(self):
        self.force_clears += 1

    def validation_copy(self, snapshot, deadline):
        # Mask 1 fails; mask 3 succeeds. Full hand succeeds only on its second block.
        passes = snapshot["mask"] == 3 or (snapshot["mask"] == 31 and snapshot["segment"] >= 2)
        return ProtocolEnvironment(passes=passes)


class ExpandingPolicy:
    def initial(self):
        return 1

    def expand(self, current_mask):
        return 3


class ProtocolTests(unittest.TestCase):
    def test_projection_freezes_unselected_and_nonfinger_targets(self):
        projector = ActuatorProjector(np.arange(6) / 10, -np.ones(6), np.ones(6),
                                      ((0,), (1,), (2,), (3,), (4,)), 5)
        actual = projector(np.ones(6) * 4)
        np.testing.assert_allclose(actual, [1, .1, 1, .3, .4, .5])

    def test_all_joints_contribute_to_deviation(self):
        self.assertAlmostEqual(normalized_deviation(np.array([.2, .4]), np.zeros(2),
                                                   np.zeros(2), np.ones(2)), .3)
        self.assertAlmostEqual(rotation_distance_deg(np.array([1., 0, 0, 0]), np.array([-1., 0, 0, 0])), 0)

    def test_same_perturbed_snapshot_six_loads_and_restored_return(self):
        env = ProtocolEnvironment()
        original = env.snapshot()
        delta = np.arange(6) * .001
        result = validation_suite(env, original, delta, SimulationConfig())
        self.assertTrue(result.passed)
        self.assertEqual(len(env.loads), 6)
        for state, force in zip(env.direction_starts, env.loads):
            np.testing.assert_allclose(state["velocity"], delta)
            np.testing.assert_allclose(state["position"], np.zeros(3))
            self.assertFalse(state["gravity"])
            self.assertAlmostEqual(np.linalg.norm(force), 2 * 9.81)
        np.testing.assert_array_equal(env.state["velocity"], original["velocity"])
        self.assertTrue(env.state["gravity"])

    def test_interrupted_test_stays_unexecuted(self):
        clock = FakeClock()
        env = ProtocolEnvironment(clock=clock, step_cost=.1)
        result = validation_suite(env, env.snapshot(), np.zeros(6), SimulationConfig(),
                                  Deadline(.15, clock=clock))
        self.assertTrue(result.deadline_exhausted)
        self.assertFalse(result.passed)
        np.testing.assert_array_equal(result.directional, np.full(6, -1))

    def test_numerical_interruption_is_not_a_physical_failure_label(self):
        class InterruptedEnvironment(ProtocolEnvironment):
            def __init__(self):
                super().__init__()
                self.did_interrupt = False

            def step(self):
                if not self.did_interrupt:
                    self.did_interrupt = True
                    raise NumericalFailure("synthetic interruption")
                super().step()

        env = InterruptedEnvironment()
        result = validation_suite(env, env.snapshot(), np.zeros(6), SimulationConfig())
        self.assertFalse(result.passed)
        self.assertEqual(result.status, "numerical_failure")
        np.testing.assert_array_equal(result.directional, [-1, 1, 1, 1, 1, 1])

    def test_transfer_numerical_failure_allows_subset_expansion(self):
        class FailedTransfer(ProtocolBackend):
            def validation_copy(self, snapshot, deadline):
                if snapshot["mask"] == 1:
                    raise NumericalFailure("synthetic transfer failure")
                return super().validation_copy(snapshot, deadline)

        backend = FailedTransfer()
        result = run_subset_search(backend, lambda *_: ExpandingPolicy(), online=OnlineConfig(100))
        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.attempts[0].status, "numerical_failure")
        self.assertEqual(result.selected_mask, 3)

    def test_expansion_restarts_original_and_policy_is_built_once(self):
        backend = ProtocolBackend()
        calls = []

        def factory(features, diagnostics, deadline):
            calls.append(features)
            return ExpandingPolicy()

        result = run_subset_search(backend, factory, online=OnlineConfig(100), rng=np.random.default_rng(1))
        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.selected_mask, 3)
        self.assertEqual([item[0] for item in backend.begins], [1, 3])
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(backend.features_at, np.zeros(5))
        for _, start_q in backend.begins:
            np.testing.assert_array_equal(start_q, np.zeros(5))
        np.testing.assert_array_equal(result.candidate.snapshot["velocity"], np.zeros(6))
        self.assertEqual(result.to_record()["selected_groups"], 2)

    def test_gpu_synchronization_is_charged(self):
        clock = FakeClock()
        backend = ProtocolBackend()

        def synchronize():
            clock.now += .6

        result = run_subset_search(backend, lambda *_: ExpandingPolicy(), synchronize=synchronize,
                                   deadline=Deadline(1, clock=clock))
        self.assertTrue(result.budget_exhausted)
        self.assertIsNone(result.candidate)
        self.assertEqual(backend.begins, [])

    def test_full_hand_retains_optimizer_across_segments(self):
        backend = ProtocolBackend()
        result = run_full_hand(backend, online=OnlineConfig(100), rng=np.random.default_rng(3))
        self.assertIsNotNone(result.candidate)
        self.assertEqual(len(backend.begins), 1)
        self.assertEqual(backend.segment_history, [1, 2])
        self.assertEqual(len(result.attempts), 2)

    def test_terminal_subset_failure_restores_original(self):
        class FullOnly:
            def initial(self):
                return 31

            def expand(self, mask):
                raise AssertionError("full hand must never expand")

        backend = ProtocolBackend()
        result = run_subset_search(backend, lambda *_: FullOnly(), online=OnlineConfig(100))
        self.assertIsNone(result.candidate)
        np.testing.assert_array_equal(backend.state["q"], np.zeros(5))
        self.assertNotIn("mask", backend.state)

    def test_cleanup_overrun_rejects_acceptance_and_restores_original(self):
        class AcceptingPolicy:
            def initial(self):
                return 3

            def expand(self, mask):
                return None

        clock = FakeClock()

        class ExpensiveCleanup(ProtocolBackend):
            def clear_correction_forces(self):
                super().clear_correction_forces()
                if self.force_clears == 3:
                    clock.now += 2

        backend = ExpensiveCleanup()
        result = run_subset_search(backend, lambda *_: AcceptingPolicy(), deadline=Deadline(1, clock=clock))
        self.assertIsNone(result.candidate)
        self.assertTrue(result.budget_exhausted)
        self.assertGreaterEqual(result.elapsed_seconds, 1)
        np.testing.assert_array_equal(backend.state["q"], np.zeros(5))


if __name__ == "__main__":
    unittest.main()
