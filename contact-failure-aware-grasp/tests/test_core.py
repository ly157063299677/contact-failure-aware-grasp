import unittest
import numpy as np
from cfgrasp.subsets import MASKS, pair_indices, expansions, best_subset
from cfgrasp.policies import contact_policy, singleton_policy
from cfgrasp.metrics import summarize_run, aggregate_runs


class CoreTests(unittest.TestCase):
    def test_pair_combinatorics_and_expansion(self):
        equal, expand = pair_indices()
        self.assertEqual(MASKS.shape, (31, 5))
        self.assertEqual(equal.shape, (110, 2))
        self.assertEqual(expand.shape, (75, 2))
        self.assertEqual(len({tuple(row) for row in equal}), 110)
        for a, b in expand:
            self.assertTrue(np.all(MASKS[a] <= MASKS[b]))
            self.assertEqual(MASKS[b].sum() - MASKS[a].sum(), 1)
        self.assertEqual(expansions(31), [])

    def test_ties_and_heuristic_priority(self):
        self.assertEqual(best_subset(range(1, 32), np.zeros(31), np.ones(31)), 1)
        diagnostics = np.zeros((5, 12))
        diagnostics[4, 1] = 1
        diagnostics[3, 10:12] = [3, 1]
        diagnostics[2, 10] = 999  # Missing measurement must not win.
        self.assertEqual(contact_policy(diagnostics).initial(), 24)
        self.assertEqual(singleton_policy(np.ones(31), np.zeros(31)).initial(), 3)

    def test_metrics_denominators_and_seed_averages(self):
        common = dict(method="proposed", seed=0, budget_seconds=5, online_seconds=2,
                      final_outcomes=[1, 1, 1, 0, 0], deviation=0.1, selected_groups=2)
        a = dict(common, state_id="a", accepted=True)
        b = dict(common, state_id="b", accepted=False, online_seconds=5)
        result = summarize_run([a, b])
        self.assertAlmostEqual(result["repair_success_rate_percent"], 30)
        self.assertAlmostEqual(result["configuration_deviation"], 0.1)
        self.assertAlmostEqual(result["online_seconds"], 3.5)
        c = dict(a, seed=1, deviation=0.3)
        d = dict(b, seed=1, accepted=True, online_seconds=4, final_outcomes=[1] * 5, deviation=0.3)
        aggregate = aggregate_runs([a, b, c, d], expected_ids={"a", "b"})["seed_averages"][0]
        self.assertAlmostEqual(aggregate["configuration_deviation"], 0.2)
        with self.assertRaises(ValueError):
            aggregate_runs([a, b, c], expected_ids={"a", "b"})
        with self.assertRaises(ValueError):
            aggregate_runs([a])
        for invalid in (dict(a, budget_seconds=float("nan")), dict(a, selected_groups=1.5)):
            with self.assertRaises(ValueError):
                summarize_run([invalid])


if __name__ == "__main__":
    unittest.main()
