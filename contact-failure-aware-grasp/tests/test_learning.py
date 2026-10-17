"""Focused learning invariants; skipped when the optional PyTorch extra is absent.

These use synthetic tensors solely to verify code behavior. They do not assess
grasp quality or reproduce any result in the manuscript.
"""
import importlib.util
import math
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from cfgrasp.losses import OutcomeComparisonLoss, estimate_deviation_scale
    from cfgrasp.models import SubsetOutcomeNetwork, pool_fingers
    from cfgrasp.subsets import MASKS


@unittest.skipUnless(HAS_TORCH, "Optional PyTorch dependency is not installed")
class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def batch(self, size=2):
        diagnostics = torch.randn(size, 5, 12)
        finger_features = torch.randn(size, 5, 35)
        finger_features[..., -12:] = diagnostics
        return {
            "points": torch.randn(size, 11, 3),
            "finger_features": finger_features,
            "diagnostics": diagnostics,
            "geometry": torch.randn(size, 5, 11, 5),
            "context": torch.randn(size, 23),
            "success": torch.rand(size, 31),
            "deviation": torch.rand(size, 31) * 0.2,
        }

    def prediction(self, size=2):
        logits = torch.zeros(size, 31, requires_grad=True)
        deviation = torch.full((size, 31), 0.2, requires_grad=True)
        return {"success_logits": logits, "success": logits.sigmoid(), "deviation": deviation}

    def test_both_architectures_predict_all_masks_and_backpropagate(self):
        for interaction in (False, True):
            with self.subTest(interaction=interaction):
                network = SubsetOutcomeNetwork({"dim": 16, "heads": 4, "blocks": 2,
                                                "interaction": interaction})
                batch = self.batch()
                prediction = network(batch)
                for value in prediction.values():
                    self.assertEqual(value.shape, (2, 31))
                    self.assertTrue(torch.isfinite(value).all())
                self.assertTrue(((prediction["success"] >= 0) & (prediction["success"] <= 1)).all())
                self.assertTrue((prediction["deviation"] >= 0).all())
                losses = OutcomeComparisonLoss(0.05)(prediction, batch["success"], batch["deviation"])
                losses["loss"].backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in network.parameters()
                                    if p.grad is not None))

    def test_full_hand_has_zero_unselected_pool_and_empty_indicator(self):
        values = torch.ones(2, 31, 5, 4)
        selected, unselected, empty = pool_fingers(values, torch.as_tensor(MASKS))
        self.assertTrue(torch.equal(selected, torch.ones_like(selected)))
        self.assertTrue(torch.equal(unselected[:, -1], torch.zeros(2, 4)))
        self.assertTrue(torch.equal(empty[:, -1], torch.ones(2, 1)))
        self.assertEqual(float(empty[:, :-1].sum()), 0.0)

    def test_requested_singletons_match_full_predictions_in_both_architectures(self):
        indices = torch.tensor([0, 1, 3, 7, 15], dtype=torch.long)
        batch = self.batch()
        for interaction in (False, True):
            with self.subTest(interaction=interaction):
                network = SubsetOutcomeNetwork({"dim": 16, "heads": 4, "blocks": 2,
                                                "interaction": interaction}).eval()
                with torch.no_grad():
                    full = network(batch)
                    singletons = network(batch, subset_indices=indices)
                for name in full:
                    self.assertEqual(singletons[name].shape, (2, 5))
                    self.assertTrue(torch.allclose(singletons[name], full[name][:, indices],
                                                   atol=1e-6, rtol=1e-5))

    def test_requested_subset_indices_are_validated(self):
        network = SubsetOutcomeNetwork({"dim": 16, "heads": 4, "interaction": False})
        batch = self.batch()
        for indices in ([], [31], [-1], [0.5], [[0, 1]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                network(batch, subset_indices=indices)

    def test_missing_deviations_do_not_poison_loss_or_gradients(self):
        prediction = self.prediction()
        success = torch.zeros(2, 31)
        deviation = torch.full((2, 31), float("nan"))
        losses = OutcomeComparisonLoss(0.1)(prediction, success, deviation)
        self.assertEqual(float(losses["deviation_huber"]), 0.0)
        self.assertEqual(float(losses["comparison"]), 0.0)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(prediction["deviation"].grad).all())
        self.assertTrue(torch.isfinite(prediction["success_logits"].grad).all())

    def test_finite_failed_deviations_still_supervise_regression(self):
        prediction = self.prediction()
        success = torch.zeros(2, 31)
        deviation = torch.zeros(2, 31)
        # (0.2 - 0.0) / 0.1 = 2; Huber(delta=1) = 1.5.
        losses = OutcomeComparisonLoss(0.1, lambda_pair=0)(prediction, success, deviation)
        self.assertAlmostEqual(float(losses["deviation_huber"]), 1.5, places=6)
        self.assertEqual(int(losses["n_deviation"]), 62)

    def test_comparison_families_are_averaged_separately(self):
        prediction = self.prediction()
        success = torch.full((2, 31), 0.5)
        deviation = torch.full((2, 31), 0.2)
        losses = OutcomeComparisonLoss(0.1, gamma=0.1)(prediction, success, deviation)
        self.assertEqual(int(losses["n_equal_pair"]), 220)
        self.assertEqual(int(losses["n_expansion_pair"]), 150)
        self.assertAlmostEqual(float(losses["equal_pair"]), math.log(2), places=6)
        # Expansion incurs gamma/5 = .02 utility cost, hence logit .2 at tau=.1.
        # At matching observed/predicted utilities, BCE equals target entropy.
        target = 1.0 / (1.0 + math.exp(-0.2))
        expansion_entropy = -target * math.log(target) - (1-target) * math.log(1-target)
        self.assertAlmostEqual(float(losses["expansion_pair"]), expansion_entropy, places=6)
        # 110 vs 75 pairs must not turn this into a count-weighted family mean.
        self.assertAlmostEqual(float(losses["comparison"]),
                               0.5 * math.log(2) + 0.5 * expansion_entropy, places=6)

    def test_unavailable_subset_is_removed_from_every_pair_containing_it(self):
        prediction = self.prediction(1)
        success = torch.full((1, 31), 0.5)
        deviation = torch.full((1, 31), 0.2)
        deviation[:, -1] = float("nan")
        losses = OutcomeComparisonLoss(0.1)(prediction, success, deviation)
        # Full hand has no equal-size partner, but five incoming expansion edges.
        self.assertEqual(int(losses["n_equal_pair"]), 110)
        self.assertEqual(int(losses["n_expansion_pair"]), 70)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(prediction["deviation"].grad).all())

    def test_scale_uses_every_finite_training_target_and_floor(self):
        records = [{"success": torch.zeros(3), "deviation": torch.tensor([0.0, 0.2, float("nan")])},
                   {"success": torch.ones(1), "deviation": torch.tensor([0.4])}]
        self.assertAlmostEqual(estimate_deviation_scale(records), math.sqrt(0.08 / 3), places=6)
        self.assertEqual(estimate_deviation_scale([{"deviation": torch.zeros(31)}]), 1e-3)
        with self.assertRaises(ValueError):
            estimate_deviation_scale([{"deviation": torch.full((31,), float("nan"))}])

    def test_label_tensors_are_detached(self):
        prediction = self.prediction(1)
        success = torch.full((1, 31), 0.5, requires_grad=True)
        deviation = torch.full((1, 31), 0.1, requires_grad=True)
        OutcomeComparisonLoss(0.1)(prediction, success, deviation)["loss"].backward()
        self.assertIsNone(success.grad)
        self.assertIsNone(deviation.grad)


if __name__ == "__main__":
    unittest.main()
