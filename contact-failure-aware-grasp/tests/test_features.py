"""Scoped numerical/contract checks; these do not validate a physics backend."""
import unittest

import numpy as np

from cfgrasp.construction import (
    audit_manifest, audit_object_partitions, partition_objects, propose_grasp_variants,
)
from cfgrasp.features import (
    aggregate_finger_diagnostics, aggregate_global_diagnostics, build_features,
    nearest_sampled_finger_surfaces, normalize_and_pad, points_to_object_frame,
    relative_geometry, resample_point_cloud, vectors_to_object_frame,
)


class FeatureTests(unittest.TestCase):
    def test_missing_contacts_and_unknown_contacts_are_distinct(self):
        diagnostics = aggregate_finger_diagnostics(
            link_finger=[0, 1, 2, 3, 4],
            required_contact=np.array([True, True, True, False, False]),
            contact_present=[[0, np.nan, 1, 0, 1], [0, np.nan, 0, 0, 1]],
            object_scale=0.1,
            separation=[[0.01, np.nan, 0, 0.02, 0], [0.03, np.nan, 0.02, 0.02, 0]],
            normal_force=np.ones((2, 5)),
            tangential_speed=[[999, np.nan, 0.2, 999, 0.1], [999, np.nan, 999, 999, 0.3]],
        )
        np.testing.assert_allclose(diagnostics[0, :3], [1, 1, 1])
        np.testing.assert_allclose(diagnostics[1, :3], [1, 0, 0])
        self.assertEqual(diagnostics[2, 5], 0.5)
        self.assertEqual(diagnostics[3, 0], 0)
        self.assertEqual(diagnostics[3, 1], 0)
        self.assertEqual(diagnostics[0, 7], 0)  # no contact, not a valid zero force
        self.assertEqual(diagnostics[1, 4], 0)  # unknown separation
        self.assertAlmostEqual(diagnostics[0, 3], 0.2)
        self.assertAlmostEqual(diagnostics[2, 10], 0.2)  # ignore no-contact 999
        self.assertEqual(diagnostics[2, 9], 0)  # tangential force unavailable

    def test_unequal_groups_have_separate_padding_validity(self):
        groups = [[0, 1], [2], [3], [], [4, 5, 6]]
        features, valid = normalize_and_pad(
            [0, 1, np.nan, 2, -1, 0, 3], np.tile([-1, 3], (7, 1)), groups
        )
        self.assertEqual(features.shape, (5, 5))
        self.assertEqual(valid.sum(), 6)
        self.assertEqual(features[0, 0], 0.25)
        self.assertEqual(features[1, 0], 0)
        self.assertEqual(valid[1, 0], 0)
        self.assertEqual(valid[4, 2], 1)
        self.assertEqual(valid[4, 3], 0)
        with self.assertRaises(ValueError):
            normalize_and_pad([0, 0], [[0, 1], [0, 1]], [[0], [0], [], [], []])

    def test_nearest_surface_geometry_uses_current_surfaces_and_normals(self):
        points = np.array([[1, 0, 0], [3, 0, 0]], dtype=float)
        object_normals = np.tile([1, 0, 0], (2, 1))
        surfaces = [np.array([[0, 0, 0], [2.5, 0, 0]]) for _ in range(5)]
        normals = [np.tile([-1, 0, 0], (2, 1)) for _ in range(5)]
        nearest, nearest_normals = nearest_sampled_finger_surfaces(points, surfaces, normals)
        geometry = relative_geometry(points, object_normals, nearest, nearest_normals, object_scale=2)
        self.assertEqual(geometry.shape, (5, 2, 5))
        np.testing.assert_allclose(geometry[0, 0], [0.5, 0, 0, 0.5, -1])
        np.testing.assert_allclose(geometry[0, 1], [0.25, 0, 0, 0.25, -1])
        with self.assertRaises(ValueError):
            relative_geometry(points, np.zeros((2, 3)), nearest, nearest_normals, 2)

    def test_world_to_object_transform_distinguishes_positions_and_vectors(self):
        pose = np.eye(4)
        pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        pose[:3, 3] = [5, 2, 0]
        np.testing.assert_allclose(points_to_object_frame([[5, 3, 0]], pose), [[1, 0, 0]])
        np.testing.assert_allclose(vectors_to_object_frame([[0, 1, 0]], pose), [[1, 0, 0]])

    def test_global_missing_flags_and_reductions(self):
        global_features = aggregate_global_diagnostics(
            object_scale=0.5, translation=[[0, 0, 0], [0.1, 0, 0]],
            rotation_angle=[np.nan, 0.3], force_residual=[[3, 4, 0], [np.nan, 0, 0]],
            qp_residual=[[3, 4], [0, 0]],
        )
        np.testing.assert_allclose(global_features, [0.2, 1, 0.3, 1, 5, 1, 0, 0, 2.5, 1])
        with self.assertRaises(ValueError):
            aggregate_global_diagnostics(object_scale=1, translation=[0, 0, 0])

    def test_resampling_preserves_point_normal_correspondence(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        normals = np.eye(3)
        sampled, sampled_normals, index = resample_point_cloud(points, normals, 1024, seed=7)
        np.testing.assert_array_equal(sampled, points[index])
        np.testing.assert_array_equal(sampled_normals, normals[index])
        np.testing.assert_array_equal(index, resample_point_cloud(points, normals, 1024, seed=7)[2])

    def test_full_feature_contract_and_exact_geometry_sampling(self):
        points = np.array([[0.1, 0, 0], [0.2, 0, 0]])
        nearest = np.tile(points[None], (5, 1, 1))
        nearest[:, :, 0] -= 0.01
        normals = np.tile([1, 0, 0], (2, 1))
        features = build_features(
            points_obj=points, object_normals_obj=normals, object_scale=0.1,
            q=np.zeros(5), q_limits=np.tile([-1, 1], (5, 1)), q_groups=[[i] for i in range(5)],
            actuator_targets=np.zeros(5), actuator_limits=np.tile([-1, 1], (5, 1)),
            actuator_groups=[[i] for i in range(5)], finger_reference_obj=np.zeros((5, 3)),
            palm_to_object=np.eye(4), diagnostics=np.zeros((5, 12)), global_diagnostics=np.zeros(10),
            nearest_points_obj=nearest, nearest_normals_obj=np.tile(normals[None], (5, 1, 1)), seed=9,
        )
        expected = {"points": (1024, 3), "finger_features": (5, 35), "diagnostics": (5, 12),
                    "geometry": (5, 1024, 5), "context": (23,)}
        self.assertEqual(set(features), set(expected))
        for name, shape in expected.items():
            self.assertEqual(features[name].shape, shape)
            self.assertEqual(features[name].dtype, np.float32)
            self.assertTrue(np.isfinite(features[name]).all())
        np.testing.assert_allclose(features["geometry"][..., 3], 0.1)
        self.assertAlmostEqual(features["context"][-1], 0.1)


class ConstructionTests(unittest.TestCase):
    def test_400_object_split_counts_and_no_overlap(self):
        objects = [f"object-{i:03}" for i in range(400)]
        splits = partition_objects(objects)
        self.assertEqual([len(splits[s]) for s in ("train", "val", "test")], [280, 40, 80])
        self.assertEqual(len(audit_object_partitions(splits)), 400)
        self.assertEqual(splits, partition_objects(list(reversed(objects))))
        with self.assertRaises(ValueError):
            partition_objects(["a", "a", "b"])
        self.assertEqual([len(v) for v in partition_objects(["a", "b", "c"]).values()], [1, 1, 1])

    def test_manifest_rejects_cross_object_split_leakage(self):
        partitions = {"train": ["a"], "val": ["b"], "test": ["c"]}
        row = {"state_id": "a/g0/original", "object_id": "a", "split": "train",
               "features": "features/a.npz", "labels": "labels/a.npz", "initially_failed": True}
        self.assertEqual(audit_manifest([row], partitions), {"train": 1, "val": 0, "test": 0})
        with self.assertRaises(ValueError):
            audit_manifest([{**row, "split": "test"}], partitions)
        with self.assertRaises(ValueError):
            audit_manifest([row, row], partitions)
        with self.assertRaises(ValueError):
            audit_manifest([row, {**row, "state_id": "c/g0/original", "object_id": "c", "split": "test"}], partitions)
        with self.assertRaises(ValueError):
            audit_manifest([{**row, "split": None, "object_id": "unknown"}], partitions)
        with self.assertRaises(ValueError):
            audit_object_partitions({"train": ["a"], "val": ["b"], "test": ["a"]})

    def test_four_perturbation_proposals_respect_limits(self):
        u = np.array([0.99, -0.99, 0.0])
        limits = np.tile([-1, 1], (3, 1))
        pose = np.eye(4)
        proposals = propose_grasp_variants(u, limits, pose, seed=123)
        self.assertEqual([p["variant"] for p in proposals], ["original", "actuator", "palm", "combined"])
        np.testing.assert_array_equal(proposals[0]["actuator_targets"], u)
        np.testing.assert_array_equal(proposals[0]["palm_to_object"], pose)
        np.testing.assert_array_equal(proposals[1]["palm_to_object"], pose)
        np.testing.assert_array_equal(proposals[2]["actuator_targets"], u)
        for proposed in proposals:
            self.assertTrue((np.abs(proposed["actuator_targets"]) <= 1).all())
            self.assertTrue((np.abs(proposed["sampled_actuator_delta"]) <= 0.05).all())
            self.assertTrue((np.abs(proposed["translation_delta"]) <= 0.005).all())
            self.assertTrue((np.abs(proposed["rotation_xyz_radians"]) <= np.deg2rad(2.5)).all())
            r = proposed["palm_to_object"][:3, :3]
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        np.testing.assert_array_equal(u, [0.99, -0.99, 0])  # input remains intact


if __name__ == "__main__":
    unittest.main()
