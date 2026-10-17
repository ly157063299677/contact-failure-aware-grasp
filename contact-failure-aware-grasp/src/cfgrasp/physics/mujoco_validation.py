"""Concrete MuJoCo free-object snapshot/load/step adapter (optional dependency).

This implements validation mechanics, not the missing hand-specific GraspADMM
binding. The caller supplies an independently owned model/data pair, explicit
object and finger addresses, and the transferred optimized state. Assets are not
inferred from guessed names. The object must be one rigid free-joint subtree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .base import NumericalFailure


@dataclass(frozen=True)
class MuJoCoSnapshot:
    integration_state: np.ndarray
    gravity: np.ndarray
    load_world: np.ndarray
    model_identity: int


class MuJoCoValidationEnvironment:
    def __init__(self, model: Any, data: Any, *, object_body_id: int,
                 object_free_joint_id: int, finger_qpos_indices: Sequence[int]):
        import mujoco  # Optional: core model/training code never imports MuJoCo.

        self.mj = mujoco
        self.model, self.data = model, data
        self.object_body_id = int(object_body_id)
        self.object_joint_id = int(object_free_joint_id)
        if not 0 < self.object_body_id < model.nbody:
            raise ValueError("object body must be a non-world body")
        if not 0 <= self.object_joint_id < model.njnt:
            raise ValueError("invalid object free-joint index")
        if model.jnt_type[self.object_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("validation object must have a free joint")
        if model.jnt_bodyid[self.object_joint_id] != self.object_body_id:
            raise ValueError("free joint does not belong to object root body")
        self.finger_indices = np.asarray(finger_qpos_indices, dtype=int)
        if (self.finger_indices.ndim != 1 or len(self.finger_indices) == 0
                or len(set(self.finger_indices)) != len(self.finger_indices)
                or np.any(self.finger_indices < 0) or np.any(self.finger_indices >= model.nq)):
            raise ValueError("explicit unique finger qpos addresses are required")
        # All descendants must be rigidly attached to this object's free root.
        self.object_bodies = []
        for body in range(1, model.nbody):
            ancestor = body
            while ancestor and ancestor != self.object_body_id:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == self.object_body_id:
                if body != self.object_body_id and model.body_jntnum[body] != 0:
                    raise ValueError("articulated objects require a different validation adapter")
                self.object_bodies.append(body)
        self.object_mass = float(np.sum(model.body_mass[self.object_bodies]))
        if self.object_mass <= 0:
            raise ValueError("object mass must be positive")
        self.timestep = float(model.opt.timestep)
        self._gravity = np.asarray(model.opt.gravity).copy()
        self._load = np.zeros(3)
        self._qpos_start = int(model.jnt_qposadr[self.object_joint_id])
        self._dof_start = int(model.jnt_dofadr[self.object_joint_id])
        self._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._held_ctrl = data.ctrl.copy()
        self._held_mocap_pos = data.mocap_pos.copy()
        self._held_mocap_quat = data.mocap_quat.copy()
        mujoco.mj_forward(model, data)

    def snapshot(self) -> MuJoCoSnapshot:
        state = np.empty(self.mj.mj_stateSize(self.model, self._state_spec))
        self.mj.mj_getState(self.model, self.data, state, self._state_spec)
        return MuJoCoSnapshot(state, self.model.opt.gravity.copy(), self._load.copy(), id(self.model))

    def restore(self, snapshot: MuJoCoSnapshot) -> None:
        if snapshot.model_identity != id(self.model):
            raise ValueError("snapshot belongs to a different MuJoCo validation model")
        self.model.opt.gravity[:] = snapshot.gravity
        self._load[:] = snapshot.load_world
        self.mj.mj_setState(self.model, self.data, snapshot.integration_state, self._state_spec)
        # mj_forward reconstructs derived geometry. Preserve warm starts explicitly.
        warmstart = self.data.qacc_warmstart.copy()
        self.mj.mj_forward(self.model, self.data)
        self.data.qacc_warmstart[:] = warmstart
        self._held_ctrl = self.data.ctrl.copy()
        self._held_mocap_pos = self.data.mocap_pos.copy()
        self._held_mocap_quat = self.data.mocap_quat.copy()

    def object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        # Track the object body's reference frame, consistently across each trial.
        start = self._qpos_start
        return self.data.qpos[start:start + 3].copy(), self.data.qpos[start + 3:start + 7].copy()

    def finger_configuration(self) -> np.ndarray:
        return self.data.qpos[self.finger_indices].copy()

    def set_gravity_enabled(self, enabled: bool) -> None:
        self.model.opt.gravity[:] = self._gravity if enabled else 0.0

    def set_velocity_delta(self, linear: np.ndarray, angular: np.ndarray) -> None:
        start = self._dof_start
        # MuJoCo free-joint translation velocity is world-frame; rotation velocity
        # is body-frame, so world angular perturbations must be rotated back.
        rotation = self.data.xmat[self.object_body_id].reshape(3, 3)
        self.data.qvel[start:start + 3] += np.asarray(linear)
        self.data.qvel[start + 3:start + 6] += rotation.T @ np.asarray(angular)
        self.mj.mj_forward(self.model, self.data)

    def set_object_force(self, force_world_at_com: np.ndarray) -> None:
        force = np.asarray(force_world_at_com, dtype=float)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError("object load must be a finite world-frame three-vector")
        self._load[:] = force

    def clear_forces(self) -> None:
        self._load[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

    def step(self) -> None:
        self.data.ctrl[:] = self._held_ctrl
        self.data.mocap_pos[:] = self._held_mocap_pos
        self.data.mocap_quat[:] = self._held_mocap_quat
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        # Rigid subtree COM may differ from its root-body COM. mj_applyFT correctly
        # maps force at that world point to generalized forces without extra torque.
        point = self.data.subtree_com[self.object_body_id].copy()
        self.mj.mj_applyFT(self.model, self.data, self._load, np.zeros(3), point,
                           self.object_body_id, self.data.qfrc_applied)
        previous_time = float(self.data.time)
        numerical_warnings = [self.mj.mjtWarning.mjWARN_BADQPOS, self.mj.mjtWarning.mjWARN_BADQVEL,
                              self.mj.mjtWarning.mjWARN_BADQACC, self.mj.mjtWarning.mjWARN_BADCTRL]
        before = [int(self.data.warning[item].number) for item in numerical_warnings]
        self.mj.mj_step(self.model, self.data)
        after = [int(self.data.warning[item].number) for item in numerical_warnings]
        if (before != after or not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or self.data.time < previous_time + self.timestep * 0.5):
            raise NumericalFailure("MuJoCo reported invalid state or automatic numerical reset")
        self.mj.mj_forward(self.model, self.data)
