# Connecting the physical backend

The numerical pipeline is provided, but the authors' executable Shadow Hand scene,
object assets, state snapshots, actuator maps, required-contact maps and modified
GraspADMM binding were not included with the manuscript. Therefore this repository
does not invent a physical objective or claim to recreate the reported experiments.
Implement `GraspBackend` in `src/cfgrasp/physics/base.py` against those assets before
generating scientific labels. The protocol tests use synthetic state machines and
are not grasp simulations or experimental results.

## Upstream locations inspected

The official [Dexonomy GraspADMM guide](https://github.com/JYChen18/Dexonomy/blob/main/getting_started/3_ADMM.md)
identifies the optimizer entry point. The native
[optimizer](https://github.com/JYChen18/Dexonomy/blob/main/dexonomy/op/gen_grasp_admm.py)
contains the object/contact/dual updates and nested hand updates. Instrument those
updates without replacing their contact-QP or ADMM objective. The corresponding
[configuration](https://github.com/JYChen18/Dexonomy/blob/main/dexonomy/config/op/grasp_admm.yaml),
[contact QP](https://github.com/JYChen18/Dexonomy/blob/main/dexonomy/qp/qp_single.py), and
[MuJoCo environment](https://github.com/JYChen18/Dexonomy/blob/main/dexonomy/sim/mujoco_env.py)
are the integration points. Pin and record the exact upstream commit you use;
these links track a moving branch. No upstream source is vendored here.

## Factory and asset contract

Supply a Python factory, for example `local_adapter:make_backend`, accepting the
configuration and one state manifest record and returning an initialized backend.
Asset loading is outside the online timer. Every subsequent restoration, feature
operation, transfer, optimizer step, settling step and validation step is charged.

The adapter must explicitly provide:

- The Shadow Hand MJCF plus object geometry, scale, mass/inertia, collision/friction
  settings and the independently constructed free-object validation model.
- Five disjoint actuator-index groups in Thumb, Index, Middle, Ring, Little order,
  their limits, all finger joint addresses and limits, and a verified
  joint-to-actuator map that handles coupled/tendon actuators.
- The original controls, complete object and hand state, velocities, palm target,
  mocap/equality state and required-contact link sets for each initial state.
- Object-frame geometry, nearest points on each finger collision surface, normals,
  and valid raw diagnostics. See `docs/features.md` and `cfgrasp.features` for the
  versioned feature channel definitions and aggregation functions.
- Native solver configuration, including the original contact-QP/ADMM terms.
  Deviation and selected-group penalties are only selection terms; do not add them
  to the physical objective.

Keep raw state identifiers outside model inputs. Record object identities and all
scale/grasp variants before creating the object-disjoint split.

## Backend methods

| Method | Required behavior |
|---|---|
| `snapshot()` / `restore(snapshot)` | Deep, independent capture and exact restoration of all mutable simulation **and native solver** state. No aliasing into live arrays. |
| `diagnose(config, deadline)` | Hold initial actuator/palm targets for 0.2 s (50 steps at 0.004 s), collecting the prescribed contact and object-motion diagnostics with validity flags. |
| `build_features(diagnostics, deadline)` | Use the restored original geometry plus the recorded diagnostics. Online orchestration restores the initial state before this call. |
| `finger_configuration(snapshot)` | All finger joint coordinates from the saved state, excluding the palm/free object. |
| `joint_limits()` | Finite lower/upper ranges in exactly the same joint order; every range must be positive. |
| `begin_refinement(mask, reference_snapshot, deadline)` | Reset contact/dual variables for a new subset, fix the original palm target, install target projection and disable recovery/squeezing. |
| `refine(iterations, deadline)` | Execute up to the requested outer iterations of native GraspADMM, checking the deadline inside inner loops. Return an explicit `RefinementOutcome`. |
| `clear_correction_forces()` | Clear every optimizer-generated generalized/Cartesian/contact correction force, including those on unselected fingers. |
| `validation_copy(snapshot, deadline)` | Transfer to an independent free-object environment with matching pose, velocities, activations and retained target controls. It must not advance or mutate the live native solver. |

Missing required measurements must be marked unavailable. A missing contact is
not a measured zero tangential force or a valid zero velocity. Contact absence must
use the supplied required link sets. Unexpected configuration/programming errors
must propagate; use `NumericalFailure` only for identified numerical failures.

## Projection inside the native loop

`ActuatorProjector` implements Eq. (13). It preserves the original controls for
unselected fingers and non-finger actuators and clips selected controls to their
limits. `ProjectedADMMRunner` supplies the callback contract; it is not another
physical solver. Your native adapter must follow this ordering for **every** hand
update, including internal `step_hand` and simulation substeps:

1. Compute the native proposed hand update.
2. Convert it with the verified joint-to-actuator mapping.
3. Call the supplied `commit_targets(proposed_targets)` callback.
4. The callback applies `ActuatorProjector`, writes the resulting targets and
   reasserts the saved palm target.
5. Advance physics, checking the same deadline before/after the step.

An outer-loop-only projection does not enforce the paper's constraint. Do not
freeze unselected joint positions or remove their contacts; unselected fingers may
move under physical and optimizer correction forces. Keep whole-hand contact data
in the native objective. Correction forces may act on all five fingers.

## Snapshot and continuation semantics

Include velocities, controls, actuator activation, acceleration warm starts,
applied forces, mocap/equality state, plugin/user data and all mutable solver
variables (contact positions, duals, accumulators, counters and RNG state when
used). Serialize model identity/configuration separately. Model edits also require
restoration if your adapter changes them at runtime.

Subset attempts restore the original snapshot and initialize the native solver
again. Full-hand refinement initializes once, then executes consecutive blocks of
50 outer iterations while retaining optimizer variables and its current physics
state. Settling and checkpoint validation use detached environments; a failed
validation must not alter that live state. Merely saving `qpos` is insufficient.

The supplied `MuJoCoValidationEnvironment` uses
[`mjSTATE_INTEGRATION`](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#integration-state)
and separately retains gravity and its external load. It operates on an
independently owned MuJoCo model/data pair. Its constructor requires verified
addresses, a rigid free-joint object subtree and every finger qpos address. It
restores only snapshots from that same environment. Transferring a grasp between
the optimization and free-object model is the asset adapter's responsibility.

## Settling and six-direction validation

`refine_candidate` performs an independent subset attempt, clears correction
forces, transfers to a free-object environment, settles for 0.1 s and records the
pre-disturbance snapshot and all-joint normalized deviation. It returns a
`CandidateOutcome`; failed attempts have no candidate. The paper gives no separate
numerical settling threshold, so this implementation explicitly uses the same
translation/rotation limits as validation. Settling retains the gravity setting
of the supplied free-object model; document that setting with your assets.

`validation_suite(env, saved_state, perturbation, config, deadline)` applies loads
in the order +x, -x, +y, -y, +z, -z. Built-in gravity is disabled. Each constant
world-axis force has magnitude object mass times 9.81 m/s² and acts at the rigid
object's center of mass. Each direction starts from the identical perturbed
snapshot and runs for 2 s (500 steps), rejecting displacement over 0.05 m or
rotation over 15 degrees at any step. The concrete MuJoCo adapter maps forces at
the subtree COM with `mj_applyFT`, including objects with rigid child bodies.

Perturbations are independent component-wise world-frame velocity increments,
uniform in ±0.01 m/s and ±0.05 rad/s. MuJoCo's free-joint rotational qvel is
body-frame, so the concrete adapter converts angular increments accordingly.
The helper returns the environment to the saved pre-validation state even on
failure/timeout. A direction is 1 (pass), 0 (observed failure), or -1
(unexecuted/interrupted); -1 never counts as a pass.

Use one zero perturbation for initial screening. For offline labels, draw a bank
of five perturbations **once per initial state** and reuse it for all 31 masks.
Online acceptance uses one fresh suite. Independent final evaluation uses five
additional independent suites on the saved accepted state and does not change
online selection or online latency. Use separate random streams for those stages.

## Runtime and outputs

`run_subset_search` takes a factory of a policy with `initial()` and `expand(mask)`
methods. Learned factories evaluate/cache all 31 subset outcomes once inside the
timer. Every expansion adds exactly one finger and starts from the original state;
the full-hand subset is attempted at most once. `run_full_hand` provides the
separate continuous full-hand baseline.

Both functions use `perf_counter`, check the deadline around operations and reject
late results, including late cleanup. If inference uses CUDA, provide the device's
synchronization callback so asynchronous work and transfers are included. A
noninterruptible native call may return after the deadline; it will be rejected,
not accepted with a truncated timing measurement. Implement inner-loop deadline
checks to keep such overshoot small.

Failed or interrupted runs clear correction forces and restore the original input
state on exit. This restoration is charged too. The final elapsed-time sample is
also the sample used to decide deadline exhaustion, preventing a late accepted
result from being paired with a reported runtime beyond the budget.

`RuntimeResult.candidate` is the saved accepted pre-validation candidate or `None`.
`to_record()` returns JSON-compatible runtime/attempt metrics without serializing
opaque simulator state. Preserve accepted state using an explicit adapter-side
serialization method when needed. All unsuccessful/deadline runs remain in the
evaluation denominator. `independent_final_validation` uses the saved state and a
separate RNG stream; it is excluded from online timing.

## Validation status of this release

The shipped unit tests verify subset projection, state restoration, six-direction
protocol, timeout rejection, one-time inference, expansion restart, and retained
full-hand optimizer state using deterministic state-machine doubles. They do not
validate native GraspADMM integration, Shadow Hand assets, MuJoCo contact behavior,
trained checkpoints or manuscript performance values. Those require the original
asset/configuration binding and actual experiment runs.
