# State features and data construction

The manuscript specifies the kinds of inputs and the main dataset protocol, but
does not give every channel, reduction, mesh-query convention, or random seed.
`cfgrasp-state-v1` is an explicit reference implementation of those missing
details. These defaults are **not a claim to recover the original experimental
implementation**. The repository contains no original meshes, trajectories,
snapshots, labels, trained parameters, or fabricated experiment results.

## Coordinate and unit contract

All positions, normals, and displacements use the **current object coordinate
frame**. Raw positions and `object_scale` use metres; joint states and the angular
position-actuator targets used by the perturbation helper use radians. Choose a
positive characteristic object length consistently (for example, the diagonal of
the scaled object geometry's bounding box), record that convention, and keep the
same convention in training and inference. The paper does not define the exact
scale statistic. Do not mix a dimensionless asset scaling multiplier with a length
in metres.

`points_to_object_frame` and `vectors_to_object_frame` take the rigid transform
from object coordinates to world coordinates. Positions subtract its translation;
vectors/normals do not. `palm_to_object` maps palm-frame coordinates into the
object frame. If the simulator provides world poses, compute it as
`inverse(object_to_world) @ palm_to_world`.

Finger order is always **thumb, index, middle, ring, little**. Supply the actual
hand model's finger joint and actuator mappings, excluding unrelated wrist or
palm entries. The repository does not guess Shadow Hand tendon, coupled-joint, or
joint-to-actuator mappings. All five fingers remain present, regardless of the
candidate subset mask.

## Network arrays

`build_features` returns these finite `float32` arrays, ready for `np.savez`:

| Key | Shape | Contents |
| --- | --- | --- |
| `points` | `[1024,3]` | Object-frame point positions divided by object scale |
| `finger_features` | `[5,35]` | Normalized joints 5, joint validity 5, normalized actuator targets 5, target validity 5, reference xyz 3, diagnostics 12 |
| `diagnostics` | `[5,12]` | The same diagnostic channels supplied separately to attention bias |
| `geometry` | `[5,1024,5]` | Nearest-surface relative displacement xyz, separation, normal alignment |
| `context` | `[23]` | Palm rotation 9, palm translation 3, global diagnostics 10, scale 1 |

The palm rotation is flattened row-major. Palm translation and finger reference
positions are divided by object scale; the final context value retains the actual
scale in metres. The finger reference is a documented, consistent landmark from
the current hand kinematics, such as the distal-link frame origin. It is not used
as a substitute for nearest collision-surface queries.

Joints and controls use `(value - lower_limit)/(upper_limit - lower_limit)`.
Values outside limits are retained as evidence; this preprocessing step does not
clip physical joint states. Unequal groups are zero-padded to five entries each,
with explicit validity channels. NaN measurements also receive zero value and
zero validity. Each joint/actuator must occur in exactly one group; larger groups
raise an error instead of silently truncating. A hand needing more than five
entries requires an explicit feature/model dimension revision.

Uniform point resampling uses a local NumPy generator, defaults to seed 0, and
samples without replacement when at least 1,024 source points are available.
Smaller source clouds use replacement. The same selected indices apply to normals
and exact nearest-surface arrays. Sampling is an implementation choice; seed 0 is
not the unknown original experiment seed. Record per-state sampling seeds in
provenance when varying them.

## Finger diagnostics

Raw arrays are `[diagnostic_steps, contact_links]`. Use 50 steps at 0.004 s under
the supplied protocol. Each link has an anatomical `link_finger` in 0..4 and an
explicit boolean `required_contact` from the source grasp's contact-link
specification. `contact_present` is 1 for contact, 0 for measured absence, and NaN
for an unavailable observation. A selection mask is never a contact observation.

| Index | Channel | Reduction / units |
| --- | --- | --- |
| 0 | `required_contact_any` | Whether this finger has at least one required contact link |
| 1 | `missing_required_contact` | Any required link with observed samples and no observed contact during diagnosis |
| 2 | `contact_valid_fraction` | Fraction of required-link time samples with known contact observations |
| 3 | `mean_separation_over_scale` | Mean finite signed link-object separation across this finger's supplied links, divided by scale |
| 4 | `separation_valid` | At least one finite separation sample |
| 5 | `required_contact_persistence` | Mean contact-present value over observed required-link time samples |
| 6 | `mean_normal_force` | Mean normal force magnitude over finite observed contact samples, N |
| 7 | `normal_force_valid` | At least one such normal-force observation |
| 8 | `mean_tangential_force` | Mean tangential force magnitude over finite observed contact samples, N |
| 9 | `tangential_force_valid` | At least one such tangential-force observation |
| 10 | `mean_tangential_speed` | Mean relative tangential speed over finite observed contact samples, m/s |
| 11 | `tangential_speed_valid` | At least one such tangential-speed observation |

No required links gives zeros in indices 0, 1, 2, and 5. A completely unobserved
required link gives no positive missing-contact finding: validity distinguishes
that unknown case from measured contact absence. If only some samples are
available, the missing-contact flag reflects available evidence and channel 2
reveals partial coverage. Absence here means no contact throughout the **observed
window**, not an inferred final-step absence. This aggregation is a documented
implementation default because the paper does not give the exact reduction rule.

Force and speed channels use all supplied contact links of a finger, including
links not marked required, and only samples where `contact_present == 1`. Missing
force/speed arrays are NaN, never invented physical zeros. The adapter must first
sum simultaneous contact contributions to obtain each link's normal/tangential
force magnitude according to its convention, and compute relative tangential
velocity projected into the contact plane. Document those backend conventions.

## Global diagnostics

The ten global channels are five `(value, valid)` pairs:

1. Maximum object translation magnitude from diagnosis start, divided by scale.
2. Maximum object relative rotation angle from diagnosis start, in radians.
3. Mean resultant force-residual magnitude, in N.
4. Mean resultant torque-residual magnitude, in N m.
5. Mean Euclidean norm of the contact-quality QP residual.

Translation, force residual, and torque residual inputs have shape `[steps,3]`.
Rotation angle is `[steps]`. QP residual is `[steps]` for a nonnegative residual
norm, or `[steps,K]` for a residual vector. A vector sample is available only when
all its components are finite. If no valid samples exist, emit `(0,0)`, not a valid
zero-error observation. Force/torque residuals must be supplied by the actual
simulator and contact solver, with applied loads, gravity convention, reference
point, and sign convention recorded. The helper does not invent their physical
definition. QP residual scaling is solver-specific and must stay consistent.

## Current collision-surface geometry

For object point `p_j` and nearest point `c_ij` on the **current collision surface
of finger i**, the five channels are:

```text
[(p_j - c_ij) / object_scale,
 ||p_j - c_ij|| / object_scale,
 dot(unit_finger_normal_at_c_ij, unit_object_normal_at_p_j)]
```

The normal dot product is signed: opposing outward normals have alignment -1.
Geometry is available for every finger even when contact is absent or that finger
is unselected. It must be recomputed from the current state, never a proposed
refinement result or the eventual label state. Collision geometry includes the
appropriate links for the entire finger; using only a fingertip center changes
the intended descriptor.

For continuous mesh/primitive closest-point queries, pass exact arrays
`nearest_points_obj` and `nearest_normals_obj` with shape `[5,M,3]` corresponding
to the full unsampled object point cloud. `build_features` resamples them together
with the cloud. The separate `relative_geometry` function also accepts the direct
query output for an already sampled cloud.

Alternatively, provide sampled current collision surfaces for each finger.
`nearest_sampled_finger_surfaces` finds the closest **sample**, which approximates
the continuous nearest point. Surface-sampling density and mesh discretization
affect accuracy; this path is explicitly an approximation, not exact mesh
distance. Normals must exist and be nonzero. No normal-validity channel is
available in the five-channel geometry contract, so unavailable normals raise an
error rather than silently becoming a neutral alignment.

## Dataset construction and leakage checks

1. Inventory canonical **base object identities**; group all scales under the
   same identity before splitting.
2. Run `python scripts/prepare_splits.py object_ids.json --output splits.json`.
   Install this package first, or set `PYTHONPATH=src`. The input is a JSON list of
   actual object ID strings. The output has `train`, `val`, and `test` lists.
3. The default ratios are 0.7/0.1/0.2. For 400 identities this gives exactly
   280/40/80. Other inventories require at least three objects: assign one to each
   split, then repeatedly assign to the greatest deficit from the target counts.
   These general-case counts are not the paper's dataset. Seed 0 is a configurable
   implementation default. The command refuses to overwrite an existing split.
4. Select five source grasps per object **after** partitioning. The original
   source-grasp selection policy and original object lists are not supplied, so
   this repository cannot reconstruct them or certify the published dataset.
5. For each source grasp, `propose_grasp_variants` returns four proposals:
   original, actuator, palm, and combined. Angular position targets receive
   independent uniform perturbations in +/-0.05 rad and are clipped to actuator
   limits. Palm translation receives independent +/-0.005 m perturbations per
   object-frame axis. Rotation uses independent +/-2.5 degree perturbations per
   axis, composed as `Rz @ Ry @ Rx @ original_rotation`; translation is perturbed
   separately about the original palm position. The rotation convention and
   independent combined draw are implementation defaults, not specified original
   choices. Original out-of-range controls are rejected for source-data repair.
6. Implement scene construction and coupled-control mapping in the simulator
   adapter. These proposals do not simulate a hand or generate complete snapshots.
   Save complete simulator state after your documented initialization procedure.
   The palm variation belongs to **initial dataset construction**; online subset
   refinement keeps each resulting state's palm target fixed.
7. Run unperturbed six-direction initial screening; record `initially_failed` as
   a real boolean. Run diagnosis while retaining the original control targets,
   restore the source snapshot, then create offline subset labels. Never infer
   failure status from contact features alone.
8. Write one JSONL manifest row per initial state, using the schema below. Run
   `audit_manifest` (or `prepare_splits.py --audit-manifest ...`) before training.
   It rejects duplicate state IDs, cross-partition object assignments, and shared
   feature/label file paths across partitions. Add
   `--check-paths` to verify feature/label file existence. Canonicalization must be
   done upstream: differently named copies of the same mesh can evade any ID-only
   audit. Keep preprocessing choices and fitted statistics training-only.

```json
{
  "state_id": "<object-id>/<source-grasp-id>/<variant>",
  "object_id": "<canonical-base-object-id>",
  "split": "train",
  "features": "features/<state-file>.npz",
  "labels": "labels/<state-file>.npz",
  "initially_failed": true,
  "source_grasp_id": "<source-grasp-id>",
  "variant": "original"
}
```

This is a schema illustration, not a real sample. Paths are relative to the
manifest directory unless absolute. `initially_failed` must be replaced with the
measured screening result; it is not a default. State/object/source IDs, paths,
failure labels, and partition names never enter the network input. Store asset
version/hash, scale convention, geometry-query mode, construction/sampling seed,
hand mapping, and feature schema version as separate provenance metadata.

With exactly 400 objects, five source grasps each, four initial states per source
grasp, and all 31 nonempty subsets, the arithmetic is 8,000 states and 248,000
state-subset records. The repository does not populate these counts or the
published finite-label and failure totals without running the real pipeline.

## Scoped verification

`python -m unittest discover -s tests -p test_features.py` checks validity-aware
aggregation, object-frame transforms, normalized padding, geometry signs and
scaling, point-normal correspondence, the feature contract, object leakage, and
perturbation bounds. These numerical tests use synthetic arrays only and do not
validate MuJoCo contacts, the external solver, the source dataset, physical grasp
success, or any reported paper metric.
