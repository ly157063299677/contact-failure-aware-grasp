# Data schema

## Manifest

The dataset root contains `manifest.jsonl`, plus `features/` and `labels/` arrays.
One manifest line identifies one initial state, with **all 31** subset outcomes in its label file:

```json
{"state_id":"object001_grasp0_original","object_id":"object001","split":"train","features":"features/state_000000.npz","labels":"labels/state_000000.npz","initially_failed":true,"source_snapshot":"assets/state_000000.npz"}
```

This is an illustrative schema, not a released sample. Relative feature/label paths are resolved against the dataset root. Paths must remain inside it. Backend-specific fields, such as asset configuration and snapshot paths, may be added. They are passed to the backend factory; the factory must define their resolution explicitly. `state_id` and `object_id` are provenance/grouping fields, never model inputs. `initially_failed` is a JSON boolean measured by the unperturbed six-direction screen, not inferred from later labels.

All grasps and scales of the same `object_id` stay in exactly one split: `train`, `val` or `test`. Each source grasp contributes `original`, `actuator`, `palm` and `combined` states. Preserve the original object identity through every variant. For the manuscript, there are 280/40/80 objects and 5600/800/1600 states. The 480 failed test states define the shared repair-evaluation set; complete-test no-refinement evaluation is separate.

`scripts/generate_labels.py --states` accepts source-state JSONL records before features and labels exist. Each has `state_id`, `object_id`, `split`, plus whatever asset/snapshot fields the native adapter needs. It writes the complete training manifest only after each state has been labeled.

## Feature NPZ

| Key | Shape | Meaning |
|---|---|---|
| `points` | `[1024,3]` | Object-frame points normalized by object scale |
| `finger_features` | `[5,35]` | Padded normalized joints/controls, validity, reference position and diagnostics |
| `diagnostics` | `[5,12]` | Same diagnostic channels used by the attention bias |
| `geometry` | `[5,1024,5]` | Surface-to-point displacement, separation and normal alignment |
| `context` | `[23]` | Palm rotation/translation, global diagnostics and object scale |

Arrays are finite float32. Missing sensor values are encoded with explicit validity indicators, not NaN network inputs. Fingers use Thumb/Index/Middle/Ring/Little order. The diagnostic matrix equals the final 12 channels of `finger_features`. See [features](features.md) for channel definitions, normalizations, requirements and sampled-surface versus exact-nearest-query behavior. The widths and aggregations are documented reference choices because the paper does not publish a complete feature schema.

## Label NPZ

Rows follow integer subset IDs 1–31; bit 0 is the thumb. Thus ID 1 is thumb only, ID 3 thumb+index, and ID 31 full hand. Tensor row index equals subset ID minus one.

| Key | Shape | Meaning |
|---|---|---|
| `masks` | `[31,5]` | Exact ordered binary candidate masks |
| `success` | `[31]` | Empirical frequency of successful six-direction repetitions |
| `deviation` | `[31]` | Pre-disturbance settled joint deviation; NaN if unavailable |
| `directional` | `[31,5,6]` | 1 pass, 0 observed physical failure, -1 unexecuted/interrupted |
| `repetition` | `[31,5]` | All six directions must pass; -1 if subset validation never ran |
| `solver_status` | `[31]` | Native refinement/settling outcome; `ok` when a candidate is available |
| `validation_status` | `[31,5]` | Complete, numerical failure or not executed |
| `iterations` | `[31]` | Completed outer iterations from the backend |
| `perturbation_bank` | `[5,6]` | Shared offline world-frame linear/angular velocity perturbations |
| `initial_screen` | `[6]` | Unperturbed initial-state screening outcomes |

Directions are `+x,-x,+y,-y,+z,-z`. The same bank is used for every subset of a state. Each direction restores the same perturbed settled snapshot. Online and final evaluation draw from separate RNG streams. The repository uses world-frame velocity components as an explicit reference convention.

Terminal failed refinement/settling gets `success=0`, unavailable `deviation=NaN`, and unexecuted directions/repetitions `-1`. An available finite deviation remains a regression target even if every validation fails. A numerical validation interruption fails that repetition and retains its status and `-1` direction, rather than inventing an observed directional failure. Ordinary unsuccessful directional tests are 0. Incomplete initial screening stops label export so the repair-evaluation membership is not fabricated.

Do not drop states where every subset fails. Do not replace missing deviations with zero: that changes regression and ranking targets. Pairwise loss excludes pairs with unavailable deviations. The label loader verifies order, shapes, bounds and consistency of observed outcomes.

## Online logs

`evaluate.py` writes one JSON line per initially failed test state, recording method, seed, budget, acceptance, selected mask, pre-validation deviation, runtime, attempt count, numerical/deadline status, timing breakdown, online directional outcomes and five independent final outcomes. Simulation snapshots are opaque backend objects and are not serialized into generic JSON logs. The runtime API returns the accepted saved state for callers that need to export it using their adapter.

`summarize_results.py` requires the dataset manifest and checks every method/seed/budget against its full failed-test ID set. An interrupted run cannot silently change the RSR denominator. Conditional means are `null` when no outputs are accepted; they are not reported as zero. Seed aggregation preserves an undefined conditional mean rather than silently discarding that seed.
