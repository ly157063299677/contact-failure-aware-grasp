# Manuscript-to-code mapping

## Mathematical method

| Manuscript | Code | Implementation |
|---|---|---|
| Eq. (1), IV-A | `features.py`, `physics/base.py` | Joint/control/palm/object/contact features with full simulator snapshot contract |
| Eq. (2) | `physics/projection.py`, native adapter | Restrict actuator targets; retain native QP/ADMM objective and whole-hand contacts |
| Eq. (3) | `physics/validation.py:normalized_deviation` | Mean absolute joint change divided by joint range, all finger joints, before disturbances |
| Eq. (4) | `subsets.py:utilities` | `alpha*p - beta*d - gamma*groups/5` |
| Eq. (5) | `labeling.py`, `physics/validation.py` | AND of six directional tests; mean of five repetitions |
| Eq. (6) | `models.py:SubsetOutcomeNetwork` | Own control-role and whole-mask conditioning before interaction |
| Eq. (7) | `models.py:GeometryBiasedCrossAttention` | Pre-normalized finger query, point keys/values, geometry/diagnostic/role bias |
| Eq. (8) | `models.py:InteractionBlock` | Residual cross-attention, all-five-finger self-attention, FFN |
| Eq. (9) | `models.py:pool_fingers` | Separate selected/unselected pooling, full-hand empty-group flag, sigmoid/softplus heads |
| Eq. (10) | `losses.py` | Soft-label BCE and finite-deviation scaled Huber regression |
| Eqs. (11)–(12) | `losses.py`, `subsets.py:pair_indices` | State-local soft preferences, separately averaged equal/expansion families |
| Eq. (13) | `physics/projection.py` and adapter hooks | Project EVERY proposed actuator update, including inner hand substeps |
| Eq. (14) | `policies.py:LearnedPolicy`, `runtime.py` | Cache predictions; after failure add exactly one finger from the best expansion |
| Eq. (15) | `metrics.py`, `scripts/evaluate.py` | Independent final repair credit on the complete initially failed test set |

The model keeps unselected fingers in both attention and pooling. An actuator mask is not a contact-validity mask or a physical-joint lock. The native solver may apply correction forces to every finger; these are cleared before settling and validation. Selection penalties are never added to the native GraspADMM objective.

## Experimental protocol

| Supplement | Code/config |
|---|---|
| S1 object partition and four candidates | `construction.py`, `prepare_splits.py` |
| S1 independent subset outcomes and shared perturbations | `generate_labels.py`, `labeling.py` |
| S1 six directions, 2 seconds, force at COM, gravity disabled | `physics/validation.py`, `physics/mujoco_validation.py` |
| S2 network/training/utility settings | `configs/paper.toml`, `models.py`, `training.py`, `losses.py` |
| S3 random/contact/Top-k and learned methods | `policies.py`, `scripts/evaluate.py` |
| S3 two-by-two ablation | `train.py --no-interaction/--outcome-only` |
| S4 full-hand native-state continuation | `runtime.py:run_full_hand` |
| S4 diagnosis-to-acceptance deadline | `physics/base.py:Deadline`, `runtime.py` |
| S4 five independent final suites | `runtime.py:independent_final_validation` |
| S4 seed-wise means and runtime records | `metrics.py`, evaluation JSONL logs |

The manuscript's result values are observations to be reproduced with the original assets and configuration. They are not constants in these programs and are not produced by the artificial schema fixtures.

## Explicit implementation choices and missing experimental inputs

The manuscript fixes the method but does not fully specify these details. This release uses transparent defaults:

- Integer-mask order 1–31 for the final tie-break; ties first prefer higher predicted success.
- Training seed values 0, 1 and 2, not claimed as the original seed values.
- Feature width/channel layout, diagnostic reductions, PointNet/shared MLP internal widths, GELU activations, FFN expansion factor 4 and attention-bias hidden width 32.
- Zero dropout, no learning-rate schedule, no gradient clipping by default. Early stopping monitors total validation objective with strict improvement.
- Population standard deviation (`ddof=0`) for finite training deviations, floored at `1e-3`.
- World-frame velocity perturbations and world-axis test loads; exact asset orientation and perturbation sampling details should be reconciled with the original implementation.
- Settling rejection uses the same 0.05 m/15 degree pose thresholds as validation because a separate settling threshold was not given. The native validation scene controls gravity during settling; matching the experiment requires explicitly setting and recording it.
- Validation rejects if thresholds are exceeded during a directional rollout, not only at its end. The exact threshold sampling policy was not specified.
- Finger collision-surface helpers can use discrete surface samples. Exact nearest-surface queries can be passed by the adapter; sampled nearest points are an approximation, not an exact collision-surface distance.

The original source-state selection, split lists, raw trajectories, sensor extraction, object scale convention, trained weights, native solver parameters and actual upstream revision were not available. The adapter contract requires these to be supplied. No guessed object split or chosen seed can establish replication of the paper's reported dataset counts and numerical results.

The asset-specific adapter is intentionally explicit. A generic quadratic objective or a toy hand would change the experiment; this repository does not substitute either for GraspADMM. Its core learning and control-flow code can be inspected independently of that binding.

The final source recheck incorporated the updated Table S3: Top-k uses the outcome-only PointNet–Finger MLP to evaluate only five singleton subsets. The evaluation CLI checks the checkpoint architecture and comparison-loss weight to enforce this baseline definition.
