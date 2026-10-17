# Delivery validation

Checked on 2026-10-16 using the local bundled Python runtime and NumPy.

| Check | Result |
|---|---|
| Python syntax compilation | Passed for source, scripts and tests |
| Numerical and protocol unit tests | 24 passed |
| Optional learning tests | 10 skipped because PyTorch was unavailable |
| Synthetic dataset generator and schema audit | Passed for 12 artificial states, kept outside the release |
| TOML/package metadata and local documentation links | Checked by the packaging audit |
| Native MuJoCo execution | Not run; dependency/assets unavailable |
| Native GraspADMM integration | Not run; asset-specific experiment adapter unavailable |
| Neural training or inference with real weights | Not run |
| Manuscript experiment reproduction | Not performed |
| GitHub Actions | Workflow provided; not executed on GitHub during delivery |

The passing tests cover comparison-pair combinatorics, tie-breaking, object split isolation,
feature layout/validity, relative geometry, perturbation bounds, actuator constraints,
all-joint deviation, six-direction state restoration, numerical interruption, deadlines,
subset expansion, full-hand optimizer continuation and evaluation denominators.

Protocol tests use small deterministic state machines; their outcomes are not experimental
evidence about grasp stability. Synthetic arrays test file interfaces only. No tests,
logs or documentation claim the reported manuscript success rates were reproduced.

The implementation was cross-reviewed against the manuscript for missing-data handling,
original-state restoration, one-time subset prediction, independent final validation,
and seed-wise aggregation. Further checks with the actual assets and trained weights
are required to establish physical and numerical reproduction.
