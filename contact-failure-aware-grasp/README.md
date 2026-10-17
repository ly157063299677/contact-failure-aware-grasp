# Contact-Failure-Aware Finger-Subset Selection

Companion reference implementation for **Contact-Failure-Aware Finger-Subset Selection for Budget-Constrained Dexterous Grasp Refinement**.

[中文说明](README_zh.md) · [Method mapping](docs/method_mapping.md) · [Data schema](docs/data_schema.md) · [Backend integration](docs/backend_integration.md)

This repository implements the learning, subset-selection and experiment-control procedures described in the manuscript. It was reconstructed from the manuscript and supplementary material; it is not an archive of the original experimental source code. The original dataset, trained weights, simulation assets and asset-specific GraspADMM adapter are not included. The reported paper results have not been reproduced by this release.

## Included

- All 31 nonempty subsets of five finger groups, with deterministic tie handling.
- PointNet-style object encoding, shared finger encoding, subset conditioning, geometry-biased hand–object attention and five-finger self-attention.
- Soft success supervision, scaled Huber regression, and 110 equal-size plus 75 one-finger expansion comparisons per initial state.
- Object-disjoint dataset utilities, four candidate variants, validity-aware contact features and state-grouped labels.
- AdamW training, three seed runs, early stopping, checkpoints, provenance and both crossed ablations.
- Deadline-aware subset expansion, native-solver continuation for the full-hand baseline, actuator projection and detached six-direction physical validation.
- Random, contact-based and individual-finger Top-k policies, measured result aggregation and budget-sensitivity plotting.
- Numerical/protocol unit tests and a GitHub Actions workflow. Fake backends in tests are protocol fixtures, not physical simulators.

## Environment

The manuscript reports Ubuntu 22.04, Python 3.12, MuJoCo 3.3.3, PyTorch 2.7 and CUDA 12.8.
The reference code uses Python 3.12+ and separate optional dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[learn,physics,plots]"
```

On Windows activate with `.venv\Scripts\Activate.ps1`. For a CUDA 12.8 PyTorch installation, select the appropriate wheel from the [official PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/); GPU/driver compatibility must match the actual machine. A CPU installation is sufficient for numerical and network tests. `environment.yml` is also provided for Conda.

```bash
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
```

The local delivery check is recorded in `docs/validation_report.md`; the GitHub workflow has not been run as part of delivery.

## Data preparation

1. Obtain the actual source grasps, Shadow Hand model, object assets and complete snapshots. Keep the actual object split if reproducing an existing experiment.
2. If constructing a new dataset, split object identities **before** perturbation and labeling. All scales and grasp variants of one object must remain together.
3. Export source-state records and connect a native backend following [backend integration](docs/backend_integration.md). No replacement ADMM objective is silently used.
4. Generate physical outcome labels and audit the exported dataset.

```bash
python scripts/prepare_splits.py assets/object_ids.json --output data/object_splits.json --seed 0
python scripts/generate_labels.py --states assets/source_states.jsonl --backend my_adapter:create_backend --config configs/paper.toml --output data/paper --seed 0
python scripts/audit_dataset.py --data data/paper
# Only for the exact manuscript dataset:
python scripts/audit_dataset.py --data data/paper --paper-counts
```

`my_adapter:create_backend` is the explicit external integration point, not a bundled module. It receives the configuration and one source-state record and returns an initialized `GraspBackend`. The feature layout and implementation choices are documented in [features](docs/features.md).

For data-interface inspection without physics, create a separate artificial fixture:

```bash
python scripts/make_synthetic_demo.py --output data/synthetic_demo
python scripts/audit_dataset.py --data data/synthetic_demo
```

These tensors are explicitly synthetic. The physical evaluation scripts reject them. They cannot substantiate any grasp-repair result.

## Training and ablation

Each command trains seeds `[0,1,2]` unless `--seed` is supplied. The manuscript specifies three runs but not their integer seeds; those values are reference defaults.

```bash
# Conditioned interaction + comparison loss
python scripts/train.py --config configs/paper.toml --data data/paper --output runs/proposed
# PointNet-Finger MLP, outcome only: paper comparison baseline
python scripts/train.py --config configs/paper.toml --data data/paper --output runs/mlp_outcome --no-interaction --outcome-only
# Other two cells of the crossed ablation
python scripts/train.py --config configs/paper.toml --data data/paper --output runs/mlp_pair --no-interaction
python scripts/train.py --config configs/paper.toml --data data/paper --output runs/interaction_outcome --outcome-only
```

Each `seed_<n>/` contains `best.pt`, `last.pt`, `config.json`, `history.jsonl`, `provenance.json` and `summary.json`. Validation loss determines early stopping; the test partition is never used for training or checkpoint selection. Regression scale `sd` is estimated only from finite training deviations. Selection always uses the predicted deviation in its original normalized units.

```bash
python scripts/predict.py --checkpoint runs/proposed/seed_0/best.pt --features data/paper/features/state_000000.npz --output results/subset_scores.csv
```

Prediction scores alone are not validated repaired grasps.

## Physical evaluation

The backend and assets must be connected first. Asset/model loading is outside the timer; diagnosis, features, inference, restoration, refinement, settling, validation and expansion are inside it. A passed suite completed after the deadline is rejected. Native operations must check the deadline internally; an opaque blocking solver cannot guarantee prompt interruption.

```bash
python scripts/evaluate.py --data data/paper --backend my_adapter:create_backend --method proposed --checkpoint runs/proposed/seed_0/best.pt --seed 0 --budget 5 --output results/proposed_seed0_B5.jsonl
python scripts/evaluate.py --data data/paper --backend my_adapter:create_backend --method mlp --checkpoint runs/mlp_outcome/seed_0/best.pt --seed 0 --budget 5 --output results/mlp_seed0_B5.jsonl
python scripts/evaluate.py --data data/paper --backend my_adapter:create_backend --method full_hand --seed 0 --budget 5 --output results/full_hand_B5.jsonl
python scripts/evaluate.py --data data/paper --backend my_adapter:create_backend --method contact --seed 0 --budget 5 --output results/contact_B5.jsonl
python scripts/evaluate.py --data data/paper --backend my_adapter:create_backend --method random --seed 0 --budget 5 --output results/random_B5.jsonl
python scripts/evaluate.py --data data/paper --backend my_adapter:create_backend --method topk --checkpoint runs/mlp_outcome/seed_0/best.pt --seed 0 --budget 5 --output results/topk_seed0_B5.jsonl
```

Repeat learned evaluations with the matching seed/checkpoint for all three seeds and budgets 2, 5 and 10. For ablation runs use `--run-name interaction_outcome` or `--run-name mlp_pair` to keep configurations in separate result groups. Top-k evaluates only the five singleton masks with the outcome-only PointNet–Finger MLP checkpoint, as specified in Table S3. Do not perform label generation or training concurrently with timing experiments.

The unrefined baseline is separate and uses the **complete** test partition:

```bash
python scripts/evaluate_unrefined.py --data data/paper --backend my_adapter:create_backend --output results/unrefined.jsonl
python scripts/summarize_results.py results/proposed_seed0_B5.jsonl results/proposed_seed1_B5.jsonl results/proposed_seed2_B5.jsonl --data data/paper --output results/summary
python scripts/plot_results.py --summary results/summary/summary.json --output results/budget_sensitivity.pdf
```

RSR assigns zero to inputs with no accepted output and scores accepted outputs with five independent final suites. Deviation and selected-group means use online-accepted outputs, even if independent final evaluation fails. Runtime averages use every evaluated input. Metrics are computed per seed before averaging. No experimental result table is hard-coded in this repository.

## Repository structure

```text
configs/paper.toml           Paper settings and explicit reference defaults
src/cfgrasp/
  features.py               Object-frame, contact and relative geometry features
  construction.py           Object splits and four perturbation proposals
  data.py                   Grouped state dataset and label audits
  subsets.py, policies.py   Mask enumeration, comparisons and selection baselines
  models.py, losses.py      Neural predictor and comparison learning
  training.py, inference.py Training/checkpoint/inference implementation
  labeling.py               Independent offline outcomes for all 31 subsets
  physics/                  Projection, native backend contracts, MuJoCo validation
  runtime.py                Budgeted online refinement and final validation
  metrics.py                Paper evaluation denominators and seed aggregation
scripts/                    Preparation, labeling, training and evaluation CLIs
tests/                      Mathematical, tensor and protocol checks
docs/                       Method mapping, integration, schema and QA scope
```

## Citation and release metadata

Use `CITATION.bib` after adding the final paper author/venue/DOI and repository metadata, which were absent from the supplied manuscript. The original data, pretrained weights, native experiment adapter and chosen code license can be added by the maintainers. See `LICENSE_NOTICE.md` and the [source references](docs/references.md).
