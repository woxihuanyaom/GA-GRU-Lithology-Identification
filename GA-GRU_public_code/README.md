# GA-GRU for within-well lithology identification

Source-code package for **A GA-tuned GRU workflow for within-well lithology identification in the Hailar Basin**. This package accompanies the fourth revised manuscript. It is a local release preparation; no public repository address has been assigned yet.

[Chinese instructions](README_中文.md) | [Data format](docs/DATA_FORMAT.md) | [Source map](docs/SOURCE_MAP.md)

## Scope

The primary task is supervised interpolation of unlabeled centers within partially labeled wells. Each well has its own fitted model. The main model is a unidirectional many-to-one GRU (`gagru/local_recurrent.py`, architecture `gru`), with GA-selected hyperparameters. Seven measured curves produce 17 channels: seven physical channels (three resistivities log-transformed), three resistivity contrasts, and seven within-window first differences. Nine samples at 0.125 m spacing span 1.0 m.

Training, validation, and test targets are distinct within a run. Measured log contexts can overlap in the primary random-center experiment. This experiment does not establish blind-well generalization or prediction into a fully withheld continuous interval. The original complete-interval sensitivity implementation is included separately. Previously selected configurations were reused in the manuscript; the primary repeated evaluation is not an independent nested estimate of configuration-selection uncertainty.

## Contents

| Location | Purpose |
| --- | --- |
| `gagru/` | Original reusable scientific implementation; legacy helpers retained for import compatibility |
| `configs/main_experiment.json` | The seven model configurations, preprocessing, seeds, and training settings used in the main experiment |
| `configs/repeated_search.json` | Equal-budget GA/random/TPE settings, three seeds, and the actual six shared initial candidates for each seed |
| `configs/strict_interval.json` | Complete-interval sensitivity settings |
| `run_demo.py` | Short synthetic example suitable for Spyder |
| `run_experiment.py` | Portable entry point for separately fitted within-well models |
| `run_search.py` | Portable complete-pipeline optimizer comparison, using original search operators and training functions |
| `research_scripts/` | Original experiment, ablation, sensitivity, and statistical-analysis scripts; restricted run artifacts are not included |
| `tests/` | Synthetic tests for windows, recurrent models, representation, balancing, and strict interval isolation |
| `docs/source_provenance.json` | Original/release source hashes and the small path-only adaptations |
| `docs/reference_environment.json` | Reference scientific-package versions and computing hardware |

No field measurements, depth-label pairs, original train/test assignments, trained weights, individual predictions, manuscript drafts, or reviewer correspondence are distributed. The synthetic example is generated independently and contains no transformed or sampled field records.

## Installation

Use Python 3.12 in a dedicated environment. From this directory:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

The requirements record the reference package versions. CPU execution is supported by the portable entry points. For CUDA, install the appropriate PyTorch 2.5.1 build using the official PyTorch instructions; the original experiments used CUDA 12.4. CPU and GPU results and timings need not match exactly.

## Synthetic example

In Spyder, open and run `run_demo.py` in the environment above. Alternatively:

```bash
python run_experiment.py --demo
python run_search.py --demo
python -m pytest -q tests
```

The first example uses one artificial well, one split, one training seed, two epochs, and GA-GRU, fixed GRU, and ExtraTrees configurations. The search example uses one artificial well, one search seed, seven candidates per method, six shared initial candidates, and one training epoch. These reduced runs test execution only. Their metrics are not the paper's results and cannot be used to support its conclusions.

Each run writes to a new directory under `outputs/`. Passing `--output` selects another empty directory. Existing nonempty output directories are not overwritten. Portable launchers start a fresh run; the historical search scripts retain the original resumable execution logic.

## Locally authorized data

Prepare one CSV per well following `docs/DATA_FORMAT.md`, outside the public repository or in the ignored `private_data/` directory. Units must already be harmonized. The supplied code does not extract curves from images or repair depth misregistration.

```bash
python run_experiment.py --data-dir /path/to/authorized_csvs --output /path/to/private_results/main --device cpu
python run_search.py --data-dir /path/to/authorized_csvs --output /path/to/private_results/search --device cpu
```

Use `--device cuda` when a compatible GPU is available. Full search settings involve 24 candidates per optimizer and three independent search seeds, each candidate fitted separately to every supplied well. This is materially longer than the synthetic demonstration.

`run_experiment.py` applies the supplied published configurations; it does **not** silently run GA or replace the configurations with the winner of a new search. `run_search.py` evaluates candidates using training and validation subsets only and reports their validation results. When developing a new study, select and document configurations before final test evaluation. Do not use test scores to select among configurations.

The portable experiment adapter reuses `prepare_task`, `recurrent_run`, `extra_trees_run`, and `classification_report` from the original main runner. Imputation, standardization, and SMOTE-Tomek are fitted on the training data for epoch selection, then refitted on the training-validation union before final prediction. Balancing uses the same deterministic seed across models within each split/well. Nine repetitions are the Cartesian product of three partition seeds and three training seeds. Additional-well experiments in the paper used three paired repetitions; their dedicated original runner is retained separately.

The portable search adapter uses the original GA, random-search, and TPE operators, shared initial designs, candidate ranking, feature preparation, and model trainer. Deterministic candidate evaluations are cached within the invocation. Accounted search cost charges each optimizer the original fitting/validation cost of reused evaluations; actual execution time is reported separately. Historical CUDA-only orchestration and checkpoint management are not required by this adapter.

## Reproduction boundary

Without the proprietary datasets and recorded interval boundaries, the published numerical results cannot be independently reproduced from code alone. Exact historical reproduction additionally requires the restricted protocol snapshots and original assignments. The scripts and configurations expose the implementation and permit execution on independently authorized data, but access to those original inputs is not granted by this package.

Original hash checks in `research_scripts/` were designed for an immutable internal project. The archived scripts are not a sequence to run against an empty public checkout. See `docs/SOURCE_MAP.md` for their input dependencies and the difference between the portable entry points and historical orchestration. Restoring private artifacts and relocating paths may require a new documented provenance manifest; do not disable hash checks and describe the result as the unchanged historical run.

## Data access and code publication

The well-log and lithology-label datasets are proprietary to the oilfield institute and are not distributed. Requests for access must be directed to the responsible oilfield authority and are subject to its approval. The synthetic example does not confer access to the field data.

The repository URL and release tag will be added after publication. `docs/DATA_AND_CODE_AVAILABILITY.txt` contains bilingual manuscript wording with an explicit URL placeholder. No software license has been selected for this prepared package; public visibility and an open-source license are separate decisions. Add the rights holders' chosen license when creating the repository.
