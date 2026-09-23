# Mapping to the revised manuscript

The numbered scripts preserve the original implementation. Most remain byte-identical to their source. `source_provenance.json` lists exact hashes and the path-only changes. New portable adapters are separately identifiable at the repository root; they do not claim to be the original historical orchestrators.

| Manuscript component | Original implementation |
| --- | --- |
| Random-center target assignment and context audit | `gagru/random_center.py`, scripts 67-68 |
| Initial model screening and first optimizer comparison | scripts 69-76; the earlier comparison used its own candidate budget and balancing settings |
| Input representation and balancing ablation | `gagru/residual_bigru.py` feature functions; `gagru/random_center_ablation.py`; scripts 77-78 |
| Main repeated evaluation | scripts 79-81; `gagru/local_recurrent.py` |
| Fixed/legacy recurrent supplementary comparisons and efficiency | scripts 90-93 |
| Complete-interval separation with 1.0-m guards | `gagru/strict_interval.py`; scripts 94-99 |
| Three independent full-pipeline searches per optimizer | scripts 113-115; `configs/repeated_search.json` |
| Additional-well evaluation with unchanged model configurations | `research_scripts/run_supplement_v1.py` |
| Paired well-level Wilcoxon tests, Holm adjustment and well-cluster bootstrap | `81_analyze_final_evaluation.py`, `99_analyze_strict_evaluation.py` |

## Main model versus historical modules

The primary fourth-manuscript architecture is `LocalRecurrentClassifier(architecture='gru')`, a unidirectional GRU with a linear head on the last sequence output. `residual_bigru.py` is imported for reusable feature preparation and data-loader helpers; retaining that module does not mean the paper's main model is a residual bidirectional GRU. Earlier generic cross-well helpers in `gagru/` are also not the main evaluation protocol.

## Dependencies that are intentionally not public

- Scripts 67-68 require the earlier prepared data and eligibility records. The public adapter regenerates random-center assignments from an independently supplied normalized CSV.
- Historical scripts 69-81 require the v5 private snapshots, center-assignment tables, and model-selection manifests.
- Scripts 90-93 require original result manifests; the legacy-parameter provenance step also refers to private reference notebooks/documents. Those documents and embedded notebook outputs are not distributed.
- Scripts 94-99 require v5 snapshots, original interval/partition records, and the inherited evaluation configurations. Their original schema includes recorded `interval_top` and `interval_bottom` boundaries and optional global lithology-family mapping. The portable main CSV schema alone is not a complete historical strict-interval protocol.
- Scripts 113-115 require the private v5 protocol and v8 provenance manifests. The public `run_search.py` generates its own local preparation records and directly calls the same operators and full-pipeline candidate trainer.
- The additional-well runner requires the harmonized additional-well files and the main evaluation plan. It retains three paired split/training repetitions, local lithology codes, and no new hyperparameter search. Its project and data roots can be configured with `GAGRU_RESEARCH_PROJECT` and `GAGRU_SUPPLEMENT_DATA`.

The historical numbered scripts expect their original project-relative arrangement. For authorized historical reproduction, restore them to a dedicated project root with `gagru/`, restricted protocols and private run manifests. Exact source-hash checks will also need the documented historical source version. Do not mix new configurations with old results or relabel a newly assembled run as the original experiment.

## Portable adapters

`run_experiment.py` calls the main runner's preparation, training, refitting, prediction, and metric functions without rewriting their scientific logic. Its new responsibilities are normalized CSV validation, generation of local protocols, CLI settings, output bookkeeping, and a short synthetic mode. It does not recreate the prior exploratory selection history.

`run_search.py` calls the original `prepare_tasks`, `run_ga`, `run_random`, `run_tpe`, and `rank`, plus the same GRU trainer. It supports CPU execution and a fresh output directory rather than requiring the historical CUDA-only manifest. Shared evaluations are charged to each optimizer at their original fitting cost, following the paper's accounted-cost definition. Timing can change candidate ordering in an exact score-and-size tie, as in the original implementation.

## Fourth-version supplementary analyses

This package exposes the model, search, evaluation, and original statistical implementations. Manuscript-building scripts, document renders, figure source datasets, and supplementary per-sample verification bundles are not public code dependencies and were not copied. Those bundles may contain depth-label or prediction records and must not be uploaded merely because their filenames say "supplementary".
