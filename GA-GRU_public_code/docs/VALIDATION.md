# Release checks

Checked on 23 September 2026 using the existing scientific Python 3.12 environment. These checks validate software operation; they are not additional manuscript experiments.

- 27 original synthetic tests passed for random-center windows, balancing, recurrent models, representations, and complete-interval isolation.
- Six public-input validation tests passed for schema acceptance, duplicate depth, sampling grid, interval identity, path-safe well IDs, missing curves, and nonpositive resistivity.
- The portable main runner completed all seven configured model families on independently generated synthetic data, using one partition, one training seed, and two selection epochs.
- The portable search runner completed GA, random search, and TPE on that synthetic example, using one search seed, seven candidates per method, six common initial candidates, and one training epoch. This exercises candidate generation beyond the shared initial population.
- All distributed Python files passed parsing and compilation checks. The copied-source hashes match the release provenance manifest.
- The release inventory contains source code, configuration, documentation, and synthetic tests only. No CSV/Excel/NumPy data, trained weights, notebooks with outputs, individual predictions, or restricted protocol snapshots are included.
- The original proprietary datasets, long-running full search budgets, archived historical experiment orchestration, and GPU timing reproduction were not rerun for this packaging task.

One original test run emitted a Windows physical-core detection warning from joblib; the tests passed. Portable entry points set a worker-count environment value. No synthetic performance scores are presented here as geological evidence.
