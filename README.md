# Sparse Normalizer Benchmark

Replication materials for **A Training-Schedule Crossover Accounts for the Reported Advantage of Sparse Attention Normalizers in Compact Classifiers** by Özkan Canay, Department of Information Systems and Technologies, Sakarya University.

The study compares dense softmax against six sparse normalizers, fixed-ratio top-k softmax at three retained fractions, sparsemax, entmax-1.5, and head-wise adaptive entmax, in a compact Transformer-style classifier across seven datasets and ten matched seeds. It finds that the advantage sparse normalizers are reported to hold over dense softmax reproduces under a short training schedule and inverts under a longer one, crossing zero between nine and twelve epochs.

## Two rounds, both kept

This repository holds the evidence from two rounds of the benchmark. The second round is the one the current manuscript reports; the first is retained because the second replicates it and the replication is only checkable if the earlier record stays visible.

| Directory | Round | Protocol |
| --- | --- | --- |
| `replication_package/r1_neural_networks/` | second, current | selection on a held-out validation split, test partition read once, text attention masked before the normalizer |
| `replication_package/code`, `results`, `tables`, `figures` | first | selection read from test evidence, 20 Newsgroups padded slots unmasked |

The first round's evidence boundary is stated below and has not been edited. The second round exists because that boundary turned out to matter.

## Second round contents

- `replication_package/r1_neural_networks/code/benchmark_r1.py`: the selection-stage benchmark. It contains no test path by design.
- `replication_package/r1_neural_networks/code/datasets_r1.py`: dataset loaders that record a SHA-256 for every archive they fetch.
- `replication_package/r1_neural_networks/code/test_eval_r1.py`: `freeze` and `test` subcommands. `test` refuses to run unless the selection record exists, hashes as recorded, and is newer than every validation log.
- `replication_package/r1_neural_networks/outputs/selection.json`: the frozen selection, written before any test evaluation.
- `replication_package/r1_neural_networks/outputs/raw/`: one record per run cell.
- `replication_package/r1_neural_networks/outputs/processed/`: per-epoch predictors derived from the raw logs.
- `replication_package/r1_neural_networks/outputs/test/`: the single test read.
- `replication_package/r1_neural_networks/docs/DESIGN_DECISIONS.md`: the four executor decisions and their pre-registered thresholds.
- `replication_package/r1_neural_networks/environment.json`: the measured software and hardware environment.
- `replication_package/r1_neural_networks/MANIFEST.csv`: SHA-256 of every file in that directory.

## First round contents

- `replication_package/code/benchmark.py`: training and evaluation implementation.
- `replication_package/code/build_results_artifacts.py`: table and figure rebuild.
- `replication_package/results/`: saved run-level and processed evidence.
- `replication_package/tables/` and `figures/`: regenerated manuscript artifacts.
- `replication_package/REPRODUCIBILITY.md`: environment and commands.

## Shared

- `docs/DATASETS_AND_LICENSES.md`: acquisition and license notes.
- `docs/PACKAGE_STATUS.md`: release status and verification record.
- `MANIFEST.csv`: release file inventory with SHA-256 hashes.

## Evidence boundary

**Second round.** Findings are bounded to compact classifiers under the evaluated protocol. They do not speak to large pretrained models, to long-context generation, or to accelerator implementations, and no claim of architecture independence is made. The crossing range of nine to twelve epochs is a measurement on CIFAR-10 rather than a constant.

**First round.** The 20 Newsgroups implementation uses unmasked position-bearing padded slots and computes density over all slots. Its accuracy comparisons are reproducible for that implementation, but they do not establish masked-text behavior or valid-token-only density. This is the defect the second round was built to remove.

Public dataset archives are downloaded through upstream loaders and are not redistributed by this repository. Each round's loaders record the SHA-256 of every archive they fetch, so an independent run can verify it obtained the same bytes.

## License and citation

Author-created code and package text are released under the MIT License. Third-party datasets retain their original terms. Cite the accompanying manuscript using `CITATION.cff`.
