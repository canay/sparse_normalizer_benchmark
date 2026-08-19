# Package Status

Last verified: 2026-08-19

## Release status

This repository is the public replication package for *A Training-Schedule
Crossover Accounts for the Reported Advantage of Sparse Attention Normalizers
in Compact Classifiers*.

- Repository: <https://github.com/canay/sparse_normalizer_benchmark>
- Previous published release: `v1.0.1`, first round only
- License for author-created code and documentation: MIT
- Citation metadata: `CITATION.cff`
- Release inventory: `MANIFEST.csv`

## Rounds

The repository holds two rounds. The second is what the current manuscript
reports. The first is retained unedited because the second round replicates it,
and a replication is only checkable while the earlier record stays visible.

### Second round, `replication_package/r1_neural_networks/`

Added 2026-08-19. Selection happens on a held-out validation split, the test
partition is read once, and text attention is masked before the normalizer
rather than after it.

Contents: the selection-stage benchmark, dataset loaders that hash every
archive they fetch, a test-evaluation tool with `freeze` and `test`
subcommands, the frozen selection record, per-cell run logs, per-epoch
processed predictors, the single test read, the design decisions with their
pre-registered thresholds, the measured environment, and a per-file SHA-256
manifest.

The test path is unreachable from the selection stage. `test_eval_r1.py test`
refuses to run unless `outputs/selection.json` exists, hashes as recorded, and
is newer than every validation log. This is what makes the single test read
auditable rather than merely asserted.

Measured environment of every reported second-round run: Ubuntu 24.04.4 LTS,
NVIDIA GeForce RTX 5060 with 8151 MiB, driver 595.71.05, Python 3.12.3, PyTorch
2.13.0+cu130, CUDA 13.0, cuDNN 9.2.0, NumPy 1.26.4. The benchmark imports torch
and numpy only.

### First round

Cleaned relative-path benchmark code, saved run-level and processed evidence, a
deterministic table and figure rebuild script, regenerated manuscript
artifacts, pinned environment versions, and upstream dataset and license notes.

The saved evidence is complete for what that round reported: 280 main benchmark
rows (four datasets, seven normalizers, ten matched seeds) and 70 KMNIST
follow-up rows (one dataset, seven normalizers, ten matched seeds). No
additional training run is required to rebuild the included tables and figures.

## Verification status

The following checks pass for the first-round package:

- Python source compilation;
- benchmark command-line help without training;
- artifact-only table and figure rebuild from saved CSV evidence;
- saved-grid, paired-test, Holm-adjustment, and manuscript-table identity
  checks; and
- SHA-256 inventory verification against `MANIFEST.csv`.

For the second-round package, every file carries a SHA-256 in
`replication_package/r1_neural_networks/MANIFEST.csv` and in the root
`MANIFEST.csv`, and the recorded values were computed from the files as
published.

## Evidence boundary

**Second round.** Findings are bounded to compact classifiers under the
evaluated protocol. They do not speak to large pretrained models, long-context
generation, or accelerator implementations, and no claim of architecture
independence is made. The crossing range of nine to twelve epochs is a
measurement on CIFAR-10 rather than a constant.

**First round.** The 20 Newsgroups implementation uses unmasked
position-bearing padded slots and computes density over all slots. Its saved
comparisons support the findings that round reported, but do not establish
masked-text behavior or valid-token-only density. This is the defect the second
round was built to remove.

Dataset archives are acquired from their upstream providers and are not
redistributed in this repository.
