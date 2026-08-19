# Package Status

Last verified: 2026-08-19

## Release status

This repository is the public replication package for *A Training-Schedule
Crossover Accounts for the Reported Advantage of Sparse Attention Normalizers
in Compact Classifiers*.

- Repository: <https://github.com/canay/sparse_normalizer_benchmark>
- License for author-created code and documentation: MIT
- Citation metadata: `CITATION.cff`
- Release inventory: `MANIFEST.csv`
- Earlier release, superseded: tag `v1.0.1`

## Contents

The selection-stage benchmark, dataset loaders that hash every archive they
fetch, a test-evaluation tool with `freeze` and `test` subcommands, the frozen
selection record, per-cell run logs, per-epoch processed predictors, the single
test read, the design decisions with their pre-registered thresholds, the
measured environment, and a per-file SHA-256 inventory.

## Protocol

Selection happens on a held-out validation split. The test partition is read
once. Text attention is masked before the normalizer rather than after it, so
padded positions receive no attention mass and density is computed over valid
keys only.

The test path is unreachable from the selection stage. `code/test_eval_r1.py
test` refuses to run unless `outputs/selection.json` exists, hashes as
recorded, and is newer than every validation log. That is what makes the single
test read auditable rather than merely asserted.

## Environment

Measured on the host that executed every reported run:

- Ubuntu 24.04.4 LTS
- NVIDIA GeForce RTX 5060, 8151 MiB, driver 595.71.05
- Python 3.12.3
- PyTorch 2.13.0+cu130, CUDA 13.0, cuDNN 9.2.0
- NumPy 1.26.4

The benchmark imports torch and numpy only.

## Verification status

Every file carries a SHA-256 and a byte size in `MANIFEST.csv`. The recorded
values were computed from the blobs as published rather than from a local
working copy, because this repository normalises line endings on write and a
manifest hashed against a Windows checkout would list bytes no downloader
receives.

Checks that pass:

- Python source compilation for all three modules;
- `MANIFEST.csv` agreeing with every published blob;
- `outputs/selection.json` parsing and carrying its recorded freeze fields; and
- no host path, private hostname, credential, or key pattern anywhere in the
  published tree.

## Relationship to the earlier round

An earlier round of this benchmark selected its settings from test evidence and
left padded text slots unmasked. The current study replicates that round under
its own regime, reproduces its direction and ordering for the top-k family, and
then shows the result inverts once the training budget grows.

That round's package is no longer carried here. Table 4 of the manuscript
quotes six paired differences from its run logs, so those six rows are retained
in `docs/earlier_run_records.csv` with their original source-file provenance.
Everything else from that round remains reachable through git history at tag
`v1.0.1`.

## Evidence boundary

Findings are bounded to compact classifiers under the evaluated protocol. They
do not speak to large pretrained models, long-context generation, or
accelerator implementations, and no claim of architecture independence is made.
The crossing range of nine to twelve epochs is a measurement on CIFAR-10 rather
than a constant.

Dataset archives are acquired from their upstream providers and are not
redistributed in this repository.
