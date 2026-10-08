# Package Status

Last verified: 2026-10-08

## Release status

This repository is the public replication package for *Training-Budget Dependence of Sparse Attention Normalizers in Compact Classifiers under a Fixed Learning Rate*.

- Repository: <https://github.com/canay/sparse_normalizer_benchmark>
- License for author-created code and documentation: MIT
- Citation metadata: `CITATION.cff`
- Release inventory: `MANIFEST.csv` (files outside `extension/`) and `extension/SHA256SUMS`
- Earlier release, superseded: tag `v1.0.1`

## Contents

The selection-stage benchmark, dataset loaders that hash every archive they
fetch, the E5-B task-expansion loaders and runner, the frozen label/power-check
analysis, a test-evaluation tool with `freeze` and `test` subcommands, the frozen
selection record, per-cell run logs, per-epoch processed predictors, paired
summaries, run manifests, deterministic table generators, the single test read, the
design decisions with their predeclared thresholds, the measured
environment, and a per-file SHA-256 inventory.

`extension/` adds the code and processed run records of the extension
analyses: the ten-task, two-architecture, three-capacity trajectory grid with the
early-feature predictor, switching schedules and confirmation; the registered
synthetic intervention assay; the matched-policy bottom-k controls; the
exploratory budget-normalized linear-decay learning-rate control; and saved-seed
diagnostics. `extension/README.md` describes each folder and what is excluded.

## Protocol

Selection happens on a held-out validation split. The test partition is read
once. Text attention is masked before the normalizer rather than after it, so
padded positions receive no attention mass and density is computed over valid
keys only.

The test path is unreachable from the selection stage. `code/test_eval_r1.py
test` refuses to run unless `outputs/selection.json` exists, hashes as
recorded, and is newer than every validation log. That is what makes the single
test read auditable rather than merely asserted.

The full-budget comparison is uniform at batch 256: 13 tasks x 7 methods x 10
seeds, 910 completed cells. A later `grid_batch256_repair` record supersedes the
three affected core-grid task rows, and only the final batch-256 executions of
the six extension tasks are present. USPS is the sole Holm-supported positive;
the predeclared positive-class floor is not met and no selection rule is
fitted.

## Environment

Measured on the host that executed every initial-benchmark run:

- Ubuntu 24.04.4 LTS
- NVIDIA GeForce RTX 5060, 8151 MiB, driver 595.71.05
- Python 3.12.3
- PyTorch 2.13.0+cu130, CUDA 13.0, cuDNN 9.2.0
- NumPy 1.26.4

The benchmark imports torch and numpy only. The extension folders record their
own environments in their run manifests.

## Verification status

Every file outside `extension/` carries a SHA-256 and a byte size in
`MANIFEST.csv`; every file inside it is listed in `extension/SHA256SUMS`. The
recorded values were computed from the blobs as published rather than from a
local working copy, because this repository normalises line endings on write
and a manifest hashed against a Windows checkout would list bytes no downloader
receives.

Checks that pass:

- Python source compilation for all eight modules;
- `MANIFEST.csv` agreeing with every published blob;
- every added run manifest recording batch 256, the expected completed-cell
  count, and zero failed cells;
- the frozen E5 power check reproducing thirteen labels with one positive and
  exit code 2;
- `outputs/selection.json` parsing and carrying its recorded freeze fields;
- `extension/SHA256SUMS` agreeing with every file it lists;
- the extension grid's saved-output verifier giving identical results on the
  original records and on the published copy after the placeholder replacement
  recorded in `extension/SANITIZATION_MANIFEST.json`; and
- no host path, private hostname, credential, or key pattern anywhere in the
  published tree.

## Relationship to the earlier round

An earlier round of this benchmark selected its settings from test evidence and
left padded text slots unmasked. The current study replicates that round under
its own regime, reproduces its direction and ordering for the top-k family, and
then shows that the top-k differences change mean sign as the training budget
grows under a fixed learning rate.

That round's package is no longer carried here. Table 4 of the manuscript
quotes six paired differences from its run logs, so those six rows are retained
in `docs/earlier_run_records.csv` with their original source-file provenance.
Everything else from that round remains reachable through git history at tag
`v1.0.1`.

## Evidence boundary

Findings are bounded to compact classifiers under the evaluated protocol and a
fixed learning rate. They do not speak to large pretrained models, long-context
generation, or accelerator implementations, and no claim of architecture
independence is made. The CIFAR-10 sign change between nine and twelve epochs
is a measurement in this setting rather than a constant.

Dataset archives are acquired from their upstream providers and are not
redistributed in this repository.
