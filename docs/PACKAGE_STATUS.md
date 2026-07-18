# Package Status

Last verified: 2026-07-18

## Release status

This repository is the public replication package for *Diagnostic Benchmark
of Sparse Attention Normalizers in Compact Classification*.

- Repository: <https://github.com/canay/sparse_normalizer_benchmark>
- Published release: `v1.0.0`
- License for author-created code and documentation: MIT
- Citation metadata: `CITATION.cff`
- Release inventory: `MANIFEST.csv`

## Package contents

The package contains cleaned relative-path benchmark code, saved run-level and
processed evidence, a deterministic table/figure rebuild script, regenerated
manuscript artifacts, pinned environment versions, and upstream dataset and
license notes.

The saved evidence is complete for the reported study: 280 main benchmark rows
(four datasets, seven normalizers, ten matched seeds) and 70 KMNIST follow-up
rows (one dataset, seven normalizers, ten matched seeds). No additional
training run is required to rebuild the included tables and figures.

## Verification status

The following checks pass for the published package:

- Python source compilation;
- benchmark command-line help without training;
- artifact-only table and figure rebuild from saved CSV evidence;
- saved-grid, paired-test, Holm-adjustment, and manuscript-table identity
  checks; and
- SHA-256 inventory verification against `MANIFEST.csv`.

## Evidence boundary

The 20 Newsgroups implementation uses unmasked position-bearing padded slots
and computes density over all slots. Its saved comparisons support the reported
implementation-specific findings, but do not establish masked-text behavior or
valid-token-only density. Dataset archives are acquired from their upstream
providers and are not redistributed in this repository.
