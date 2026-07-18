# Sparse Normalizer Benchmark

Replication materials for **Task-Conditional Accuracy–Support Trade-offs of Sparse Attention Normalizers in Compact Classifiers** by Özkan Canay, Department of Information Systems and Technologies, Sakarya University.

The study compares softmax, sparsemax, entmax-1.5, fixed-ratio top-k softmax, and head-wise adaptive entmax in compact classifiers. It reports ten-seed predictive summaries, paired comparisons, attention-support density, and implementation-specific timing diagnostics.

## Contents

- `replication_package/code/benchmark.py`: training and evaluation implementation.
- `replication_package/code/build_results_artifacts.py`: table and figure rebuild.
- `replication_package/results/`: saved run-level and processed evidence.
- `replication_package/tables/` and `figures/`: regenerated manuscript artifacts.
- `replication_package/REPRODUCIBILITY.md`: environment and commands.
- `docs/DATASETS_AND_LICENSES.md`: acquisition and license notes.
- `MANIFEST.csv`: release file inventory with SHA-256 hashes.

## Evidence boundary

The 20 Newsgroups implementation uses unmasked position-bearing padded slots and computes density over all slots. Its accuracy comparisons are reproducible for that implementation, but they do not establish masked-text behavior or valid-token-only density. Public dataset archives are downloaded through upstream loaders and are not redistributed.

## License and citation

Author-created code and package text are released under the MIT License. Third-party datasets retain their original terms. Cite the accompanying manuscript using `CITATION.cff`.
