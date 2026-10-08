# Sparse Normalizer Benchmark

Replication package for **Training-Budget Dependence of Sparse Attention Normalizers in Compact Classifiers under a Fixed Learning Rate** by Özkan Canay, Department of Information Systems and Technologies, Sakarya University.

The study compares dense softmax against six sparse normalizers, fixed-ratio top-k softmax at retained fractions of 0.125, 0.25 and 0.5, sparsemax, entmax-1.5, and head-wise adaptive entmax, in compact image, text, and synthetic-marker classifiers under a fixed learning rate. The initial grid covers seven tasks with ten matched seeds; a validation-only extension with a predeclared power condition expands the full-budget analysis to thirteen tasks.

On CIFAR-10, the top-k (r = 0.25) accuracy difference from dense softmax moves from +2.25 to -1.74 percentage points between three and thirty epochs along reproducible training prefixes. Under the uniform full-budget protocol, only USPS among thirteen tasks shows a Holm-supported sparse advantage, so the predeclared positive-class floor is not met and no normalizer-selection rule is fitted. The analyses in `extension/` add a ten-task, two-architecture, three-capacity grid, tests of an early-feature predictor and switching schedules, a synthetic intervention assay, matched-policy bottom-k controls and an exploratory linear-decay learning-rate control. The predictor, the switching schedules and the synthetic assay fail their comparative or registered criteria.

## Contents

| Path | Contents |
| --- | --- |
| `code/benchmark_r1.py` | the selection-stage benchmark. It contains no test path by design |
| `code/datasets_r1.py` | dataset loaders that record a SHA-256 for every archive they fetch |
| `code/datasets_e5b.py`, `code/run_e5b.py` | loaders and runner for the six added extension tasks |
| `code/e5_rule.py` | the frozen label, power-condition and conditional rule analysis |
| `code/build_*_table.py` | deterministic generators for the full-budget and added-task result tables |
| `code/test_eval_r1.py` | `freeze` and `test` subcommands for the single test read |
| `outputs/selection.json` | the frozen selection, written before any test evaluation |
| `outputs/raw/` | one record per run cell |
| `outputs/processed/` | per-epoch predictors, paired summaries and run manifests |
| `outputs/test/` | the single test read |
| `extension/` | code, processed run records, decision records and verification for the extension analyses; see `extension/README.md` |
| `docs/DESIGN_DECISIONS.md` | the four executor decisions and their predeclared thresholds |
| `docs/E5_PROTOCOL.md`, `docs/E5_RESULT.md` | public protocol and outcome record for the task expansion |
| `docs/e5b_rule_output.txt` | unedited stdout of the predeclared power check |
| `docs/DATASETS_AND_LICENSES.md` | acquisition sources and terms for each dataset |
| `docs/earlier_run_records.csv` | the six earlier values Table 4 of the manuscript quotes |
| `environment.json` | the measured software and hardware environment |
| `MANIFEST.csv` | SHA-256 and size of every file outside `extension/` |
| `extension/SHA256SUMS` | SHA-256 of every file in `extension/` |

## Running it

```
python -m pip install -r requirements.txt
python code/benchmark_r1.py --help
python code/run_e5b.py --help
python code/e5_rule.py outputs/raw
```

The pinned runtime dependencies are `torch`, `numpy`, and `scipy`; SciPy is used only to parse SVHN's official MATLAB archive. The other loaders fetch canonical archives directly rather than through torchvision or scikit-learn.

The extension grid can be checked from saved rows without training; the command is in `extension/README.md`.

## The single test read

The test path is unreachable from the selection stage. `code/test_eval_r1.py test` refuses to run unless `outputs/selection.json` exists, hashes as recorded, and is newer than every validation log. That is what makes the single test read auditable rather than merely asserted.

## Failed cells, and one edit to the published logs

`outputs/raw/r1_select_20260817_172222_grid_compact.jsonl` records 296 failed cells out of 1875. They come from a single CUDA launch-timeout cascade: the host's GPU also drove its display, one kernel exceeded the watchdog, and the poisoned CUDA context failed every subsequent cell in the same process. Those cells were re-run afterwards under per-dataset processes with a smaller batch, and the reported numbers come from the re-runs. The failures are kept rather than deleted, because a run record that hides its failures is not a run record.

Each failed cell carries a `traceback`. In the published copy the leading directories of every path in those tracebacks are replaced with `<package>/` and `<site-packages>/`. The exception type, message, module, function and line number are unchanged. The host interpreter had an unrelated project's virtual environment on its path, so the raw tracebacks disclosed that project's directory tree and nothing about this study. No other file outside `extension/` was altered and no field other than `traceback` was touched. The placeholders used inside `extension/` are listed in `extension/SANITIZATION_MANIFEST.json`.

## Uniform batch-size repair and E5-B extension

A pre-submission manifest audit found that MNIST, Fashion-MNIST, KMNIST and the first execution of the six extension tasks had used batch 512 instead of the predeclared batch 256. The three core tasks and all six extension tasks were rerun at 256. The published evidence keeps the earlier core-grid record for provenance and adds a later `grid_batch256_repair` record that mechanically supersedes those three task rows; only the final batch-256 executions of the six extension tasks are included.

The corrected evidence base contains 13 tasks x 7 methods x 10 seeds = 910 completed full-budget cells, all at batch 256. The correction removes the two previously supported positive task labels from Fashion-MNIST and KMNIST. USPS is the only supported positive among thirteen tasks, so the power condition fixed before the extension triggers and the rule is not fitted. `docs/E5_PROTOCOL.md` states the frozen design and `docs/E5_RESULT.md` records the bounded outcome.

## Earlier round

An earlier round of this benchmark selected its settings from test evidence and left padded slots unmasked in the text task. That round is what the current study replicates and then supersedes, and its package is no longer carried here.

Table 4 of the manuscript quotes six paired differences from the earlier run logs, so those six rows are retained in `docs/earlier_run_records.csv` with their original source-file provenance. Everything else from that round is reachable through this repository's git history at tag `v1.0.1`.

## Evidence boundary

Findings are bounded to compact classifiers under the evaluated protocol and a fixed learning rate. They do not speak to large pretrained models, to long-context generation, or to accelerator implementations, and no claim of architecture independence is made. The CIFAR-10 sign change between nine and twelve epochs is a measurement in this setting rather than a constant. Calibrated crossover prevalence, a causal mechanism, a general selection rule and runtime speedup are not established.

Public dataset archives are downloaded from their upstream sources and are not redistributed here. The loaders record the SHA-256 of every archive they fetch, so an independent run can verify it obtained the same bytes.

## License and citation

Author-created code and package text are released under the MIT License. Third-party datasets retain their original terms. Cite the accompanying manuscript using `CITATION.cff`.
