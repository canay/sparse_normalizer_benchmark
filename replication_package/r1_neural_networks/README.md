# r1 replication package (Neural Networks submission)

This directory holds every artifact behind the revised manuscript. It is
separate from the earlier package because the protocol changed: selection now
happens on a held-out validation split, the test partition is read once, and
text attention is masked before the normalizer rather than after it.

## Layout

| Path | Contents |
| --- | --- |
| `code/benchmark_r1.py` | selection-stage benchmark; contains no test path by design |
| `code/datasets_r1.py` | dataset loaders with SHA-256 provenance for every archive |
| `code/test_eval_r1.py` | `freeze` and `test` subcommands; refuses to run without a hash-verified selection record newer than every validation log |
| `outputs/selection.json` | the frozen selection, written before any test evaluation |
| `outputs/raw/*.jsonl` | one record per run cell |
| `outputs/processed/*.csv` | per-epoch predictors derived from the raw logs |
| `outputs/test/*.jsonl` | the single test read |
| `docs/DESIGN_DECISIONS.md` | the four executor decisions and their pre-registered thresholds |
| `environment.json` | the measured software and hardware environment |
| `MANIFEST.csv` | SHA-256 of every file in this directory |

## Reproducing

```
python -m pip install -r requirements.txt
python code/benchmark_r1.py --help
```

The dataset archives are not redistributed. `code/datasets_r1.py` fetches each
from its canonical source and checks the archive against the SHA-256 recorded
in the run manifests, so an independent run can confirm it obtained the same
bytes.

The test path is deliberately unreachable from the selection stage.
`test_eval_r1.py test` refuses to run unless `outputs/selection.json` exists,
hashes as recorded, and is newer than every validation log, which is what makes
the single test read auditable rather than merely asserted.

## Failed cells, and one edit to the published logs

`outputs/raw/r1_select_20260817_172222_grid_compact.jsonl` records 296 failed
cells out of 1875. They come from a single CUDA launch-timeout cascade on
2026-08-17: the host's GPU also drove its display, one kernel exceeded the
watchdog, and the poisoned CUDA context failed every subsequent cell in the
same process. The affected cells were re-run afterwards under per-dataset
processes with a smaller batch, and the reported numbers come from those
re-runs. The failures are kept rather than deleted because a run record that
hides its failures is not a run record.

Each failed cell carries a `traceback`. In the published copy the leading
directories of every path in those tracebacks are replaced with `<package>/`
and `<site-packages>/`. The exception type, message, module, function, and line
number are unchanged. The reason is that the host's interpreter had an
unrelated project's virtual environment on its path, so the raw tracebacks
disclosed that project's directory tree and nothing about this study. No other
file was altered, and no field other than `traceback` was touched.
