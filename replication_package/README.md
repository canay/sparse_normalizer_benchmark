# Replication Package

## Saved evidence

- `results/main_combined_runs.csv`: 280 completed rows from the four-dataset main benchmark.
- `results/main_combined_descriptive_summary.csv`: main descriptive summary.
- `results/main_paired_vs_softmax.csv`: main paired softmax-reference tests.
- `results/main_hae_vs_entmax15.csv`: paired HAE versus entmax-1.5 tests.
- `results/kmnist_followup_runs.csv`: 70 completed KMNIST follow-up rows.
- `results/kmnist_followup_summary.csv` and `kmnist_followup_paired_tests.csv`: KMNIST summaries and paired tests.

## Code

`code/benchmark.py` preserves the saved benchmark implementation and writes new outputs below `outputs/`. `code/build_results_artifacts.py` regenerates the manuscript tables and figures from the CSV evidence.

The 20 Newsgroups path deliberately preserves the original unmasked fixed-length padding behavior. See `REPRODUCIBILITY.md` and `../docs/DATASETS_AND_LICENSES.md` before interpreting or rerunning that task.

## Quick verification

```powershell
python code/benchmark.py --help
python code/build_results_artifacts.py
```

Training is intentionally not launched by the artifact rebuild command.
