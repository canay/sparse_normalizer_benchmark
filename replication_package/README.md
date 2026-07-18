# Replication Package

## Requirements

- Python 3.12 (the recorded runs used Python 3.12.12)
- Dependency versions pinned in `requirements.txt`
- A CUDA-capable GPU for practical full reruns; the saved artifact rebuild is CPU-safe

## Install

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

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

## Run

```powershell
.venv\Scripts\python code\benchmark.py --help
.venv\Scripts\python code\build_results_artifacts.py
```

Training is intentionally not launched by the artifact rebuild command.

## Expected output

The artifact-only command rewrites five LaTeX tables under `tables/` and four
plot files under `figures/` from the saved CSV evidence. A successful run ends
with a message reporting the output directory. The command does not download
datasets or create new training results.

For the exact full-rerun commands, recorded environment, and reproducibility
boundary, see `REPRODUCIBILITY.md`.
