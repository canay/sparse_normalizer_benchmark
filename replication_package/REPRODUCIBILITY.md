# Reproducibility

## Recorded environment

- Python 3.12.12
- Package versions: see `requirements.txt`
- Source-run device: NVIDIA GeForce GTX 1050 with CUDA
- Operating system used for source runs: Windows

Create an isolated environment and install the recorded dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Artifact-only rebuild

This command performs no training and regenerates all manuscript tables and figures from the saved CSV evidence:

```powershell
.venv\Scripts\python code\build_results_artifacts.py
```

## Main four-dataset benchmark

```powershell
.venv\Scripts\python code\benchmark.py `
  --datasets fashion_mnist cifar10 twenty_news synthetic_marker `
  --seeds 0 1 2 3 4 5 6 7 8 9 `
  --epochs 6 --train-limit 8000 --test-limit 2000 `
  --text-train-limit 4000 --text-test-limit 1500 `
  --synthetic-train-limit 8000 --synthetic-test-limit 2000 `
  --seq-len 96
```

## KMNIST follow-up

```powershell
.venv\Scripts\python code\benchmark.py `
  --datasets kmnist --seeds 0 1 2 3 4 5 6 7 8 9 `
  --epochs 8 --train-limit 12000 --test-limit 3000
```

New runs are written below `outputs/`; dataset caches remain below `data_cache/`. Full reruns can differ across hardware and library builds despite fixed seeds. The saved CSV files are the evidence used to regenerate the manuscript artifacts.
