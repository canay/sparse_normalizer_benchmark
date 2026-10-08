# Extension evidence

This folder holds the code and processed run records behind the analyses that extend the initial benchmark at the repository root. The initial benchmark files are unchanged.

| Folder | What it contains | Where it is reported |
| --- | --- | --- |
| `m1_m3_crossover_schedule/` | The ten-task, two-architecture, three-capacity trajectory grid, the early-feature predictor and switching-schedule discovery, and the confirmation-validation and single frozen test read. Training and analysis programs, saved run rows, analysis outputs, registration bindings and a saved-output verifier. | Main text; its own `README.md` gives the verification command |
| `n3a_synthetic_assay/` | The registered synthetic intervention assay (6,144 units): protocol, configuration, source, tests, per-unit outputs, the terminal scientific decision and an independent recomputation. | Main text; Supplementary Evidence, Section S5 |
| `matched_bottomk_control/` | The current-runtime matched full-budget control (50 cells, ten paired seeds, common 30-epoch and patience-zero policy) and the historical matched-patience readout of the same seed trajectories. | Main text; Supplementary Evidence, Sections S2 and S6 |
| `lr_policy_control/` | The exploratory, untuned budget-normalized linear-decay learning-rate control: protocol, source, per-cell records, decision record and result report. | Main text; Supplementary Evidence, Section S10 |
| `saved_seed_diagnostics/` | Descriptive diagnostics recomputed from saved records: seed contrasts, leave-one-seed-out checks, seed-cluster summaries and selection-metric comparisons. | Supplementary Evidence |
| `supplement_tables/` | The derived CSV files written when the Supplementary Evidence was generated (line endings normalized to LF). | Supplementary Evidence |

## Verifying without training

The M1-M3 grid can be checked from saved rows alone:

```text
cd m1_m3_crossover_schedule
python -m pip install -r requirements-verification.txt
python verify_saved_outputs.py --output-dir verification_output
```

Each other folder keeps its decision or analysis record (`SCIENTIFIC_DECISION.json`, or `ANALYSIS_MANIFEST.json` for the saved-seed diagnostics) next to the per-cell or per-unit outputs it summarizes. The N3-A folder also carries `independent_check/recompute.py` and its output.

## What is not here, and why

Model checkpoints, array dumps, archive bundles, engineering smoke runs, compute-resource receipts and host-specific launch scripts are not redistributed. The raw N3-A unit archive (about 15 GB) is not redistributed either; its size and SHA-256 are recorded in `EXCLUSION_DISPOSITION.json`, together with the count of every excluded file class. Public benchmark datasets are downloaded from their upstream providers by the loaders.

## Placeholders

Before publication, local and remote directory prefixes, physical host names and one private server address were replaced by the placeholders listed in `SANITIZATION_MANIFEST.json` (for example `<gpu-server-home>`, `<controlled-workspace>`, `workstation-a`). Result values were not edited. For every affected file the manifest records the SHA-256 before and after the replacement. The saved-output verifier above gives identical results on the original and the published copy.

## Integrity

`SHA256SUMS` lists the SHA-256 of every file in this folder.

## Evidence boundary

These records support the bounded comparisons reported in the article. The early-feature predictor, the switching schedules and the synthetic assay fail their comparative or registered criteria, and those negative outcomes are kept as recorded. Calibrated crossover prevalence, a causal mechanism, a general selection rule and runtime speedup are not established by these records.
