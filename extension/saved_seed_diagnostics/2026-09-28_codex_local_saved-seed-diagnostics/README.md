# Saved seed diagnostics: analysis plan

Date/time: 2026-09-28 03:20 +03:00  
Tool: Codex  
Model: gpt-6-astra / xhigh  
Operation ID: snd-approved-b-repair-20260928

Status: PREPARED_NOT_EXECUTED. Register this post-hoc analysis before execution, after the independent design-review guard has frozen its terminal result. Canonical state, registry, original source and old result files remain untouched while that review reads them.

Purpose: F01/F03/F04/F05/F06 evidence-conservative disclosure from existing saved results, no training, test evaluation, label access, threshold adjustment, model selection or registered-outcome revision.

Existing-tool inventory: central `tool_inventory.py uncertainty --project .` returned no suitable combined saved-seed diagnostic producer. Existing analyze_m1.py implements the registered target/predictor analysis; e5_rule.py owns raw-grid selection. The new script reuses their population definitions and raw-grid merge, cross-checks independent processed CSVs, and adds descriptive seed dispersion and sensitivity exports they do not provide. It does not replace or rewrite either analysis.

Outputs planned: M1 per-seed contrasts and per-cell descriptive dispersion/leave-one-seed-out targets; validation accuracy and accuracy-selected F1 summary over 910 saved cells; initial frozen-test 350-cell seed summary; the 70-cell CIFAR replicated validation corner; M3 saved-test seed-cluster accuracy/F1 summary and exact three-cluster bootstrap support; M3 validation raw/pairwise verification of its registered time-to-target reduction and the corresponding three-cluster bootstrap. Nominal paired intervals for n=10 are exploratory and unadjusted; they never replace original Holm tests or decisions. M1 selected-window statistics are post-hoc and selection-naive; no calibrated crossover or population prevalence is certified.

Every consumed file and producing source is hashed; raw versus independently processed metrics must match, and cardinality/configuration assertions fail closed. Completed originals remain immutable. The script writes new derived artifacts only into its dated run and refuses an existing output directory. CLI transcript and final analysis manifest are required before manuscript use.

F03 also checks all fields of the 300 shorter-budget JSON epoch logs against their corresponding 30-epoch trajectory prefixes. Separate execution is preserved as a historical fact; exact prefix agreement supports the narrower constant-learning-rate trajectory-prefix estimand and does not certify independent trajectories or learning-rate-policy generalization.
