# Supplementary code and saved results

This supplement accompanies the trajectory, prediction, and scheduling extension of the compact sparse-attention benchmark. It contains 600 discovery trajectories, 2,400 schedule-discovery runs, 720 confirmation-validation cells, and 720 frozen test evaluations. The saved records preserve failed infrastructure attempts separately from the complete registered populations. The original benchmark is available at https://github.com/canay/sparse_normalizer_benchmark.

## Verify the reported results without training

Use Python 3.12 with NumPy and SciPy. From this directory, run:

```text
python -m pip install -r requirements-verification.txt
python verify_saved_outputs.py
```

The verifier checks the full completed-cell census, raw-file hashes, descriptive crossover map, arithmetic of saved predictor errors, and confirmation differences. It writes to a new `verification_output` directory and refuses to overwrite a previous verification. It does not train a model, refit the predictor, or read a test dataset through a model. The pooled task-level tests are reproduced for traceability; shared seed-specific splits across model configurations limit their inferential interpretation.

## Contents and reproduction boundary

The `experiments` tree preserves the relative import layout of the extension and its original classifier/data-loader dependencies. Training programs and frozen analysis programs are provided for inspection and independent replication. Historical host-specific launcher scripts, model checkpoints, dataset archives, and unrelated auxiliary assays are excluded. The scientific saved-output verifier above was tested from a clean extraction; a fresh training campaign was not performed as part of packaging. Training needs PyTorch and the original dataset acquisition dependencies in addition to the verification requirements. The recorded source environment used Python 3.12.3 and PyTorch 2.13.0+cu130 on an NVIDIA GeForce RTX 5060.

The `provenance` directory supplies original result verification and registration/source bindings. `INPUT_MANIFEST.json` identifies byte-exact copied inputs. `CHECKSUMS.sha256` covers the delivered files. The historical registration manifest contains states from its creation time; current completion and arithmetic come from the saved-result verifier. Dataset archives remain with their original providers and are not redistributed.

The predictor and schedule failed their registered criteria. The confirmation target times are identical to fixed top-k and occur before switching. Final-accuracy noninferiority allows a small loss and does not establish equal accuracy, a switching benefit, a general selection rule, or runtime speedup.
