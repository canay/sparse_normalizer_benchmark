# E5-B Result and Batch-256 Repair

## Configuration audit

A pre-submission manifest audit found that MNIST, Fashion-MNIST, KMNIST and the
first execution of all six E5-B tasks had used batch 512 instead of the frozen
batch 256. The affected tasks were rerun without changing the method set,
seeds, task set, predictors, thresholds, label rule, or test-read boundary.

The corrected full-budget evidence contains 13 tasks x 7 methods x 10 seeds =
910 completed cells, all at batch 256. The raw `grid_batch256_repair` record
contains the 210 replacement cells for the three core tasks. The six final
E5-B records contain 70 cells each. Every corresponding manifest reports zero
failures. Execution-host names in the public manifest copies are redacted; no
scientific field is changed.

## Frozen analysis outcome

Only USPS is labelled *sparse-helps*. Its best supported full-budget gain is
top-k 0.25 at +0.0126 validation accuracy. The other twelve tasks carry no
Holm-supported sparse advantage. The positive-class size is therefore one,
below the pre-registered floor of three.

`code/e5_rule.py` exits with code 2 at the power condition. The proposed
selection rule is not fitted. No task, predictor, threshold, or success
criterion is changed after observing this result.

## Corrected core-grid consequence

The uniform-batch repair removes the previously supported positive labels for
Fashion-MNIST and KMNIST. The corrected dense-softmax validation accuracies for
MNIST, Fashion-MNIST, and KMNIST are 0.9807, 0.8861, and 0.9647. In KMNIST,
head-wise adaptive entmax changes from a previously reported +0.0057 with Holm
support to -0.0026 with Holm support.

These corrections affect the full-budget grid and E5 outcome. The schedule
crossover, step-matched pair, bottom-k control, and single frozen test read use
their own recorded configurations and are unchanged.
