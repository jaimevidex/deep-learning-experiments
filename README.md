# Deep Learning Experiments

Two study tracks kept together because they follow the same arc: implement the fundamentals by hand, then use a framework to run experiments that would be impractical otherwise — architecture sweeps, a CNN on medical images, and an attention study on sequence data.

## Part 1 — fundamentals and architecture studies

Everything starts in NumPy: a **perceptron**, **logistic regression** and a **multi-layer perceptron** with hand-written forward passes, cross-entropy, backpropagation and updates. Then the same ideas become experiments:

- **Pooling study** — 4×4 average pooling washes out thin strokes; 2×2 max pooling keeps the signal. Input dimensionality drops 784 → 196 and training time from ~80 s to ~35 s with negligible accuracy loss.
- **Depth study** — 1, 3, 5, 7 and 9 hidden layers.
- **Width study** — 16 to 256 hidden units × learning rate × dropout × weight decay.

| Model | Test accuracy |
|---|---|
| Perceptron | 61.1% |
| Logistic regression | 69.6% |
| Logistic regression, tuned | 72.1% |
| MLP | **87.8%** |

Width saturates around 128 units; beyond that the training accuracy keeps climbing while validation stalls — the classic capacity/overfitting trade-off. Dropout only helps at width 256.

## Part 2 — real data, real pipelines

**Blood cell classification.** A CNN trained from scratch on the MedMNIST blood-cell dataset (`medmnist`), with the best checkpoint selected on validation accuracy.

**RNA binding affinity.** Predicting protein–RNA binding intensity from sequence. The hard parts are not the architecture:

- **Masked loss** — failed experiments are `NaN` in the target; they are replaced by `0.0` but excluded from the loss via a validity mask, otherwise the model learns to predict zero.
- **Masked Spearman correlation** — the metric ignores invalid rows.
- **Architecture comparison** — CNN, LSTM and an attention model, each with a hyperparameter study, plus loss and correlation curves per configuration.

`part-2/rna-binding/results/` holds the JSON histories and the plots used in the write-up.

## Layout

```
part-1/
  models/              NumPy models (perceptron, logistic regression, MLP, feedforward)
  experiments/         depth, width and hyperparameter runners
  results/, figures/   metrics and plots
  REPORT.md            written analysis

part-2/
  bloodmnist_cnn.py    CNN on the blood-cell dataset
  rna-binding/
    code/              training, evaluation, tuning, plotting
    results/           JSON metrics + PDFs
```

## Run

```bash
pip install numpy matplotlib torch torchvision medmnist scikit-learn openpyxl
```

```bash
# Part 1
cd part-1
python experiments/logistic_regression_run.py -q a
python experiments/run_width_grid.py

# Part 2
python part-2/bloodmnist_cnn.py
cd part-2/rna-binding/code && python train.py
```

Datasets are not committed; the problem statement and `data-notes.md` document where to get them.

## Credits

Part 1 was built with **Henrique Machado** and **Gil Arroteia**.
