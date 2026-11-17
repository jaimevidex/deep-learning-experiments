# Deep Learning — Coursework

Two assignments from the *Deep Learning* course at Instituto Superior Técnico, kept together because they trace the same arc: implement the fundamentals by hand, then use the frameworks to run experiments that would be impractical otherwise.

## Homework 1 — from scratch to grid search

Everything starts in NumPy: a **perceptron**, **logistic regression** and a **multi-layer perceptron**, with hand-written forward passes, cross-entropy, backpropagation and updates. Then the same ideas become experiments:

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

## Homework 2 — real data, real pipelines

**Q1 — BloodMNIST.** A CNN classifier on the MedMNIST blood-cell dataset (`medmnist`), trained from scratch with the best checkpoint selected on validation accuracy.

**Q2 — RNAcompete.** Predicting protein–RNA binding intensity from sequence. The hard parts are not the architecture:

- **Masked loss** — failed experiments are `NaN` in the target; they are replaced by `0.0` but excluded from the loss via a validity mask, otherwise the model learns to predict zero.
- **Masked Spearman correlation** — the evaluation metric ignores invalid rows.
- **Architecture comparison** — CNN, LSTM and an attention model, each with a hyperparameter study, plus loss/correlation curves per configuration.

The `results/` folder holds the JSON histories and the plots used in the report.

## Layout

```
Homework1/
  skeleton_code/       NumPy models
  experiments/         depth/width/hyperparameter runners
  results/, figures/   metrics and plots
  REPORT.md            written analysis

Homework2/
  hw2_q1.py            BloodMNIST CNN
  hw2_q2/              RNAcompete pipeline, models and study
    code/              training, evaluation, tuning, plotting
    results/           JSON metrics + PDFs
```

## Run

```bash
pip install numpy matplotlib torch torchvision medmnist scikit-learn openpyxl
```

```bash
# Homework 1
cd Homework1
python experiments/logistic_regression_run.py -q a
python experiments/run_width_grid.py

# Homework 2
python Homework2/hw2_q1.py
cd Homework2/hw2_q2/code && python hw2_q2_train.py
```

Datasets are not committed; each homework's README/PDF documents where to get them.

## Credits

Homework 1 was a group effort by **Henrique Machado**, **Jaime Silva** and **Gil Arroteia** (Group 14). Homework 2 follows the same course structure.
