# Convolutional Neural Network for Intel Image Classification

- [English](README_EN.md)
- [Русский](README.md)

**An efficient convolutional neural network** achieving **90% accuracy** on the Intel Image Classification dataset in just 15-30 training epochs **without transfer learning** with comprehensive metric monitoring.

- [Results](#results)
- [Metrics](#metrics)
- [TensorBoard](#tensorboard)

  - [Scalars](#scalars)
  - [PR-Curves](#pr-curves)
  - [Confusion Matrix](#confusionmatrix)
  - [Histograms](#histograms)
  - [Distributions](#distributions)

- [Architecture](#architecture)

  - [Optimizer](#optimizer)

- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Results

- **90.07% accuracy**
- **Fast convergence**: 89%+ accuracy in just 12-30 epochs
- **ResNet-like architecture**, implemented **from scratch**
- **Comprehensive monitoring** via TensorBoard
- **Excellent PR-Curves** and histograms
- **Overfitting gap** of only **0.17%**

## Metrics

The project includes a comprehensive set of metrics and visualizations:

### TensorBoard

- **Confusion Matrix** - error matrix with absolute and normalized values
- **PR-Curves** - Precision-Recall curves with good coverage
- **Weight/Bias/Gradient histograms and distributions** for all model layers
- **Key metrics** loss and accuracy for train and test sets

### Scalars

<div align="center">

![Accuracy](result/scalars/accuracy_graph.png)

![Loss](result/scalars/loss_graph.png)

![Learning Rate](result/scalars/learning_rate_graph.png)

</div>

### PR-Curves

<div align="center" style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;"><img src="result/pr_curves/forest_epoch_27.png" width="30%" alt="PR-Curve Forest"> <img src="result/pr_curves/sea_epoch_27.png" width="30%" alt="PR-Curve Sea"> <img src="result/pr_curves/street_epoch_27.png" width="30%" alt="PR-Curve Street"></div>

### Confusion Matrix

<div align="center"><img src="result/confusion_matrix/confusion_matrix_epoch_2.png" width="75%" alt="Confusion Matrix Epoch 2"> <img src="result/confusion_matrix/confusion_matrix_epoch_27.png" width="75%" alt="Confusion Matrix Epoch 27">

_Confusion matrices at epochs 2 and 27_

</div>

### Histograms

<div align="center"><img src="result/histograms/gradients_histograms.png" width="75%" alt="Gradients Histograms"> <img src="result/histograms/biases_histograms.png" width="75%" alt="Biases Histograms">

_Gradients and biases histograms_

</div>

### Distributions

<div align="center"><img src="result/distributions/biases_distributions.png" width="75%" alt="Biases Distributions">
<img src="result/distributions/gradients_distributions.png" width="75%" alt="Gradients Distributions">

_Biases and gradients distributions_

</div>

## Architecture

<details><summary>Full diagram</summary>

![CNN Full Graph](result/diagram/cnn_graph_full.png)

---

</details>

---

![CNN Principal Scheme](result/diagram/cnn_principal_scheme.png)

### Optimizer

Various optimizers were tested, including PNMBelief, DiffPNM, Adam, AdamW, SGD, NSGD. As a result of testing and fine-tuning, it was found that optimizers with positive-negative momentum estimation do not provide accuracy improvements for this model. SGD and Nesterov SGD optimizers also showed no advantages in either accuracy or convergence speed.

The best optimizer turned out to be AdamW, achieving maximum accuracy of 90.07% with the fastest convergence speed. The model with AdamW optimizer reaches about 89% accuracy in just 12-30 epochs, which is significantly faster than other optimizers that lag behind by an average of 10-15 epochs under similar conditions.

Optimal parameters were determined empirically:

```bash
  num_epochs: 30
  batch_size: 40

  learning_rate: 0.000375
  beta0: 0.915
  beta1: 0.985
  weight_decay: 0.0017
```

## Installation

```bash
poetry install
```

## Usage

```bash
python scripts/train.py

tensorboard --logdir=outputs/logs

python scripts/predict.py
```

## Dependencies

- Python >=3.13,<3.15
- NumPy (>=2.3.5,<3.0.0)
- Tqdm (>=4.67.1,<5.0.0)
- Torch (>=2.9.1,<3.0.0)
- Torchvision (>=0.24.1,<0.25.0)
- Scikit-learn (>=1.7.2,<2.0.0)
- Tensorboard (>=2.20.0,<3.0.0)
- Torchviz (>=0.0.3,<0.0.4)
