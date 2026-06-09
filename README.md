# AI-Based Violence Detection in Video

[![GitHub License](https://img.shields.io/github/license/Nihal1l/violence_detection_project)](https://github.com/Nihal1l/violence_detection_project/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

An AI-driven computer vision and deep learning project aimed at detecting violent activities within video sequences. This project leverages a hybrid CNN-RNN architecture optimized through systematic, iterative experimentation to push accuracy boundaries past existing baselines.

## 🎯 Research Goal
The primary objective of this project is to achieve **>96.00% accuracy** on video violence detection, successfully surpassing prior published work in this domain which typically plateaus at $\le$ 93.00% accuracy.

---

## 🏗️ Model Architecture

The core architecture uses a time-distributed feature extractor coupled with a sequential model to capture both spatial and temporal dynamics across video frames.

```text
[ Input Video Frames ] (20 frames, uniform sampling)
         │
         ▼
 ┌───────────────┐
 │   ResNet18    │  (Spatial Feature Extraction)
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │    BiLSTM     │  (Temporal Context Learning: 2 Layers, 256 Hidden Units)
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ Attention/FC  │  (Dynamic Timestep Weighting & Classification)
 └───────┬───────┘
         │
         ▼
[ Violence / Non-Violence Prediction ]
```


### Dataset Specifications
* **Split:** 901 Train samples | 600 Validation samples | 400 Test samples
* **Classes:** Binary Classification (`Violence` / `Non-violence`)
* **Temporal Resolution:** 20 frames per video via uniform sampling

---

## 📈 Experimentation Log & Results

We follow a rigorous iterative approach, tracking how data preprocessing variations, training parameters, and architectural adjustments impact model metrics.

### Summary Metrics Table

| Experiment | Key Change | Test Accuracy | Recall | F1 Score | Performance vs. Baseline | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | ResNet18 + BiLSTM (20 epochs) | 93.75% | 94.00% | 0.9377 | *Benchmark* | 🟢 Done |
| **Exp 2** | Sharpness(2.0) + Contrast(1.5) | 93.25% | 94.00% | 0.9330 | `-0.50%` | 🟢 Done |
| **Exp 3** | Sharpness(1.5) Only | 94.00% | 94.50% | 0.9403 | `+0.25%` | 🟢 Done |
| **Exp 4** | 30 Epochs (Extended Training) | 95.25% | 97.50% | 0.9535 | `+1.50%` | 🟢 Done |
| **Exp 5** | 40 Epochs + `ReduceLROnPlateau` | 95.25% | 98.00% | 0.9538 | `+1.50%` | 🟢 Done |
| **Exp 6** | 🧠 Added Temporal Attention Mechanism | *Pending* | *Pending* | *Pending* | *TBD* | ⏳ Running |

---

## 🔬 Detailed Experiment Breakdown

### Baseline — ResNet18 + BiLSTM
* **Details:** Standard ResNet18 feature extractor + 2-layer BiLSTM classifier. Preprocessing included GaussianBlur, ColorJitter, and RandomHorizontalFlip. Trained for 20 epochs using Adam ($lr=0.0001$) and a batch size of 4.
* **Error Analysis:** Crowd scenes (24%), motion blur (24%), fast motion (20%), low illumination (12%), occlusion (12%), camera shake (8%).

### Exp 2 — Sharpness(2.0) + Contrast(1.5)
* **Hypothesis:** Motion blur caused 24% of baseline errors; aggressive sharpening was expected to extract cleaner edges.
* **Outcome:** Performance dropped. Stacking `Contrast(1.5)` on top of `ColorJitter(contrast=0.3)` modified contrast twice, while `Sharpness(2.0)` over-sharpened clean frames, creating artificial artifacts.

### Exp 3 — Sharpness(1.5) only
* **Hypothesis:** Retain gentle sharpening to combat blur, but drop the extra contrast enhancement and GaussianBlur to avoid over-processing.
* **Outcome:** Performance improved past baseline. Val accuracy peaked at 95.17% at Epoch 20 with signs showing that the model had not yet fully converged.

### Exp 4 — 30 Epochs
* **Hypothesis:** Extend training duration to allow the rising accuracy trajectory from Exp 3 to fully mature.
* **Outcome:** Significant jump of `+1.25%` in overall accuracy, bringing the model to 95.25% accuracy and 97.50% recall.

### Exp 5 — 40 Epochs + ReduceLROnPlateau
* **Hypothesis:** Train for 40 epochs while introducing a scheduler (`patience=5`, `factor=0.5`) to decay the learning rate when validation accuracy plateaus, allowing finer convergence.
* **Outcome:** Reached **98.00% Recall**. The model rarely misses any actual violence, but a slight precision gap remains to hit the 96% overall accuracy target.

### Exp 6 — Attention Mechanism on BiLSTM (In Progress)
* **Hypothesis:** Crowd scenes (24% of errors) confuse the model because the violent action often happens in only a few brief moments. Instead of pooling or evaluating just the final timestep (`lstm_out[:,-1,:]`), a learned linear attention layer weights all 20 timesteps dynamically to construct a context vector focusing on the exact frames where violence occurs.

---

## 💡 Key Research Findings

1. **Preprocessing Sensitivity:** Data augmentation requires precise calibration. Aggressive enhancement (`Sharpness 2.0`) introduces artificial edge noise that confuses the CNN feature extractor, whereas a conservative enhancement (`Sharpness 1.5`) aids performance.
2. **Augmentation Stacking Risks:** Layering explicit contrast enhancements on top of integrated color jitters causes cascading degradation. Augmentations affecting the same visual property should be unified, not stacked.
3. **Training Duration Significance:** Simply extending the pipeline from 20 to 30 epochs yielded a `+1.25%` accuracy increase, highlighting that sequential video models require deeper epoch spaces to prevent underfitting before architectural shifting.
4. **The Precision-Recall Gap:** While our current optimal configuration successfully catches almost all instances of real violence (98.00% Recall), it introduces minor false positives. Resolving this requires structural shifts (like the Attention Layer in Exp 6) rather than parameter tuning.

---

## 🛠️ Getting Started

### Prerequisites
* Python 3.8+
* PyTorch (with CUDA support recommended)
* Torchvision
* Pillow

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/Nihal1l/violence_detection_project.git](https://github.com/Nihal1l/violence_detection_project.git)
   cd violence_detection_project
Install required dependencies:

Bash
pip install -r requirements.txt
Project Structure
train.py - Main script for setting hyperparameters and initiating model execution.

dataset_loader.py - Custom dataset class processing the 20-frame uniform sampling and data augmentations.

bilstm_model.py - Architecture configuration containing the ResNet backbone, BiLSTM layers, and attention mechanisms.

📬 Contact & Contributions

This project is an ongoing research endeavor. If you want to contribute, feel free to open an issue or submit a Pull Request. For inquiries or collaborations, please open an issue in this repository.
