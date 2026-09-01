# 🌿 Chinese Herbs Classification

A deep learning computer vision project for the automated classification of **Chinese medicinal herbs from images** using Convolutional Neural Networks (CNNs).

The project investigates whether data augmentation and dropout can improve the generalisation and classification performance of a CNN when identifying multiple categories of Chinese medicinal herbs.

---

## Overview

Chinese medicinal herbs can have visually similar characteristics, making image-based identification a challenging computer vision problem.

This project develops and compares two CNN-based image classification models:

1. **Basic CNN** trained without data augmentation
2. **Enhanced CNN** incorporating dropout and data augmentation

The models are evaluated on an unseen test dataset using accuracy, precision, recall, F1-score and confusion matrices.

The enhanced model achieved a **test accuracy of 83.68%**, compared with **82.18%** for the basic CNN.

---

## Problem

Manual identification of medicinal herbs can require specialist knowledge and can become challenging when different herbs share similar visual characteristics.

This project investigates the following question:

> **Can a convolutional neural network accurately classify Chinese medicinal herbs from images, and can data augmentation and dropout improve its generalisation?**

The problem is formulated as an **11-class image classification task**.

---

## Dataset

The dataset contains images belonging to **11 categories of Chinese medicinal herbs**.

### Classes

| Class              |
| ------------------ |
| Cordyceps          |
| Cornus Fruit       |
| Dried Ginger       |
| Dried Mulberry     |
| Ginseng            |
| Hawthorn           |
| Honeysuckle Flower |
| Lily Bulb          |
| Oyster Shell       |
| Schisandra Berry   |
| Smoke Plum         |

The original dataset contains approximately **1,575 images per category**, with some variation between classes.

### Dataset distribution

The dataset was divided into:

| Dataset    |     Images | Proportion |
| ---------- | ---------: | ---------: |
| Training   |     12,150 |        70% |
| Validation |      3,474 |        20% |
| Testing    |      1,742 |        10% |
| **Total**  | **17,366** |   **100%** |

The repository contains a miniature dataset archive for demonstration purposes.

---

## Methodology

The project follows the workflow below:

```text
                  Chinese Herb Images
                         │
                         ▼
                  Dataset Exploration
                         │
                         ▼
                  Data Preprocessing
                         │
                         ├── Resize → 128 × 128
                         └── Normalisation → [0, 1]
                         │
                         ▼
                 Train / Validation / Test
                    70% / 20% / 10%
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        Basic CNN              Enhanced CNN
        No Augmentation        + Augmentation
                              + Dropout
              │                     │
              └──────────┬──────────┘
                         ▼
                    Model Training
                         │
                         ▼
                     Evaluation
                         │
                         ▼
        Accuracy / Precision / Recall / F1
                         │
                         ▼
                  Model Comparison
```

---

## Data Preprocessing

The images were resized to:

```text
128 × 128 × 3
```

Pixel values were normalised to the range:

```text
0 – 1
```

by applying:

```python
rescale=1./255
```

### Label Encoding

TensorFlow's `ImageDataGenerator.flow_from_directory()` was used to automatically assign class labels based on the directory structure.

### Data Splitting

The dataset was randomly divided into:

* **70% training**
* **20% validation**
* **10% testing**

A random seed of **42** was used to improve reproducibility.

---

## Data Augmentation

The enhanced model used data augmentation on the training images.

The following transformations were applied:

* Horizontal flipping
* Zooming up to 10%

```python
ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.1,
    horizontal_flip=True
)
```

The motivation was to expose the model to variations that may occur when herb images are captured at different orientations and scales.

Validation and test images were **not augmented**.

---

# Model Architecture

## 1. Basic CNN

The baseline model consists of three convolutional blocks followed by fully connected layers.

```text
Input
128 × 128 × 3
      │
      ▼
Conv2D
16 filters
3 × 3
ReLU
      │
      ▼
MaxPooling 2 × 2
      │
      ▼
Conv2D
32 filters
3 × 3
ReLU
      │
      ▼
MaxPooling 2 × 2
      │
      ▼
Conv2D
64 filters
3 × 3
ReLU
      │
      ▼
MaxPooling 2 × 2
      │
      ▼
Flatten
      │
      ▼
Dense
128 units
ReLU
      │
      ▼
Dense
11 units
Softmax
```

### Configuration

| Parameter            | Value                    |
| -------------------- | ------------------------ |
| Input size           | 128 × 128 × 3            |
| Convolutional layers | 3                        |
| Filters              | 16 → 32 → 64             |
| Activation           | ReLU                     |
| Pooling              | MaxPooling               |
| Dense layer          | 128 units                |
| Output units         | 11                       |
| Output activation    | Softmax                  |
| Loss                 | Categorical Crossentropy |
| Optimizer            | Adam                     |
| Batch size           | 16                       |
| Maximum epochs       | 10                       |
| Early stopping       | Yes                      |

---

# 2. Enhanced CNN

The enhanced model extends the baseline CNN by introducing:

* An additional convolutional layer
* Dropout regularisation
* Training-time data augmentation

```text
Input
128 × 128 × 3
      │
      ▼
Conv2D
16 filters
3 × 3
ReLU
      │
      ▼
MaxPooling
      │
      ▼
Conv2D
32 filters
3 × 3
ReLU
      │
      ▼
MaxPooling
      │
      ▼
Dropout 0.25
      │
      ▼
Conv2D
64 filters
3 × 3
ReLU
      │
      ▼
MaxPooling
      │
      ▼
Dropout 0.25
      │
      ▼
Conv2D
128 filters
3 × 3
ReLU
      │
      ▼
MaxPooling
      │
      ▼
Dropout 0.50
      │
      ▼
Flatten
      │
      ▼
Dense
512 units
ReLU
      │
      ▼
Dropout 0.50
      │
      ▼
Dense
11 units
Softmax
```

### Configuration

| Parameter            | Value                     |
| -------------------- | ------------------------- |
| Input size           | 128 × 128 × 3             |
| Convolutional layers | 4                         |
| Filters              | 16 → 32 → 64 → 128        |
| Activation           | ReLU                      |
| Pooling              | MaxPooling                |
| Dense layer          | 512 units                 |
| Dropout              | 0.25 / 0.25 / 0.50 / 0.50 |
| Output units         | 11                        |
| Output activation    | Softmax                   |
| Loss                 | Categorical Crossentropy  |
| Optimizer            | Adam                      |
| Batch size           | 16                        |
| Maximum epochs       | 10                        |
| Early stopping       | Yes                       |
| Data augmentation    | Yes                       |

---

## Training

Both models were trained using:

* Adam optimiser
* Categorical cross-entropy loss
* Batch size of 16
* Maximum of 10 epochs
* Early stopping

Early stopping monitored validation loss:

```python
EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

This was used to reduce unnecessary training and retain the model weights from the best validation performance.

---

# Results

The models were evaluated against the unseen test set containing **1,742 images**.

## Model Comparison

| Metric             | Enhanced CNN + Augmentation | Basic CNN |
| ------------------ | --------------------------: | --------: |
| Test Accuracy      |                  **83.68%** |    82.18% |
| Macro Precision    |                    **0.84** |      0.83 |
| Macro Recall       |                    **0.84** |      0.82 |
| Macro F1-score     |                    **0.84** |      0.82 |
| Weighted Precision |                    **0.84** |      0.83 |
| Weighted Recall    |                    **0.84** |      0.82 |
| Weighted F1-score  |                    **0.83** |      0.82 |
| Test Loss          |                  **0.4660** |    0.5952 |

### Key Finding

The enhanced CNN performed better than the basic CNN across the main evaluation metrics.

The test accuracy increased from:

```text
82.18% → 83.68%
```

representing an improvement of approximately **1.50 percentage points**.

The enhanced model also achieved a higher macro-averaged F1-score:

```text
0.82 → 0.84
```

This indicates improved overall performance across the different herb classes.

---

## Per-Class Performance

The enhanced model achieved the following classification performance:

| Class              | Precision | Recall |   F1 |
| ------------------ | --------: | -----: | ---: |
| Cordyceps          |      0.89 |   0.79 | 0.84 |
| Cornus Fruit       |      0.89 |   0.98 | 0.93 |
| Dried Ginger       |      0.89 |   0.71 | 0.79 |
| Dried Mulberry     |      0.76 |   0.76 | 0.76 |
| Ginseng            |      0.72 |   0.73 | 0.73 |
| Hawthorn           |      0.98 |   0.90 | 0.94 |
| Honeysuckle Flower |      0.88 |   0.97 | 0.93 |
| Lily Bulb          |      0.74 |   0.89 | 0.81 |
| Oyster Shell       |      0.83 |   0.90 | 0.87 |
| Schisandra Berry   |      0.95 |   0.77 | 0.85 |
| Smoke Plum         |      0.74 |   0.80 | 0.77 |

The strongest F1-scores were observed for **Hawthorn (0.94)**, **Cornus Fruit (0.93)** and **Honeysuckle Flower (0.93)**.

The comparatively lower F1-scores for classes such as **Ginseng (0.73)** and **Dried Mulberry (0.76)** suggest that these categories present greater classification difficulty.

---

## Confusion Matrix

Confusion matrices were generated for both models to investigate class-level prediction errors.

The enhanced model generally produced stronger classification results across the classes, although some confusion remained between visually similar categories.

For example, the enhanced model showed notable confusion involving:

* Dried Ginger
* Lily Bulb
* Dried Mulberry
* Smoke Plum
* Ginseng

This suggests that further improvements in feature extraction, dataset diversity and augmentation could be beneficial.

---

## Technologies

### Programming

* Python
* Jupyter Notebook

### Deep Learning

* TensorFlow
* Keras
* Convolutional Neural Networks

### Data Processing

* NumPy
* Pandas

### Machine Learning Evaluation

* Scikit-learn

### Visualisation

* Matplotlib

### Development Environment

* Google Colab
* Google Drive
* Git
* GitHub

---

## Project Structure

```text
Chinese_Herbs_Classification/
│
├── CW1_Nkansah_Richard_S2424536.ipynb
│   └── Complete data preparation, modelling and evaluation workflow
│
├── ChineseHerbs_mini.zip
│   └── Miniature dataset
│
└── README.md
    └── Project documentation
```

> The full dataset used during development is not included in the repository.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Worshipper600/Chinese_Herbs_Classification.git
cd Chinese_Herbs_Classification
```

Install the main dependencies:

```bash
pip install tensorflow matplotlib numpy scikit-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
CW1_Nkansah_Richard_S2424536.ipynb
```

> The original notebook was developed using Google Colab and Google Drive. The dataset paths in the notebook therefore reference the Colab/Google Drive environment and may need to be modified when running locally.

---

## Usage

The complete workflow is contained in the Jupyter Notebook.

The notebook performs:

1. Dataset loading
2. Dataset exploration
3. Class distribution analysis
4. Image visualisation
5. Dataset splitting
6. Image preprocessing
7. Data augmentation
8. Basic CNN construction
9. Enhanced CNN construction
10. Model training
11. Model evaluation
12. Classification report generation
13. Confusion matrix generation
14. Model comparison

---

## Reproducibility

A random seed of **42** was used for:

```python
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

This was intended to improve reproducibility of the experimental workflow.

---

## Limitations

The project was developed within a limited timeframe, which restricted the amount of model training and experimentation that could be performed.

The main limitations identified were:

* Limited training duration
* Maximum of 10 training epochs
* Relatively simple CNN architectures
* Limited augmentation techniques
* No transfer-learning experiments
* The full dataset is not included in the repository

---

## Future Work

### Transfer Learning

Future experiments could evaluate pretrained architectures such as:

* ResNet
* EfficientNet
* MobileNet
* Vision Transformers

Transfer learning could provide stronger feature representations and potentially improve classification performance.

### Improved Data Augmentation

Additional augmentation techniques could be investigated, including:

* Rotation
* Translation
* Brightness adjustment
* Contrast adjustment
* Random cropping

### Hyperparameter Optimisation

Future experiments could systematically investigate:

* Learning rate
* Batch size
* Number of convolutional filters
* Dropout rates
* Optimiser configuration
* Number of epochs

### Explainable AI

Techniques such as **Grad-CAM** could be incorporated to visualise the image regions influencing model predictions.

This would provide greater insight into whether the model is learning meaningful visual characteristics of the herbs.

### Deployment

The trained classifier could be deployed as an interactive application where users upload an image and receive a predicted herb class.

Potential technologies include:

* Streamlit
* FastAPI
* Docker

### Mobile Application

A future version could integrate the model into a mobile application for image-based herb classification.

---

## Citation

If you use this project in your work, please reference the repository:

```bibtex
@software{nkansah_chinese_herbs_classification,
  author = {Richard Baah Nkansah},
  title = {Chinese Herbs Classification},
  year = {2026},
  url = {https://github.com/Worshipper600/Chinese_Herbs_Classification}
}
```

---

## Author

**Richard Baah Nkansah**

GitHub: https://github.com/Worshipper600

---

## Disclaimer

This project is an academic machine learning project focused on image classification.

The model's predictions should not be interpreted as medical advice and should not be used as a substitute for professional identification or expert advice concerning medicinal herbs.
