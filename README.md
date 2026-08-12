# Hybrid Ensemble Brain Tumor Identification System

**Final MS Thesis Project**

An end-to-end brain MRI classification system for identifying **Brain Tumor** and **Healthy** images using a hybrid deep-learning and machine-learning ensemble.

## Overview

The final system combines:
1. a Convolutional Neural Network (CNN),
2. Wavelet image features,
3. Histogram of Oriented Gradients (HOG),
4. CNN-derived deep features,
5. PCA and feature scaling,
6. Random Forest, SVM, KNN, and Logistic Regression,
7. an accuracy-weighted soft-voting ensemble,
8. a CustomTkinter desktop application for interactive prediction.

## Final Performance

| Evaluation | Ensemble Accuracy |
|---|---:|
| Validation set (920 images) | **97.50%** |
| Separate unseen set (600 images) | **97.17%** |

The unseen set was balanced with 300 Healthy and 300 Tumor MRI images.

For model-by-model results, see `results/RESULTS.md` and `results/model_performance.csv`.

## Repository Structure

```text
04-Final-MS-Thesis-Brain-Tumor-Identification-System/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── app.py
├── notebooks/
│   └── thesis_model_development.ipynb
├── models/
│   ├── cnn_for_new_identification.h5
│   ├── rf_new_model.pkl
│   ├── svm_new_model.pkl
│   ├── knn_new_model.pkl
│   ├── logr_new_model.pkl
│   ├── PCA_*.pkl
│   └── scaler_*.pkl
├── results/
│   ├── RESULTS.md
│   └── model_performance.csv
└── data/
    └── README.md
```

## Dataset

The thesis archive supplied for cleanup contained:

- **4,600 main MRI images**
  - 2,513 Brain Tumor
  - 2,087 Healthy
- **600 unseen evaluation images**
  - 300 Tumor
  - 300 Healthy
- **73 Buner Hospital images**

Raw datasets are intentionally excluded from the public GitHub-ready package. See `data/README.md` for the expected structure and publication/privacy notes.

## Installation

```bash
pip install -r requirements.txt
```

## Run the Desktop App

```bash
cd src
python app.py
```

The trained model files are provided in `models/`.

## Research Notes

This public version separates the final saved pipeline from additional exploratory feature-engineering experiments in the original thesis notebook. Reported headline metrics correspond specifically to the saved hybrid model used by the application.

## Academic Context

**Student:** Shahab Khan  
**Degree:** MS Cyber Security  
**Research area:** AI/ML, medical image classification, ensemble learning

## Disclaimer

This project is an academic research implementation. It is **not a certified medical device and must not be used for clinical diagnosis or treatment decisions**.
