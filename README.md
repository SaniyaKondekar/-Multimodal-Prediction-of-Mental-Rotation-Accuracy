# **Multimodal Prediction of Mental Rotation Accuracy** 

<p align="center">
  <strong style="font-size: 24px;">IITB EdTech 2025, DYPCET</strong><br>
  <strong style="font-size: 20px;">Track 1 - Educational Data Analysis (EDA)</strong>
</p>

---

**Group Name :** Visionaries  
**Group ID :** T1_G13  

---

## 🚀 Project Overview

Understanding and predicting **human cognitive performance** using physiological and behavioral signals has wide applications in **educational technology, adaptive learning, and neuroscience**.  

This project focuses on predicting **mental rotation task accuracy** — a key measure of spatial reasoning — using **multimodal human signals**. By analyzing EEG, eye-tracking, skin responses, and facial expressions, we aim to develop models that can predict **whether a participant will respond correctly** before or during task performance.

**Key Research Context:**

- Mental rotation tasks evaluate the ability to mentally manipulate 2D/3D objects.  
- EEG, eye movements, and physiological responses correlate strongly with **cognitive load and task performance**.  
- Multimodal models leverage **complementary information** from different sensors to improve prediction accuracy.

---

## 🎯 Problem Statement

**Objective:**  
Predict the correctness of a participant’s response (**binary classification**) to mental rotation questions using multimodal data:  

- **EEG (Brain Signals):** Delta, Theta, Alpha, Beta, Gamma bands  
- **Eye-tracking:** Fixations, saccades, pupil dilation, gaze dispersion  
- **GSR (Galvanic Skin Response):** Skin conductance and resistance  
- **Facial expressions (TIVA):** Emotion scores, Action Units  

**Models Explored:**  

- Baseline ML: Random Forest, XGBoost, Logistic Regression  
- Modality-wise models + Fusion via stacking or ensemble  

**Goal:** Build an **interpretable**, high-performance model to predict task correctness and analyze modality contributions.

---

## 📊 Dataset Description

The dataset is **multimodal** and organized as follows:

| File        | Description |
|------------|-------------|
| `PSY.csv`  | Target variable (`Result` = Correct/Incorrect) and timestamps |
| `EEG.csv`  | EEG band features (Delta, Theta, Alpha, Beta, Gamma) |
| `EYE.csv` & `IVT.csv` | Eye-tracking features (fixations, saccades, gaze) |
| `GSR.csv`  | Skin conductance and resistance |
| `TIVA.csv` | Facial expressions and emotions (AU, affective states) |

**Data Preparation Highlights:**

- Synchronized each modality using question timestamps  
- Aggregated statistics: mean, max, min, std per question  
- Handled missing data and class imbalance using **SMOTE**  

---

## 🛠 Methodology

### Step 1: Preprocessing

1. Merge multimodal features per question per participant  
2. Standardize features using **StandardScaler**  
3. Encode target (`Correct=1`, `Incorrect=0`)  
4. Balance dataset with **SMOTE**  

### Step 2: Feature Engineering

- **EEG:** mean & variance of Delta, Theta, Alpha, Beta, Gamma bands  
- **GSR:** mean, peak count, temporal changes  
- **Eye-tracking:** fixation duration, saccade amplitude, pupil size  
- **Facial expressions:** mean emotion scores, AU activation  

### Step 3: Model Development

**Baseline Models:**  

- Random Forest  
- XGBoost  
- Logistic Regression  

**Hyperparameter Tuning:**  

- Performed with **Optuna** to maximize F1-score  

**Intermediate Fusion:**  

- Train separate models per modality  
- Fuse logits/embeddings via stacking/ensemble for final prediction  

---

### Step 4: Evaluation & Interpretability

**Metrics:** Accuracy, F1-score, ROC-AUC, Confusion Matrix  

**Interpretability Tools:**  

- **SHAP:** Global feature importance visualization  
- **LIME:** Local explanation for individual predictions  
- Analysis of top contributing features across EEG, Eye, GSR, and Facial modalities  

---

### Step 5: Model Selection

- Best-performing model selected based on **F1-score**  
- Model and scaler saved for deployment:  

```text
xg_model.pkl
scaler.pkl
```

---

## 🧩 Modeling Approaches Implemented

### 1️⃣ Baseline Machine Learning Models

We trained several baseline models using concatenated features from all modalities:

| Model               | Description |
|--------------------|-------------|
| Random Forest       | Ensemble tree-based model to handle non-linear interactions |
| XGBoost             | Gradient boosting for higher performance and regularization |
| Logistic Regression | Simple linear baseline for comparison |

**Hyperparameter Tuning:**  
- Used **Optuna** to automatically search for optimal hyperparameters  
- Metrics optimized: **F1-score** (primary), Accuracy, ROC-AUC  

**Outcome:**  
- Random Forest and XGBoost showed **stronger performance** compared to Logistic Regression.  
- Feature importance was extracted via **SHAP** and **LIME** to understand key contributors from EEG, Eye-tracking, GSR, and Facial features.

---

### 2️⃣ Multimodal Fusion Model

To leverage complementary information across modalities:

**Steps:**

1. Train separate models for each modality:
   - **EEG → logits**
   - **Eye → logits**
   - **GSR → logits**
   - **Facial → logits**
2. Fuse logits or hidden embeddings from each modality.
3. Train a **meta-classifier** (stacking ensemble or neural network) on fused outputs.  

**Interpretability:**  
- SHAP analysis on the final fusion model shows which modalities dominate predictions.  
- LIME allows **local explanations** for individual participant predictions.  

**Outcome:**  
- The fusion model achieved **higher F1-score and ROC-AUC** than single-modality models.  
- EEG and Eye-tracking contributed most to prediction, followed by GSR and Facial expressions.  

---

## 🧪 Evaluation Metrics

| Metric       | Description |
|-------------|-------------|
| Accuracy    | Overall correct predictions |
| Precision   | True positives / predicted positives |
| Recall      | True positives / actual positives |
| F1-score    | Harmonic mean of precision and recall |
| ROC-AUC     | Area under the ROC curve |
| Confusion Matrix | True vs predicted labels visualization |

> All models were evaluated using **cross-validation** and **hold-out test sets** for robustness.

---

## 📂 Repository Structure
```
project/
├── data/
│ ├── PSY_feature_engineered.csv       # Target labels and timestamps
│ ├── EEG_feature_engineered.csv       # EEG features
│ ├── GSR_feature_engineered.csv       # Skin conductance/resistance
│ ├── EYE_feature_engineered.csv       # Eye-tracking features
│ ├── IVT_feature_engineered.csv       # Additional eye metrics
│ └── TIVA_feature_engineered.csv      # Facial expressions and emotions
├── notebooks/
│ ├── 01_preprocessing.ipynb           # Data cleaning & merging
│ ├── 02_feature_engineering.ipynb     # Feature computation
│ ├── 03_modeling_baseline.ipynb       # Random Forest, XGBoost, Logistic Regression
│ ├── 04_modeling_fusion.ipynb         # Multimodal fusion models
│ └── 05_analysis.ipynb                # Evaluation & interpretability
├── models/
│ ├── scaler.pkl                       # Preprocessing scaler
│ ├── xgb_model.pkl                    # XGBoost trained model(Best Accuarcy Model)
│ └── fusion_model.pt                  # Multimodal fusion model
└── README.md                          # Project documentation
```
---
<p align="center">
Made with ❤️ by <strong>Visionaries</strong> | IITB EdTech 2025
</p>
