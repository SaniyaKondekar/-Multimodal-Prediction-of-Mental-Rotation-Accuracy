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

## 2️⃣ Multimodal Fusion Model

This part of the project predicts outcomes in a Mental Rotation Task by combining brain signals (EEG), eye tracking, skin response (GSR), and facial expression data from the STD dataset. It uses a two-step process to make accurate predictions.

### Approach
1. **Process Each Data Type Separately**:
   - **EEG (10 features)**: Captures brain activity patterns.
   - **Eye Tracking (7 features)**: Tracks eye movement behavior.
   - **GSR (4 features)**: Measures changes in skin response.
   - **Facial Expressions (26 features)**: Analyzes facial feature changes.
   - Each data type is processed using machine learning models (XGBoost, LightGBM, CatBoost for tree-based approaches or a custom neural network for advanced strategies) to create compact outputs called embeddings.

2. **Optimize Models with Optuna**:
   - Optuna tunes model settings (e.g., complexity, learning speed) to boost performance for EEG, eye, GSR, and facial data, improving prediction accuracy.

3. **Combine the Outputs**:
   - Embeddings from all four data types are merged (e.g., 24 features for tree-based models: 4 data types × 3 models × 2 probabilities) and reduced (e.g., 116 → 40 features) for efficiency.
   - Three strategies are available for testing:
     - **VAE Best**: Creates synthetic data with a neural network (Variational Autoencoder) and uses tree-based models. This tied for the best performance.
     - **SMOTE Best**: Balances uneven data with synthetic samples, using tree-based models.
     - **Overall Best**: Mixes tree-based or neural models per data type, based on training results (customize in the script). Tied with VAE Best.

4. **Final Prediction**:
   - A group of eight models (Logistic Regression L1/L2, Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, Extra Trees) combines the merged outputs, averaging their results for better accuracy.

### Interpretability
- **SHAP Analysis**: Shows EEG and eye tracking have the biggest impact on predictions, followed by GSR and facial expressions.
- **LIME Explanations**: Highlights key features for individual participant predictions.

### Outcome
- The **VAE Best** and **Overall Best** strategies performed best, achieving:
  - **AUC**: 0.6573 (moderate reliability)
  - **Accuracy**: 0.7631 (76.31% correct predictions)
  - **F1-Score**: 0.8623 (strong for imbalanced data)
- The high F1-score shows good handling of imbalanced data, but the AUC (0.6573) is below the target (~85%). Cross-validation showed higher potential (e.g., Logistic_L2 AUC: 0.9137), suggesting room for improvement with better model selection or tuning.
- Optuna’s tuning improved model performance, with Logistic_L2 excelling for `vae_best`/`overall_best` and XGBoost for `smote_best`.

## Evaluation Metrics
Models were tested using cross-validation and a separate test set, with Optuna tuning model settings. Below are the key metrics and strategy leaderboard:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correct predictions (e.g., 0.7631 for VAE Best). |
| **Precision** | How many predicted positives were correct. |
| **Recall** | How many actual positives were correctly predicted. |
| **F1-Score** | Balances precision and recall (e.g., 0.8623 for VAE Best). |
| **ROC-AUC** | Measures reliability (e.g., 0.6573 for VAE Best, up to 0.9137 in cross-validation). |
| **Confusion Matrix** | Visualizes true vs. predicted outcomes to spot error patterns. |

### Strategy Leaderboard
| Rank | Strategy       | AUC    | Accuracy | F1-Score | Models |
|------|----------------|--------|----------|----------|--------|
| 1    | Overall Best   | 0.6573 | 0.7631   | 0.8623   | 8      |
| 2    | VAE Best       | 0.6573 | 0.7631   | 0.8623   | 8      |
| 3    | SMOTE Best     | 0.6235 | 0.6899   | 0.7954   | 8      |




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
