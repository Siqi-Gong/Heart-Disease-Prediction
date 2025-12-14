# Heart Disease Prediction: Handling High Class Imbalance

## Project Overview
This project builds a robust machine learning pipeline to predict heart disease risk using the CDC Personal Key Indicators dataset (320k+ records). 

**Key Challenge:** The dataset suffers from severe **Class Imbalance**, where negative cases significantly outnumber positive cases. A standard model would yield high accuracy but fail to identify actual patients (high False Negatives).

**Objective:** To maximize **Recall (Sensitivity)**, ensuring the model effectively identifies potential heart disease patients for early screening, minimizing critical misses.

## Methodology

### 1. Feature Engineering
- **Categorical Encoding:** Tailored strategies were applied:
  - **Binary features** (e.g., Smoking): Encoded using `OneHotEncoder(drop='if_binary')`.
  - **Ordinal features** (e.g., Age, Health Status): Preserved inherent order using `OrdinalEncoder`.
  - **Nominal features** (e.g., Race): Standard `OneHotEncoder`.
- **Scaling:** `StandardScaler` applied to numerical features like BMI and SleepTime.

### 2. Handling Class Imbalance
We rejected data resampling (SMOTE/Under-sampling) in favor of **Cost-Sensitive Learning** to preserve all valid information:
- **Logistic Regression & Random Forest:** Utilized `class_weight='balanced'` to penalize misclassifications of the minority class.
- **XGBoost:** Applied `scale_pos_weight` calculated dynamically as `count(negative) / count(positive)` to adjust the loss function gradient.

### 3. Model Optimization
- **Stratified Cross-Validation:** Used 5-fold StratifiedKFold to ensure stable class distribution across all training folds.
- **Hyperparameter Tuning:** Performed GridSearchCV targeting `recall` as the scoring metric, optimizing `max_depth` and `learning_rate` to balance model complexity and generalization.

## Key Results

| Model | Recall (Sensitivity) | Interpretation |
| :--- | :--- | :--- |
| **Logistic Regression** | ~75% | Good baseline, high interpretability. |
| **Random Forest** | ~79% | Strong non-linear capture, tuned depth prevents overfitting. |
| **XGBoost (Best)** | **~80%** | **Best Performance.** Effectively balanced trade-off between Precision and Recall. |

**Conclusion:** The tuned XGBoost model successfully identifies ~80% of heart disease patients, significantly outperforming accuracy-driven baselines which typically achieve <10% recall on this dataset.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run pipeline: `python src/main.py`
