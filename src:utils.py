# src/utils.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def load_data(filepath_or_url):
    """
    Load the dataset and print basic information (shape and class balance).
    """
    print(f"[Data] Loading dataset from {filepath_or_url}...")
    df = pd.read_csv(filepath_or_url)
    print(f"[Data] Shape: {df.shape}")
    print("[Data] Class Balance:\n", df['HeartDisease'].value_counts(normalize=True))
    return df

def build_preprocessor():
    """
    Construct the preprocessing pipeline.
    
    Strategies:
    1. Ordinal Encoding for ordinal features (preserving order).
    2. One-Hot Encoding for nominal features.
    3. Standardization (Z-score) for numerical features.
    """
    # 1. Ordinal Features - Need to preserve the inherent order
    # 'GenHealth': Poor -> Excellent
    # 'AgeCategory': 18-24 -> 80 or older
    # Note: In a production environment, strictly define the 'categories' list 
    # to ensure the correct order. Using default here for simplicity.
    ordinal_cols = ['AgeCategory', 'GenHealth'] 
    
    # 2. Nominal Features - Unordered categorical data
    categorical_cols = ['Race', 'Diabetic', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Sex', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Smoking']
    
    # 3. Numerical Features - Need standardization
    numeric_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']

    # Build the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), categorical_cols),
            ('ord', OrdinalEncoder(), ordinal_cols) 
        ],
        remainder='passthrough' # Keep unspecified columns if any
    )
    return preprocessor

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance, focusing on Recall (Sensitivity) to minimize False Negatives.
    """
    y_pred = model.predict(X_test)
    
    # Key Metrics
    recall = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n[{model_name}] Performance Metrics:")
    print(f"  > Recall (Sensitivity): {recall:.2%}  <-- KEY METRIC (Minimize False Negatives)")
    print(f"  > Accuracy:             {acc:.2%}")
    print(f"  > F1-Score:             {f1:.2%}")
    
    # Visualization: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.show()
    
    return recall