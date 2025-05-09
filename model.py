import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score,accuracy_score
from sklearn.metrics import classification_report
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE


def load_data():
    data_path = r"./Prostate_Cancer.csv"
    df = pd.read_csv(data_path)

    df.drop(columns=['id'], inplace=True)

    # Monte Carlo resampling
    new_size = 30000
    new_data = []
    for _ in range(new_size):
        sampled_row = df.sample(n=1, replace=True)
        new_data.append(sampled_row.values[0])
    df = pd.DataFrame(new_data, columns=df.columns)

    # Split X and y
    X = df.drop(columns=['diagnosis_result'])
    y = df['diagnosis_result']

    # SMOTE oversampling
    smt = SMOTE()
    X_res, y_res = smt.fit_resample(X, y)
    print("Resampled shape:", X_res.shape)

    return X_res, y_res

def train_random_forest(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = rf_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return rf_classifier, scaler, accuracy, report
