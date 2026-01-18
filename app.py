import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# App Title
# ----------------------------
st.title("Breast Cancer Classification App")
st.write("Machine Learning Models Deployment using Streamlit")

# ----------------------------
# Load Models
# ----------------------------
MODEL_DIR = "model"

models = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic.pkl")),
    "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl")),
    "KNN": joblib.load(os.path.join(MODEL_DIR, "knn.pkl")),
    "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl")),
    "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl")),
    "XGBoost": joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
}

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# ----------------------------
# Model Selection
# ----------------------------
selected_model_name = st.selectbox("Select a Model", list(models.keys()))
model = models[selected_model_name]

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)

    # Assign column names
    df.columns = ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, df.shape[1]-1)]
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop(["id", "diagnosis"], axis=1)
    y = df["diagnosis"]

    # Scale
    X_scaled = scaler.transform(X)

    # Predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Metrics
    st.subheader("Evaluation Metrics")

    st.write("Accuracy:", accuracy_score(y, y_pred))
    st.write("AUC:", roc_auc_score(y, y_prob))
    st.write("Precision:", precision_score(y, y_pred))
    st.write("Recall:", recall_score(y, y_pred))
    st.write("F1 Score:", f1_score(y, y_pred))
    st.write("MCC:", matthews_corrcoef(y, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
