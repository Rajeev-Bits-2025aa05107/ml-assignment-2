
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("ML Assignment 2 – Classification Models")

# Load trained models
with open("model/saved_models.pkl", "rb") as f:
    models, results, scaler = pickle.load(f)

# Upload CSV
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

model_name = st.selectbox("Select Model", list(models.keys()))

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_scaled = scaler.transform(X)
    model = models[model_name]

    preds = model.predict(X_scaled)

    st.subheader("Evaluation Metrics")
    st.write(results[model_name])

    st.subheader("Classification Report")
    st.text(classification_report(y, preds))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)
