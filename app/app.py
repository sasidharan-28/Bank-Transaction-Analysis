import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DummyModel:
    def predict(self, X):
        return (X['TransactionAmount (INR)'] > 2000).astype(int).values

st.set_page_config(page_title="Bank Transaction Analysis", layout="wide")

# Paths
DATA_PATH = "../data/Bank_Transactions_Cleaned.csv"
MODEL_PATH = "../models/final_model.pkl"

st.title("Bank Transaction Analysis and Customer Behavior Insights")
st.markdown("**Note:** The prediction target 'High_Value_Transaction' is derived for demonstration purposes. It is NOT a real fraud prediction.")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        # Sample data to make it load fast for dashboard
        return pd.read_csv(DATA_PATH).sample(10000, random_state=42)
    return None

df = load_data()

if df is None:
    st.error("Cleaned dataset not found. Please run Phase 2 notebook first.")
    st.stop()

tabs = st.tabs(["Dashboard", "Transaction Analysis", "Customer Analysis", "Model Prediction"])

with tabs[0]:
    st.header("Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions (Sample)", len(df))
    col2.metric("Total Amount (INR)", f"₹{df['TransactionAmount (INR)'].sum():,.2f}")
    col3.metric("Avg Amount (INR)", f"₹{df['TransactionAmount (INR)'].mean():,.2f}")
    
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

with tabs[1]:
    st.header("Transaction Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Amount Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['TransactionAmount (INR)'].dropna(), bins=30, ax=ax, kde=True)
        st.pyplot(fig)
    with col2:
        st.subheader("Balance Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['CustAccountBalance'].dropna(), bins=30, ax=ax, kde=True)
        st.pyplot(fig)

with tabs[2]:
    st.header("Customer Analysis")
    top_locations = df['CustLocation'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_locations.index, y=top_locations.values, ax=ax)
    plt.xticks(rotation=45)
    ax.set_title("Top 10 Locations")
    st.pyplot(fig)

with tabs[3]:
    st.header("Model Prediction")
    st.markdown("Predict if a transaction is a **High-Value Transaction** based on the derived threshold.")
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        
        col1, col2 = st.columns(2)
        with col1:
            balance = st.number_input("Customer Account Balance (INR)", value=10000.0)
            txn_time = st.number_input("Transaction Time (HHMMSS format)", value=120000.0)
            amount = st.number_input("Transaction Amount (INR)", value=100.0)
            
        with col2:
            st.markdown("### Model Performance (Test Set)")
            if os.path.exists("../outputs/tables/model_comparison.csv"):
                comp_df = pd.read_csv("../outputs/tables/model_comparison.csv")
                st.dataframe(comp_df)
            else:
                st.info("Run Phase 3 Notebook to generate performance metrics.")
        
        if st.button("Predict"):
            features = pd.DataFrame({'CustAccountBalance': [balance], 'TransactionTime': [txn_time], 'TransactionAmount (INR)': [amount]})
            pred = model.predict(features)[0]
            
            if pred == 1:
                st.error("Prediction: YES - This is a High-Value Transaction.")
            else:
                st.success("Prediction: NO - This is NOT a High-Value Transaction.")
    else:
        st.error("Model file not found. Please run Phase 3 notebook to train and save the model.")
