import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_csv('data/bank_transactions.csv')

# --- PHASE 1 ---
# (Some basic plotting on sample)
sample_df = df.sample(10000, random_state=42).copy()
plt.figure(figsize=(8,5))
sns.histplot(sample_df['TransactionAmount (INR)'].dropna(), bins=50, kde=True)
plt.title('Distribution of Transaction Amount (Sampled)')
plt.savefig('outputs/figures/amount_histogram.png')
plt.close()

plt.figure(figsize=(8,5))
sns.boxplot(x=sample_df['TransactionAmount (INR)'].dropna())
plt.title('Box Plot of Transaction Amount (Sampled)')
plt.savefig('outputs/figures/amount_boxplot.png')
plt.close()

plt.figure(figsize=(8,5))
top_locations = df['CustLocation'].value_counts().head(10)
sns.barplot(x=top_locations.index, y=top_locations.values)
plt.title('Top 10 Customer Locations')
plt.xticks(rotation=45)
plt.savefig('outputs/figures/location_barchart.png')
plt.close()

plt.figure(figsize=(8,5))
customer_freq = df['CustomerID'].value_counts().head(10)
sns.barplot(x=customer_freq.index, y=customer_freq.values)
plt.title('Top 10 Customers by Transaction Frequency')
plt.xticks(rotation=45)
plt.savefig('outputs/figures/customer_freq.png')
plt.close()

# --- PHASE 2 ---
# Data Cleaning
df['CustGender'].fillna('Unknown', inplace=True)
df['CustLocation'].fillna('Unknown', inplace=True)
df.dropna(subset=['CustAccountBalance', 'TransactionAmount (INR)'], inplace=True)
df.drop_duplicates(inplace=True)

# Standardize
df['CustGender'] = df['CustGender'].str.upper().str.strip()

# Save Cleaned Data
df.to_csv('data/Bank_Transactions_Cleaned.csv', index=False)

sample_clean_df = df.sample(10000, random_state=42)

plt.figure(figsize=(8,5))
sns.histplot(sample_clean_df['CustAccountBalance'], bins=50, kde=True)
plt.title('Balance Histogram')
plt.savefig('outputs/figures/balance_histogram.png')
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(data=sample_clean_df, x='CustAccountBalance', y='TransactionAmount (INR)', alpha=0.3)
plt.title('Amount vs Balance')
plt.savefig('outputs/figures/amount_vs_balance.png')
plt.close()

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr = df[numeric_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('outputs/figures/correlation_heatmap.png')
plt.close()

# --- PHASE 3 ---
# Target creation
threshold = 2000
features = ['CustAccountBalance', 'TransactionTime', 'TransactionAmount (INR)']
X = df[features].copy()
y = (X['TransactionAmount (INR)'] > threshold).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

models = {
    'Logistic Regression': Pipeline([('prep', preprocessor), ('clf', LogisticRegression())]),
    'Decision Tree': Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(max_depth=5, random_state=42))]),
    'Random Forest': Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1))])
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    results.append(metrics)

results_df = pd.DataFrame(results)
results_df.to_csv('outputs/tables/model_comparison.csv', index=False)

# Save final model
joblib.dump(models['Random Forest'], 'models/final_model.pkl')

print("Pipeline execution completed. Cleaned data, figures, tables, and models have been saved.")
