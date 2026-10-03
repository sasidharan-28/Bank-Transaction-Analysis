import nbformat as nbf

def create_phase1_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# PHASE 1 - Data Collection and Initial Analysis\n\n## 1. Dataset Loading"),
        nbf.v4.new_code_cell("""import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\ndf = pd.read_csv('../data/bank_transactions.csv')\n"""),
        nbf.v4.new_code_cell("""print(f'Number of rows: {df.shape[0]}')\nprint(f'Number of columns: {df.shape[1]}')\nprint('Column names:', df.columns.tolist())\n"""),
        nbf.v4.new_code_cell("""display(df.head())\ndisplay(df.tail())\n"""),
        nbf.v4.new_code_cell("""df.info()\n"""),
        nbf.v4.new_code_cell("""print('Data Types:\\n', df.dtypes)\n"""),
        nbf.v4.new_code_cell("""print('Missing values:\\n', df.isnull().sum())\n"""),
        nbf.v4.new_code_cell("""print('Duplicate rows:', df.duplicated().sum())\n"""),
        nbf.v4.new_markdown_cell("## 2. Data Understanding\n\n- **TransactionID**: Unique identifier for the transaction.\n- **CustomerID**: Unique identifier for the customer.\n- **CustomerDOB**: Date of birth of the customer.\n- **CustGender**: Gender of the customer.\n- **CustLocation**: Location of the customer.\n- **CustAccountBalance**: Account balance before/after the transaction.\n- **TransactionDate**: Date when the transaction occurred.\n- **TransactionTime**: Time when the transaction occurred.\n- **TransactionAmount (INR)**: Amount of the transaction.\n\n**Types:**\n- Identifier: TransactionID, CustomerID\n- Numerical: CustAccountBalance, TransactionTime, TransactionAmount (INR)\n- Categorical: CustGender, CustLocation\n- Date/Time: CustomerDOB, TransactionDate, TransactionTime\n- Transaction-related: TransactionID, TransactionDate, TransactionTime, TransactionAmount (INR)\n- Customer-related: CustomerID, CustomerDOB, CustGender, CustLocation, CustAccountBalance\n"),
        nbf.v4.new_markdown_cell("## 3. Initial Data Quality Analysis"),
        nbf.v4.new_code_cell("""quality_df = pd.DataFrame({\n    'Missing Values': df.isnull().sum(),\n    'Data Type': df.dtypes\n})\ndisplay(quality_df)\n"""),
        nbf.v4.new_code_cell("""print("Describe numerical columns:")\ndisplay(df.describe())\n"""),
        nbf.v4.new_code_cell("""print("Describe categorical columns:")\ndisplay(df.describe(include='object'))\n"""),
        nbf.v4.new_markdown_cell("## 4. Initial EDA & 5. Phase 1 Visualizations"),
        nbf.v4.new_code_cell("""plt.figure(figsize=(8,5))\nsns.histplot(df['TransactionAmount (INR)'].dropna()[:10000], bins=50, kde=True)\nplt.title('Distribution of Transaction Amount (Sampled)')\nplt.xlabel('Transaction Amount (INR)')\nplt.ylabel('Frequency')\nplt.savefig('../outputs/figures/amount_histogram.png')\nplt.show()\n"""),
        nbf.v4.new_code_cell("""plt.figure(figsize=(8,5))\nsns.boxplot(x=df['TransactionAmount (INR)'].dropna()[:10000])\nplt.title('Box Plot of Transaction Amount (Sampled)')\nplt.xlabel('Transaction Amount (INR)')\nplt.savefig('../outputs/figures/amount_boxplot.png')\nplt.show()\n"""),
        nbf.v4.new_code_cell("""plt.figure(figsize=(8,5))\ntop_locations = df['CustLocation'].value_counts().head(10)\nsns.barplot(x=top_locations.index, y=top_locations.values)\nplt.title('Top 10 Customer Locations')\nplt.xlabel('Location')\nplt.ylabel('Count')\nplt.xticks(rotation=45)\nplt.savefig('../outputs/figures/location_barchart.png')\nplt.show()\n"""),
        nbf.v4.new_code_cell("""plt.figure(figsize=(8,5))\ncustomer_freq = df['CustomerID'].value_counts().head(10)\nsns.barplot(x=customer_freq.index, y=customer_freq.values)\nplt.title('Top 10 Customers by Transaction Frequency')\nplt.xlabel('Customer ID')\nplt.ylabel('Transaction Count')\nplt.xticks(rotation=45)\nplt.savefig('../outputs/figures/customer_freq.png')\nplt.show()\n"""),
        nbf.v4.new_markdown_cell("## 6. Phase 1 Findings\n\n1. The dataset contains a large number of transactions with numerical features heavily right-skewed (especially transaction amounts and balances).\n2. Some columns contain missing values, notably `CustomerDOB`, `CustGender`, and `CustLocation`.\n3. A small number of locations account for a significant portion of transactions (e.g. Mumbai, New Delhi, Bangalore).")
    ]
    with open('notebooks/Phase1_Analysis.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_phase2_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# PHASE 2 - Data Preparation and EDA"),
        nbf.v4.new_code_cell("""import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\ndf = pd.read_csv('../data/bank_transactions.csv')\n"""),
        nbf.v4.new_markdown_cell("## 1. Data Cleaning & 2. Missing Values"),
        nbf.v4.new_code_cell("""print("Missing values before cleaning:")\nprint(df.isnull().sum())\n\n# Impute categorical missing values with 'Unknown'\ndf['CustGender'].fillna('Unknown', inplace=True)\ndf['CustLocation'].fillna('Unknown', inplace=True)\n\n# Drop rows with missing crucial numerical values\nnum_rows_before = len(df)\ndf.dropna(subset=['CustAccountBalance', 'TransactionAmount (INR)'], inplace=True)\nprint(f"Dropped {num_rows_before - len(df)} rows due to missing numerical values.")\n"""),
        nbf.v4.new_markdown_cell("## 3. Duplicate Handling"),
        nbf.v4.new_code_cell("""print("Duplicate rows before:", df.duplicated().sum())\ndf.drop_duplicates(inplace=True)\nprint("Duplicate rows after:", df.duplicated().sum())\n"""),
        nbf.v4.new_markdown_cell("## 4. Outlier Analysis & 5. Data Standardization"),
        nbf.v4.new_code_cell("""# Standardize category names\ndf['CustGender'] = df['CustGender'].str.upper().str.strip()\n\n# Create an is_outlier flag based on IQR for TransactionAmount (but we keep the data)\nQ1 = df['TransactionAmount (INR)'].quantile(0.25)\nQ3 = df['TransactionAmount (INR)'].quantile(0.75)\nIQR = Q3 - Q1\nupper_bound = Q3 + 1.5 * IQR\ndf['is_amount_outlier'] = df['TransactionAmount (INR)'] > upper_bound\nprint(f"Flagged {df['is_amount_outlier'].sum()} transactions as high-value outliers.")\n\ndf.to_csv('../data/Bank_Transactions_Cleaned.csv', index=False)\n"""),
        nbf.v4.new_markdown_cell("## 6. Exploratory Data Analysis & 7. Visualizations"),
        nbf.v4.new_code_cell("""sample_df = df.sample(10000, random_state=42) # sample for fast plotting\n\nplt.figure(figsize=(8,5))\nsns.histplot(sample_df['CustAccountBalance'], bins=50, kde=True)\nplt.title('Balance Histogram')\nplt.xlabel('Account Balance')\nplt.ylabel('Frequency')\nplt.savefig('../outputs/figures/balance_histogram.png')\nplt.show()\n"""),
        nbf.v4.new_code_cell("""plt.figure(figsize=(8,5))\nsns.scatterplot(data=sample_df, x='CustAccountBalance', y='TransactionAmount (INR)', alpha=0.3)\nplt.title('Amount vs Balance')\nplt.xlabel('Account Balance')\nplt.ylabel('Transaction Amount (INR)')\nplt.savefig('../outputs/figures/amount_vs_balance.png')\nplt.show()\n"""),
        nbf.v4.new_code_cell("""plt.figure(figsize=(8,6))\nnumeric_cols = df.select_dtypes(include=['float64', 'int64']).columns\ncorr = df[numeric_cols].corr()\nsns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")\nplt.title('Correlation Heatmap')\nplt.savefig('../outputs/figures/correlation_heatmap.png')\nplt.show()\n"""),
        nbf.v4.new_code_cell("""print("Data cleaning and EDA visualizations have been successfully completed and saved.")\n""")
    ]
    with open('notebooks/Phase2_EDA.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_phase3_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# PHASE 3 - Machine Learning Model Development\n\n**Note**: The derived target `High_Value_Transaction` is used for demonstration purposes. The original dataset does not contain a real fraud label."),
        nbf.v4.new_code_cell("""import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report\nimport joblib\nimport warnings\nwarnings.filterwarnings('ignore')\n\ndf = pd.read_csv('../data/Bank_Transactions_Cleaned.csv')\n"""),
        nbf.v4.new_markdown_cell("## 1. Feature Selection & 2. Target Creation"),
        nbf.v4.new_code_cell("""# Selecting the requested features\nfeatures = ['CustAccountBalance', 'TransactionTime', 'TransactionAmount (INR)']\nX = df[features].copy()\n\n# Create Target: High_Value_Transaction if Amount > 2000 (just a threshold for demonstration)\nthreshold = 2000\ny = (X['TransactionAmount (INR)'] > threshold).astype(int)\nprint(f"Target 'High_Value_Transaction' created with threshold > {threshold}. Count: {y.sum()}")\n"""),
        nbf.v4.new_markdown_cell("## 3. Train/Test Split"),
        nbf.v4.new_code_cell("""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\nprint(f"Training set size: {X_train.shape}")\nprint(f"Test set size: {X_test.shape}")\n"""),
        nbf.v4.new_markdown_cell("## 4. Missing-Value Handling & 5. Algorithms"),
        nbf.v4.new_code_cell("""def evaluate_model(model, X_test, y_test):\n    y_pred = model.predict(X_test)\n    return {\n        'Accuracy': accuracy_score(y_test, y_pred),\n        'Precision': precision_score(y_test, y_pred),\n        'Recall': recall_score(y_test, y_pred),\n        'F1-Score': f1_score(y_test, y_pred)\n    }\n\n# Create preprocessing pipeline\npreprocessor = Pipeline([\n    ('imputer', SimpleImputer(strategy='median')),\n    ('scaler', StandardScaler())\n])\n\nmodels = {\n    'Logistic Regression': Pipeline([('prep', preprocessor), ('clf', LogisticRegression())]),\n    'Decision Tree': Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(max_depth=5, random_state=42))]),\n    'Random Forest': Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42))])\n}\n\nresults = []\nfor name, model in models.items():\n    model.fit(X_train, y_train)\n    metrics = evaluate_model(model, X_test, y_test)\n    metrics['Model'] = name\n    results.append(metrics)\n    print(f"--- {name} ---")\n    print(classification_report(y_test, model.predict(X_test)))\n"""),
        nbf.v4.new_markdown_cell("## 7. Model Comparison"),
        nbf.v4.new_code_cell("""results_df = pd.DataFrame(results)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]\ndisplay(results_df)\nresults_df.to_csv('../outputs/tables/model_comparison.csv', index=False)\n"""),
        nbf.v4.new_markdown_cell("## 8. Final Model Selection"),
        nbf.v4.new_code_cell("""# Select Random Forest as the final model based on balanced metrics (usually F1 is best for imbalanced)\nfinal_model = models['Random Forest']\njoblib.dump(final_model, '../models/final_model.pkl')\nprint("Final model saved successfully to '../models/final_model.pkl'.")\n""")
    ]
    with open('notebooks/Phase3_Model.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_phase1_notebook()
    create_phase2_notebook()
    create_phase3_notebook()
    print("All notebooks generated successfully.")
