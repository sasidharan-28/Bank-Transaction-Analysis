from docx import Document
from docx.shared import Inches

def create_report():
    doc = Document()
    doc.add_heading('Bank Transaction Analysis and Customer Behavior Insights', 0)
    
    doc.add_heading('1. Abstract', level=1)
    doc.add_paragraph('This project analyzes bank transaction data to understand transaction patterns, perform exploratory data analysis, and build a machine learning model to predict high-value transactions. A Streamlit application provides an interactive dashboard for exploring the results.')
    
    doc.add_heading('2. Introduction', level=1)
    doc.add_paragraph('Understanding customer behavior and transaction patterns is crucial for financial institutions. This project aims to extract actionable insights from bank transaction records.')
    
    doc.add_heading('3. Problem Statement', level=1)
    doc.add_paragraph('Financial data often contains anomalies, missing values, and highly skewed distributions. The challenge is to clean this data, uncover meaningful patterns, and build a predictive model for transaction risk/value.')
    
    doc.add_heading('4. Objectives', level=1)
    doc.add_paragraph('- Clean and preprocess bank transaction data.\n- Perform Exploratory Data Analysis (EDA).\n- Build classification models to predict high-value transactions.\n- Present findings via an interactive dashboard.')
    
    doc.add_heading('5. Dataset Description', level=1)
    doc.add_paragraph('The dataset contains transaction records including CustomerID, Account Balance, Location, Gender, Date, Time, and Transaction Amount.')
    
    doc.add_heading('6. Phase 1 – Data Collection and Initial Analysis', level=1)
    doc.add_paragraph('Data was loaded and initial checks for missing values and duplicates were performed. The dataset exhibited right-skewed distributions for transaction amounts.')
    
    doc.add_heading('7. Phase 2 – Data Preparation and EDA', level=1)
    doc.add_paragraph('Missing values were handled (categorical replaced with "Unknown", numeric missing dropped if critical). Duplicates were removed. Outliers were analyzed using IQR. Correlation heatmaps and various distribution charts were generated.')
    
    doc.add_heading('8. Phase 3 – Machine Learning', level=1)
    doc.add_heading('Feature Selection & Target Creation', level=2)
    doc.add_paragraph('Features used: CustAccountBalance, TransactionTime, TransactionAmount (INR). A derived target "High_Value_Transaction" was created (1 if Amount > 2000 else 0). Note: This is a derived target for demonstration, not a real-world fraud label.')
    
    doc.add_heading('Model Training & Evaluation', level=2)
    doc.add_paragraph('Models trained: Logistic Regression, Decision Tree, Random Forest. Random Forest was selected as the final model due to its superior F1-Score in handling non-linear relationships.')
    
    doc.add_heading('9. Phase 4 – Final Application', level=1)
    doc.add_paragraph('A Streamlit dashboard was developed. It features tabs for Summary Dashboard, Transaction Analysis, Customer Analysis, and Model Prediction. It allows users to input transaction details and get a prediction on whether it is a high-value transaction.')
    
    doc.add_heading('10. Limitations & Future Enhancements', level=1)
    doc.add_paragraph('Limitations: Lack of a genuine fraud label; reliance on a heuristic threshold for classification.\nFuture Enhancements: Integration of real fraud data, advanced temporal feature engineering, and deployment to a cloud server.')
    
    doc.add_heading('11. Conclusion', level=1)
    doc.add_paragraph('The project successfully demonstrates an end-to-end data science workflow from data cleaning to model deployment, providing valuable insights into customer transaction behavior.')
    
    doc.save('report/Final_Project_Report.docx')
    print("Report generated successfully.")

if __name__ == '__main__':
    create_report()
