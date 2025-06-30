# 🔬 Heart Disease Prediction – Internship Project

This repository contains my complete AI & ML internship project on the **Heart Disease dataset**. It includes data preprocessing, exploratory data analysis (EDA), regression, and classification modeling using Python and scikit-learn.

---

## 📁 Folder Structure

heart-disease-analysis-internship/
│
├── task-1-preprocessing/
│ ├── heart.csv
│ ├── task1_heart_preprocessing.ipynb
│ └── README.md
│
├── task-2-eda/
│ ├── task2_eda.ipynb
│ └── README.md
│
├── task-3-regression/
│ ├── task3_linear_regression.ipynb
│ └── README.md
│
├── task-4-logistic-regression/
│ ├── task4_logistic_regression.ipynb
│ └── README.md

yaml
Copy
Edit

---

## 📊 Dataset Used

- **Name:** Heart Disease UCI Dataset  
- **Target Variable:** `HeartDisease` → (1 = disease present, 0 = no disease)  
- **Data Types:** Mix of numerical and categorical features  
- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

---

## ✅ Task Overview

---

### 🔹 Task 1: Data Preprocessing

📄 File: `task1_heart_preprocessing.ipynb`

**Steps Performed:**
- Handled missing values
- Encoded categorical columns (Sex, ChestPainType, etc.)
- Standardized numeric features using `StandardScaler`
- Detected and removed outliers using IQR method
- Final cleaned dataset was saved for future tasks

---

### 🔹 Task 2: Exploratory Data Analysis (EDA)

📄 File: `task2_eda.ipynb`

**Key Analyses:**
- Histograms for age, cholesterol, blood pressure
- Boxplots for detecting outliers
- Correlation matrix using heatmap
- Pairplots to visualize class separation

**Findings:**
- `Oldpeak` and `MaxHR` had strong influence on heart disease
- `Cholesterol` showed weak correlation
- Few outliers detected in `Cholesterol` and `Oldpeak`

---

### 🔹 Task 3: Linear Regression

📄 File: `task3_linear_regression.ipynb`

**Objective:**  
Predict the `RestingBP` (resting blood pressure) using features like Age, Cholesterol, MaxHR, and Oldpeak.

**Results:**
- **Mean Absolute Error (MAE):** ~12.26
- **Mean Squared Error (MSE):** ~150.92
- **R² Score:** ~0.23 (moderate)

**Regression Equation Example:**
\[
RestingBP = 84.93 + (0.27 \times Age) - (0.026 \times Cholesterol) + (0.22 \times MaxHR) + (8.82 \times Oldpeak)
\]

---

### 🔹 Task 4: Logistic Regression

📄 File: `task4_logistic_regression.ipynb`

**Objective:**  
Predict whether a person has heart disease using logistic regression.

**Model Evaluation:**
- **Confusion Matrix:**  
[[0 0]
[1 2]]

yaml
Copy
Edit
- **Precision (class 1):** 1.00  
- **Recall (class 1):** 0.67  
- **F1-Score (class 1):** 0.80  
- **ROC-AUC:** Not computed (only one class in test set)

**Key Concepts Covered:**
- Sigmoid Function
- Confusion Matrix
- Precision vs Recall
- ROC-AUC Curve
- Class imbalance handling

---

## 📌 Technologies Used

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook / Google Colab
- GitHub

---

## 🧠 Learning Outcomes

- Data preprocessing techniques (scaling, encoding, outlier removal)
- Exploratory data analysis and feature insights
- Applied linear and logistic regression from scratch
- Evaluated models using real-world metrics
- Improved understanding of binary classification
- Practiced GitHub project structure and submission

---

## 👨‍💻 Author

**Pavithiran**  
B.Tech – Artificial Intelligence & Machine Learning  
Sri Shakthi Institute of Engineering and Technology, Coimbatore

---

## 📎 License

This project is for educational and internship submission purposes only.
