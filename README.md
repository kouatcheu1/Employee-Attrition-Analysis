# Employee Attrition Prediction Using Machine Learning 👨‍💼📊

## 📌 Introduction
This project implements various **machine learning models to predict employee attrition** based on multiple factors, including **work environment, compensation, and personal characteristics**. The analysis helps identify **key factors contributing to employee turnover** and builds predictive models to forecast potential attrition.

## ❓ Problem Statement
Employee attrition can significantly impact organizational performance and costs. This project aims to:
- **Predict the likelihood of employee attrition**  
- **Identify key factors influencing employee turnover**  
- **Compare different ML models' performance in predicting attrition**  
- **Provide insights for HR decision-making**  

## 📂 Dataset
The dataset contains **1,000 employee records** with **26 features**, categorized as follows:

### **Employee Information**
- **Personal**: Employee_ID, Age, Gender, Marital_Status  
- **Professional**: Department, Job_Role, Job_Level  
- **Experience**: Years_at_Company, Years_in_Current_Role, Years_Since_Last_Promotion  

### **Work Metrics**
- **Compensation**: Monthly_Income, Hourly_Rate  
- **Performance**: Performance_Rating, Job_Involvement  
- **Workload**: Project_Count, Average_Hours_Worked_Per_Week, Overtime  
- **Training**: Training_Hours_Last_Year  

### **Satisfaction Metrics**
- **Work_Life_Balance**  
- **Job_Satisfaction**  
- **Work_Environment_Satisfaction**  
- **Relationship_with_Manager**  

### **Other Metrics**
- **Distance_From_Home**  
- **Number_of_Companies_Worked**  
- **Absenteeism**  
- **Target Variable**: Attrition (**Yes/No**)  

## 🛠 Technologies Used
- **Python 3.x**  
- **Libraries**:  
  - `pandas`, `numpy` – Data manipulation  
  - `scikit-learn` – Machine learning implementation  
  - `matplotlib`, `seaborn` – Data visualization  
  - `imblearn` – Handling imbalanced data  
  - `xgboost` – XGBoost implementation  

## 🔬 Methodology

### **1️⃣ Data Preprocessing**
- **Label encoding** for categorical variables  
- **Feature standardization** using `StandardScaler`  
- **SMOTE** for handling class imbalance  

### **2️⃣ Model Implementation**
- **Logistic Regression**  
- **K-Nearest Neighbors (KNN)**  
- **Decision Tree**  
- **Random Forest**  
- **XGBoost**  

### **3️⃣ Model Optimization**
- **Hyperparameter tuning** using `GridSearchCV`  
- **Cross-validation** for model evaluation  
- **ROC-AUC score** as the primary evaluation metric  

## 📊 Results

### **Model Performance (ROC-AUC Scores)**
| Model | ROC-AUC Score |
|--------|--------------|
| Logistic Regression | 0.694 |
| KNN | 0.955 |
| Decision Tree | 0.799 |
| Random Forest | 0.947 |
| XGBoost | 0.938 |

### **Best Performing Models**
#### ✅ **K-Nearest Neighbors (KNN)**
- **Best Parameters**: `{'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}`  
- **Highest ROC-AUC Score**: `0.954`  

#### ✅ **Random Forest**
- **Best Parameters**: `{'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}`  
- **ROC-AUC Score**: `0.946`  

## 🎯 Key Features and Functionality
- **Comprehensive data preprocessing pipeline**  
- **Implementation of multiple ML algorithms**  
- **Hyperparameter optimization**  
- **Model performance comparison**  
- **Visualization of results**  
- **Handling of imbalanced data**  

---

🚀 **Explore the Jupyter Notebook for complete code and insights!**
![Employee Attrition Analysis](https://github.com/kouatcheu1/Employee-Attrition-Analysis/blob/main/Attrition.ipynb)
