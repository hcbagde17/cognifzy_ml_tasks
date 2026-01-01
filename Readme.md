# 🍽️ Restaurant Data Analysis & Machine Learning Project

This repository contains a comprehensive data analysis and machine learning workflow performed on a restaurant dataset.  
The project is structured into multiple tasks, each focusing on a different real-world data science problem such as regression, classification, and geospatial analysis.

---

## 📁 Project Structure

COGNIFYZ_ML_TASKS/
│
├── task1/
│   ├── cleaned_dataset.csv
│   └── final_preprocessing&regression.ipynb
│
├── task3/
│   └── preprocess&prediction.ipynb
│
├── task4/
│   ├── preprocess&analysis.ipynb
│   └── restaurants_map.html
│
├── Dataset.csv
├── partial_cleaned_dataset.csv
├── eda.ipynb
├── preprocessing.ipynb
├── .gitattributes
└── README.md

---

## 📊 Dataset Overview

- **Source**: Restaurant dataset (Kaggle-style)
- **Key Attributes**:
  - Aggregate rating
  - Votes
  - Cuisines
  - Average cost for two
  - Price range
  - Online delivery & table booking
  - Latitude & Longitude
  - Country and city information

The dataset undergoes multiple stages of cleaning and transformation before being used in different machine learning tasks.

---

## 🔍 Exploratory Data Analysis (EDA)

### 📓 `eda.ipynb`
- Initial exploration of the raw dataset
- Handling missing values
- Distribution analysis of ratings and cost
- Correlation analysis
- Identification of skewness and outliers
- Insights used to guide preprocessing decisions

---

## 🧹 Data Preprocessing

### 📓 `preprocessing.ipynb`
- Currency conversion to INR
- Handling missing cuisine values
- Feature engineering (e.g., cuisine count)
- Encoding binary categorical variables
- Exporting intermediate cleaned datasets

### 📄 `partial_cleaned_dataset.csv`
- Output of early-stage preprocessing

---

## ✅ Task 1: Restaurant Rating Prediction (Regression)

### 🎯 Objective
Predict the **Aggregate Rating** of a restaurant using numerical and categorical features.

### 📁 Location
`task1/final_preprocessing&regression.ipynb`

### 🛠️ Key Steps
- Leakage-aware feature selection
- Log transformation of skewed cost feature
- Feature scaling
- Train-test split
- Model training and comparison

### 🤖 Models Used
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

### 📈 Evaluation Metrics
- R² Score
- Mean Squared Error (MSE)

### 🧠 Key Insight
Two versions of the model were analyzed:
- **With Votes** (high R² but leakage-prone)
- **Without Votes** (realistic and deployable)

---

## 🍜 Task 3: Cuisine Classification (Multi-class Classification)

### 🎯 Objective
Classify restaurants based on their **primary cuisine**.

### 📁 Location
`task3/preprocess&prediction.ipynb`

### 🛠️ Key Steps
- Extraction of primary cuisine from multi-cuisine entries
- Label encoding of target variable
- Feature selection and encoding
- Stratified train-test split
- Feature scaling

### 🤖 Models Used
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### ⚠️ Challenges Addressed
- Class imbalance across cuisines
- Overlapping restaurant characteristics
- Multi-cuisine ambiguity

---

## 🌍 Task 4: Geospatial Analysis of Restaurants

### 🎯 Objective
Analyze restaurant distribution and patterns using geographical data.

### 📁 Location
`task4/preprocess&analysis.ipynb`

### 🛠️ Key Steps
- Latitude & longitude based analysis
- City and region-wise clustering
- Visualization of restaurant density
- Cost and rating variation by location

### 🗺️ Output
- **Interactive Map**: `task4/restaurants_map.html`
  - Visualizes restaurant locations and spatial patterns

---

## 🧪 Technologies & Libraries Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Folium (for geospatial visualization)

---

## ✅ Key Learnings

- Importance of preventing data leakage
- Difference between analytical and deployable models
- Feature engineering for skewed and categorical data
- Regression vs classification modeling strategies
- Practical geospatial data analysis

---

## 📌 Conclusion

This project demonstrates an end-to-end data science workflow:
- From raw data exploration
- To preprocessing and feature engineering
- To multiple machine learning tasks
- To visualization and interpretation

Each task is modular, reproducible, and aligned with real-world machine learning practices.

---

📬 **Author**: Harsh Bagde  
🎓 **Domain**: Data Science & Machine Learning
