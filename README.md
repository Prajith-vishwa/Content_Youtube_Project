# Content_Youtube_Project
A Machine Learning + Streamlit project to estimate YouTube video ad revenue based on performance and audience engagement metrics.

**Project Overview**

YouTube monetization depends on multiple factors such as views, likes, comments, watch time, category, device type, and location.
This project builds a regression-based ML model to predict expected YouTube Ad Revenue and provides an easy-to-use Streamlit web application for real-time predictions.

**Key Features**

Predict estimated YouTube ad revenue

Interactive Streamlit UI

Automated feature engineering

Preprocessing pipeline (scaling + encoding)

Trained ML models (Random Forest, Linear, Ridge, etc.)

Visualization support in the app

Deployed using Streamlit locally

**Project Structure**
├── app.py                 # Streamlit web application (version 1)
├── main.py                # Enhanced Streamlit + prediction logic
├── content.ipynb          # Jupyter notebook: preprocessing & training
├── best_model.pkl         # Saved trained ML model
├── preprocessor.pkl       # Saved preprocessing pipeline
├── README.md              # Documentation file
└── dataset.csv            # Your dataset (if added)

**Technologies Used**

Python

Streamlit

Scikit-Learn

Pandas, NumPy

Matplotlib / Seaborn

Joblib

**Model Building Workflow**
Step 1 — Data Collection

Dataset contains:

Views

Likes

Comments

Watch Time

Video Length

Category

Device

Country

Step 2 — Data Cleaning

Removed missing or null values

Converted data types

Handled outliers in views/likes/comments

Ensured consistency in categorical fields

Step 3 — Feature Engineering

New features created:

Engagement Rate = (Likes + Comments) / Views

Watch Time per View = WatchTime / Views

Likes per View

Comments per View

Watch Share = WatchTime / VideoLength

Date Features: Month, DayOfWeek, WeekendFlag

Step 4 — Data Preprocessing

Scaling (StandardScaler) for numeric values

OneHotEncoding for categorical values

Train/Test split (80% / 20%)

Step 5 — Model Selection

Models tested:

Linear Regression

Ridge & Lasso

Decision Tree

Random Forest

Random Forest performed best → selected as final model.

Step 6 — Model Training

Trained on processed data

Performed hyperparameter tuning

Exported the model as best_model.pkl

Step 7 — Model Evaluation

Metrics used:

R² Score

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Random Forest delivered the highest evaluation performance.

Step 8 — Deployment (Streamlit App)

The Streamlit app provides:
✔ Real-time user inputs
✔ Automated preprocessing
✔ Revenue prediction
✔ Visual charts
✔ User-friendly interface
