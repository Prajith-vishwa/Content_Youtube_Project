import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("youtube_ad_revenue_dataset.csv")
print(f" Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Basic feature engineering
df["likes"] = df["likes"].fillna(0)
df["comments"] = df["comments"].fillna(0)
df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"].replace(0, np.nan)
df["engagement_rate"] = df["engagement_rate"].fillna(0)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["upload_month"] = df["date"].dt.month
df["upload_dayofweek"] = df["date"].dt.dayofweek

# Drop unnecessary columns
df = df.drop(columns=["video_id", "date"], errors="ignore")

# Define features and target
target = "ad_revenue_usd"
X = df.drop(columns=[target], errors="ignore")
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_features = ["views", "likes", "comments", "watch_time_minutes", "video_length_minutes", "subscribers", "engagement_rate"]
categorical_features = ["category", "device", "country", "upload_month", "upload_dayofweek"]

numeric_features = [f for f in numeric_features if f in X.columns]
categorical_features = [f for f in categorical_features if f in X.columns]

numeric_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(random_state=42),
    "Lasso Regression": Lasso(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
}

results = {}
for name, model in models.items():
    print(f"\n Training {name}...")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
    print(f" {name}: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

# Choose best model
results_df = pd.DataFrame(results).T.sort_values(by="R2", ascending=False)
best_model_name = results_df.index[0]
print("\n Best Model:", best_model_name)

# Save model
best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", models[best_model_name])
])
best_pipeline.fit(X_train, y_train)
joblib.dump(best_pipeline, "best_model.joblib")
print(" Model saved as best_model.joblib")
