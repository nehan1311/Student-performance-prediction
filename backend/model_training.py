import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
import joblib

print("🔧 Loading dataset...")

# Load dataset
df = pd.read_csv("../data/student_habits_performance.csv")

# Normalize attendance (VERY IMPORTANT)
df["attendance_percentage"] = df["attendance_percentage"] / 100

# Add classification target
df["pass_fail"] = (df["exam_score"] >= 60).astype(int)

# Features used in the ML model
feature_cols = [
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "attendance_percentage",
    "sleep_hours"
]

X = df[feature_cols]
y_class = df["pass_fail"]
y_reg = df["exam_score"]

print("🔧 Single train-test split for BOTH regression & classification...")

# ONE consistent split for ALL tasks
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

# Regression target (same split!)
y_train_reg = y_reg.loc[X_train.index]
y_test_reg = y_reg.loc[X_test.index]

print("🔧 Scaling features...")

scaler = StandardScaler()
scaler.fit(X_train)  # Fit ONLY on training data

# Transform both train and test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Both classification and regression use SAME scaled data
X_train_scaled_cls = X_train_scaled
X_test_scaled_cls = X_test_scaled

X_train_scaled_reg = X_train_scaled
X_test_scaled_reg = X_test_scaled

print("🤖 Training classification models...")

clf_models = [
    LogisticRegression(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    KNeighborsClassifier(),
    SVC(probability=True)
]

for model in clf_models:
    model.fit(X_train_scaled_cls, y_train_class)

print("🤖 Training regression models...")

reg_models = [
    LinearRegression(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    KNeighborsRegressor(),
    SVR()
]

for model in reg_models:
    model.fit(X_train_scaled_reg, y_train_reg)

print("💾 Saving models...")

joblib.dump(clf_models, "./models/classifier_ensemble.pkl")
joblib.dump(reg_models, "./models/regressor_ensemble.pkl")
joblib.dump(scaler, "./models/scaler.pkl")

print("🎯 Training KMeans clustering model...")

# Clustering must use classification-scaled features
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train_scaled_cls)

joblib.dump(kmeans, "./models/kmeans_clusters.pkl")

print("📦 KMeans clustering model saved!")
print("🎉 All models trained & saved successfully!")
