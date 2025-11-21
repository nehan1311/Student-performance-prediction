import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
import joblib

print("🔧 Loading dataset...")

# Load dataset
df = pd.read_csv("../data/student_habits_performance.csv")

# Create Pass/Fail label
df["pass_fail"] = (df["exam_score"] >= 60).astype(int)

# Use 5 user-input features
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

print("🔧 Splitting dataset...")

# Train-test split
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

print("🔧 Scaling features...")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled_reg = scaler.fit_transform(X_train_reg)

print("🤖 Training classification models...")

# --- CLASSIFICATION MODELS ---
clf_models = [
    LogisticRegression(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    KNeighborsClassifier(),
    SVC(probability=True)
]

for model in clf_models:
    model.fit(X_train_scaled, y_train_class)

def classifier_ensemble_predict(X):
    probs = [model.predict_proba(X)[0][1] for model in clf_models]
    return 1 if np.mean(probs) >= 0.5 else 0

print("🤖 Training regression models...")

# --- REGRESSION MODELS ---
reg_models = [
    LinearRegression(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    KNeighborsRegressor(),
    SVR()
]

for model in reg_models:
    model.fit(X_train_scaled_reg, y_train_reg)

def regressor_ensemble_predict(X):
    preds = [model.predict(X)[0] for model in reg_models]
    return np.mean(preds)

print("💾 Saving models...")

joblib.dump(clf_models, "./models/classifier_ensemble.pkl")
joblib.dump(reg_models, "./models/regressor_ensemble.pkl")
joblib.dump(scaler, "./models/scaler.pkl")

print("🎉 Advanced Ensemble Models trained & saved successfully!")
