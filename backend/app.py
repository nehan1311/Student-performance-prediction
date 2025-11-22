# backend/app.py
from flask import Flask, render_template, request, jsonify, url_for
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import traceback
import os
import json

# Set matplotlib to use non-interactive backend for web applications
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

BASE = Path(__file__).resolve().parent

# Load models
clf_models = joblib.load(BASE/"models/classifier_ensemble.pkl")
reg_models = joblib.load(BASE/"models/regressor_ensemble.pkl")
scaler = joblib.load(BASE/"models/scaler.pkl")

# Load KMeans model (if it exists)
try:
    kmeans = joblib.load(BASE/"models/kmeans_clusters.pkl")
except FileNotFoundError:
    kmeans = None
    print("⚠️  KMeans model not found. Run model_training.py to generate it.")

# Load cluster mapping (if it exists)
try:
    cluster_map = json.load(open(BASE/"models/cluster_map.json"))
except FileNotFoundError:
    cluster_map = {"0": "high_risk", "1": "moderate", "2": "good"}  # Default mapping
    print("⚠️  Cluster map not found. Using default mapping.")


def generate_eda_images(force=False):
    """
    Generates EDA PNG images into backend/static/eda/.
    If force=True, regenerate even if files exist.
    Returns dict of relative static paths (for templates).
    """
    eda_dir = BASE.joinpath("static", "eda")
    os.makedirs(eda_dir, exist_ok=True)

    # prefer project data copy if present, else fallback to uploaded path
    data_path_candidates = [
        BASE.joinpath("..", "data", "student_habits_performance.csv"),  # recommended
        Path("/mnt/data/student_habits_performance.csv")                # uploaded
    ]
    df = None
    for p in data_path_candidates:
        if Path(p).exists():
            df = pd.read_csv(p)
            break
    if df is None:
        raise FileNotFoundError("Dataset not found. Please place student_habits_performance.csv in data/ or use /mnt/data/ path.")

    # Numeric columns we care about (existing in your CSV)
    numeric_cols = [
        "study_hours_per_day",
        "social_media_hours",
        "netflix_hours",
        "attendance_percentage",
        "sleep_hours",
        "exam_score"
    ]
    # Keep only present columns
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    imgs = {}

    # 1) Histogram of exam_score
    hist_path = eda_dir.joinpath("exam_score_hist.png")
    if force or not hist_path.exists():
        plt.figure(figsize=(6,4))
        sns.histplot(df["exam_score"].dropna(), bins=20, kde=True)
        plt.title("Exam Score Distribution")
        plt.xlabel("Exam Score")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=150)
        plt.close()
    imgs["exam_hist"] = url_for("static", filename=f"eda/{hist_path.name}")

    # 2) Correlation heatmap (only numeric columns)
    corr_path = eda_dir.joinpath("corr_heatmap.png")
    if force or not corr_path.exists():
        corr_df = df[numeric_cols].dropna()
        plt.figure(figsize=(7,6))
        sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
        plt.title("Correlation Heatmap (numeric features)")
        plt.tight_layout()
        plt.savefig(corr_path, dpi=150)
        plt.close()
    imgs["corr"] = url_for("static", filename=f"eda/{corr_path.name}")

    # 3) Scatter: study_hours vs exam_score
    if "study_hours_per_day" in df.columns and "exam_score" in df.columns:
        scatter_path = eda_dir.joinpath("study_vs_score.png")
        if force or not scatter_path.exists():
            plt.figure(figsize=(6,4))
            sns.scatterplot(data=df, x="study_hours_per_day", y="exam_score")
            sns.regplot(data=df, x="study_hours_per_day", y="exam_score", scatter=False, color="red")
            plt.title("Study Hours vs Exam Score")
            plt.tight_layout()
            plt.savefig(scatter_path, dpi=150)
            plt.close()
        imgs["study_scatter"] = url_for("static", filename=f"eda/{scatter_path.name}")

    # 4) Boxplot: attendance
    if "attendance_percentage" in df.columns:
        box_path = eda_dir.joinpath("attendance_box.png")
        if force or not box_path.exists():
            plt.figure(figsize=(6,3))
            sns.boxplot(x=df["attendance_percentage"].dropna())
            plt.title("Attendance (%) - Boxplot")
            plt.tight_layout()
            plt.savefig(box_path, dpi=150)
            plt.close()
        imgs["attendance_box"] = url_for("static", filename=f"eda/{box_path.name}")

    # 5) Distribution social_media_hours
    if "social_media_hours" in df.columns:
        sm_path = eda_dir.joinpath("social_media_dist.png")
        if force or not sm_path.exists():
            plt.figure(figsize=(6,4))
            sns.histplot(df["social_media_hours"].dropna(), bins=20, kde=True)
            plt.title("Social Media Hours Distribution")
            plt.tight_layout()
            plt.savefig(sm_path, dpi=150)
            plt.close()
        imgs["social_dist"] = url_for("static", filename=f"eda/{sm_path.name}")

    return imgs


# FEATURE ORDER (must match training)
model_feature_order = [
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "attendance_percentage",
    "sleep_hours"
]

# Mapping from frontend → backend expected feature names
# Note: health_score is NOT mapped as it's not a model feature
frontend_to_backend = {
    "study_hours": "study_hours_per_day",
    "social_activity": "social_media_hours",
    "attendance": "attendance_percentage",
    "sleep_hours": "sleep_hours",
    # netflix_hours must be provided directly or will be missing
    "netflix_hours": "netflix_hours"
}


def prepare_features(payload):
    values = []
    missing = []

    for model_feature in model_feature_order:
        # find frontend field name for this feature
        frontend_key = None
        for fk, mv in frontend_to_backend.items():
            if mv == model_feature:
                frontend_key = fk
                break

        if frontend_key is None:
            # Fallback: use model feature name if mapping not found
            frontend_key = model_feature

        # Try frontend key first, then model feature name as fallback
        value = payload.get(frontend_key) or payload.get(model_feature)

        if value in (None, ""):
            # Store the frontend key name for user-friendly error message
            missing.append(frontend_key)
            values.append(np.nan)
            continue

        val = float(value)

        # Normalize attendance (divide by 100)
        if frontend_key == "attendance":
            val = val / 100

        values.append(val)

    if missing:
        raise ValueError(f"Missing input(s): {', '.join(missing)}")

    return np.array(values).reshape(1, -1)


def ensemble_classify(X):
    probs = []
    for model in clf_models:
        if hasattr(model, "predict_proba"):
            probs.append(model.predict_proba(X)[0][1])
        else:
            probs.append(float(model.predict(X)[0]))
    mean_prob = np.mean(probs)
    return (1 if mean_prob >= 0.5 else 0), mean_prob


def ensemble_regress(X):
    preds = [float(model.predict(X)[0]) for model in reg_models]
    return float(np.mean(preds)), preds


def clustering_recommendations(features):
    """
    Use KMeans cluster to give group-based recommendations.
    """
    if kmeans is None:
        return ["⚠️ Clustering model not available. Please train the model first."]
    
    # Use the same feature preparation as prediction to ensure consistency
    try:
        fvals = prepare_features(features)
    except:
        # Fallback if prepare_features fails - use correct feature names
        # Note: health_score is NOT used as it's not a model feature
        fvals = np.array([
            float(features.get("study_hours", 0) or features.get("study_hours_per_day", 0)),
            float(features.get("social_activity", 0) or features.get("social_media_hours", 0)),
            float(features.get("netflix_hours", 0)),  # Use netflix_hours directly, not health_score
            float(features.get("attendance", 0) or features.get("attendance_percentage", 0)),
            float(features.get("sleep_hours", 0)),
        ]).reshape(1, -1)

    # Scale just like training
    fvals_scaled = scaler.transform(fvals)

    # Predict cluster
    cluster_id = int(kmeans.predict(fvals_scaled)[0])
    
    # Get cluster type from mapping
    cluster_type = cluster_map.get(str(cluster_id), "moderate")

    rec = []

    # ---------------------------
    # CLUSTER-BASED RECOMMENDATIONS
    # ---------------------------

    if cluster_type == "high_risk":
        rec.append("🟥 Cluster: High-risk behavior detected.")
        rec.append("• Increase study time and reduce distractions.")
        rec.append("• Start small daily study targets (30–45 mins).")
        rec.append("• Reduce social media + Netflix during study hours.")
        rec.append("• Try to attend more classes regularly.")

    elif cluster_type == "moderate":
        rec.append("🟧 Cluster: Moderate / average group.")
        rec.append("• You have balanced habits, but improvement is possible.")
        rec.append("• Optimize study hours with planned scheduling.")
        rec.append("• Improve sleep consistency for better retention.")
        rec.append("• Try mock tests weekly to boost exam performance.")

    elif cluster_type == "good":
        rec.append("🟩 Cluster: Good habits group.")
        rec.append("• Great job! Maintain consistent study hours.")
        rec.append("• Practice advanced revision techniques.")
        rec.append("• Continue your healthy sleep and routine habits.")
        rec.append("• Keep distractions low for best performance.")

    return rec


# ROUTES
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = {k: request.form.get(k) for k in request.form.keys()}

        X_raw = prepare_features(payload)
        X_scaled = scaler.transform(X_raw)

        cls, prob = ensemble_classify(X_scaled)
        reg_pred, reg_preds = ensemble_regress(X_scaled)

        # Get clustering recommendations
        cluster_based = clustering_recommendations(payload)
        
        # Combine recommendations (if other recommendation systems exist, add them here)
        recommendations = (
            ["--- Based on your behavior group (Clustering) ---"]
            + cluster_based
        )

        return render_template(
            "index.html",
            predicted_label="PASS" if cls == 1 else "FAIL",
            predicted_score=round(reg_pred, 2),
            mean_prob=round(prob, 3),
            individual_reg_preds=[round(p, 2) for p in reg_preds],
            recommendations=recommendations
        )
    except Exception as e:
        return render_template("index.html", error=str(e)), 400


@app.route("/dashboard")
def dashboard():
    # existing model lists
    clf_names = [type(m).__name__ for m in clf_models]
    reg_names = [type(m).__name__ for m in reg_models]

    # generate EDA files (safe - won't re-create if exist)
    try:
        eda_images = generate_eda_images(force=False)
    except Exception as e:
        eda_images = {}
        eda_error = str(e)
    else:
        eda_error = None

    # typical student calculation (keep your existing code, then pass to template)
    try:
        df = pd.read_csv(BASE.joinpath("..", "data", "student_habits_performance.csv"))
        means = []
        for mf in model_feature_order:
            if mf in df.columns:
                means.append(float(df[mf].mean()))
            else:
                means.append(0.0)
        Xm = np.array(means).reshape(1, -1)
        Xm_scaled = scaler.transform(Xm)
        cls_typical, prob_typical = ensemble_classify(Xm_scaled)
        reg_typical, _ = ensemble_regress(Xm_scaled)
    except Exception:
        cls_typical, prob_typical, reg_typical = None, None, None

    return render_template("dashboard.html",
                           clf_names=clf_names,
                           reg_names=reg_names,
                           typical={"pass": cls_typical, "prob": prob_typical, "score": reg_typical},
                           eda_images=eda_images,
                           eda_error=eda_error)


# API
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.json
        X_raw = prepare_features(payload)
        X_scaled = scaler.transform(X_raw)
        cls, prob = ensemble_classify(X_scaled)
        reg_pred, reg_preds = ensemble_regress(X_scaled)
        return jsonify({
            "pass_fail": int(cls),
            "pass_probability": float(prob),
            "predicted_score": float(reg_pred)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
