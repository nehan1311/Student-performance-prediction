import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load everything
df = pd.read_csv("../data/student_habits_performance.csv")

feature_cols = [
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "attendance_percentage",
    "sleep_hours"
]

X = df[feature_cols]

scaler = joblib.load("./models/scaler.pkl")
X_scaled = scaler.transform(X)

kmeans = joblib.load("./models/kmeans_clusters.pkl")

# Cluster centers (scaled)
scaled_centers = kmeans.cluster_centers_

# Convert centers to original scale for understanding
centers_original = scaler.inverse_transform(scaled_centers)

centers_df = pd.DataFrame(
    centers_original,
    columns=feature_cols
)

print("\n=== Cluster Centers (Original Values) ===\n")
print(centers_df)
print("\n")

# Add cluster ID column
centers_df.insert(0, 'Cluster', range(len(centers_df)))

# Save to file
centers_df.to_csv("./models/cluster_centers.csv", index=False)
print("✅ Cluster centers saved to models/cluster_centers.csv")

# Print interpretation guide
print("\n=== Cluster Interpretation Guide ===\n")
print("Analyze the cluster centers above to determine:")
print("- Cluster 0: Check values for study hours, social media, attendance")
print("- Cluster 1: Check values for study hours, social media, attendance")
print("- Cluster 2: Check values for study hours, social media, attendance")
print("\nBased on the values, determine which cluster represents:")
print("- High-risk (low study, high distractions, low attendance)")
print("- Moderate (medium values)")
print("- Good habits (high study, low distractions, high attendance)")

