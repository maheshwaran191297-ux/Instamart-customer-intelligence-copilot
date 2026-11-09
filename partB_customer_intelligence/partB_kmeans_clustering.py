
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# Input and output paths
input_path = ROOT / "outputs" / "rfm_table.csv"
output_path = ROOT / "outputs" / "kmeans_evaluation.csv"

# Load RFM data
rfm = pd.read_csv(input_path)

# Select the three RFM columns
rfm_features = rfm[["recency", "frequency", "monetary"]]

# Standardize the RFM values (equivalent to StandardScaler's default behavior)
feature_std = rfm_features.std(ddof=0).replace(0, 1)
scaled_data = ((rfm_features - rfm_features.mean()) / feature_std).to_numpy()

# Store evaluation results
results = []

# Test K values from 2 to 8
for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(scaled_data)

    inertia = model.inertia_
    silhouette = silhouette_score(scaled_data, labels)

    results.append({
        "K": k,
        "Inertia": inertia,
        "Silhouette Score": silhouette
    })

# Convert results into a DataFrame
results_df = pd.DataFrame(results)

# Display results
print("\nK-Means Evaluation Results:")
print(results_df.round(3).to_string(index=False))

# Save results
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)

print("\nEvaluation results saved successfully!")
