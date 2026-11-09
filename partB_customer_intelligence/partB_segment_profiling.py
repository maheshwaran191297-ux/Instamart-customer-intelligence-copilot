
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# File paths
input_path = ROOT / "outputs" / "rfm_table.csv"
output_path = ROOT / "outputs" / "customer_segments.csv"

# Load RFM data
rfm = pd.read_csv(input_path)

# Select RFM features
rfm_features = rfm[["recency", "frequency", "monetary"]]

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(rfm_features)

# Create and train K-Means model
kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = kmeans.fit_predict(scaled_data)

# Calculate segment profiles
segment_profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).round(2)

print("\nCustomer Segment Profiles:")
print(segment_profile)

# Identify the segment with the highest mean monetary value
premium_segment = segment_profile["mean_monetary"].idxmax()

# Assign business names and actions
segment_names = {}
segment_actions = {}

for segment_id, row in segment_profile.iterrows():

    if segment_id == premium_segment:
        name = "Premium Customers"
        action = "Reward loyalty and provide exclusive offers"

    elif row["mean_recency"] <= segment_profile["mean_recency"].median():
        name = "Active Customers"
        action = "Encourage repeat purchases"

    elif row["mean_frequency"] <= segment_profile["mean_frequency"].median():
        name = "At-Risk Customers"
        action = "Send personalized re-engagement offers"

    else:
        name = "Regular Customers"
        action = "Recommend relevant products and bundles"

    segment_names[segment_id] = name
    segment_actions[segment_id] = action

rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"] = (rfm["segment_id"] == premium_segment)

# Save final customer segments
rfm.to_csv(output_path, index=False)

print("\nSegment Names and Recommended Actions:")

for segment_id in segment_profile.index:
    print(
        f"\nSegment {segment_id}: {segment_names[segment_id]}"
        f"\nAction: {segment_actions[segment_id]}"
    )

print("\nPremium Segment ID:", premium_segment)
print("\nCustomer segments saved to:", output_path)
# PCA analysis
pca = PCA(n_components=2)

pca_data = pca.fit_transform(scaled_data)

explained_variance = pca.explained_variance_ratio_

print("\nPCA Explained Variance:")
print("PC1:", round(explained_variance[0], 3))
print("PC2:", round(explained_variance[1], 3))
print(
    "Total explained variance:",
    round(explained_variance.sum(), 3)
)