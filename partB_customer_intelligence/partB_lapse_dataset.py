
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# =====================================================
# STEP 1: LOAD DATA
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

orders = pd.read_csv(ROOT / "data" / "clean" / "orders.csv")
customers = pd.read_csv(ROOT / "data" / "clean" / "customers.csv")
products = pd.read_csv(ROOT / "data" / "clean" / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

cutoff_date = pd.Timestamp("2025-09-01 00:00:00")

print("=" * 60)
print("STEP 1: DATA LOADING")
print("=" * 60)

print("Orders shape:", orders.shape)
print("Customers shape:", customers.shape)
print("Products shape:", products.shape)
print("Cutoff date:", cutoff_date)


# =====================================================
# STEP 2: DEFINE CUSTOMER COHORT
# =====================================================

historical_orders = orders[
    (orders["order_ts"] < cutoff_date)
    & (orders["status"].str.lower() == "completed")
].copy()

cohort_ids = historical_orders["customer_id"].unique()

cohort = customers[
    customers["customer_id"].isin(cohort_ids)
].copy()

print("\n" + "=" * 60)
print("STEP 2: CUSTOMER COHORT")
print("=" * 60)

print("Cohort size:", len(cohort))


# =====================================================
# STEP 3: DEFINE LAPSE LABEL
# =====================================================

september_orders = orders[
    (orders["order_ts"] >= cutoff_date)
    & (orders["order_ts"] < pd.Timestamp("2025-10-01"))
    & (orders["status"].str.lower() == "completed")
]

september_customer_ids = september_orders["customer_id"].unique()

cohort["lapsed"] = (
    ~cohort["customer_id"].isin(september_customer_ids)
).astype(int)

print("\nLapse distribution:")
print(cohort["lapsed"].value_counts())

print(
    "Lapse percentage:",
    round(cohort["lapsed"].mean() * 100, 2),
    "%"
)


# =====================================================
# STEP 4: CALCULATE RFM FEATURES
# =====================================================

rfm = historical_orders.groupby("customer_id").agg(
    last_order_date=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

rfm["recency"] = (
    cutoff_date - rfm["last_order_date"]
).dt.days

discount = historical_orders.groupby("customer_id").agg(
    avg_discount_pct=("discount_pct", "mean")
).reset_index()

features = rfm.merge(
    discount,
    on="customer_id",
    how="left"
)


# =====================================================
# STEP 5: CALCULATE CANCELLATION RATE
# =====================================================

final_orders = orders[
    (orders["order_ts"] < cutoff_date)
    & (
        orders["status"].str.lower().isin(
            ["completed", "cancelled"]
        )
    )
].copy()

final_orders["status"] = final_orders["status"].str.lower()

cancellation = final_orders.groupby("customer_id").agg(
    total_final_orders=("order_id", "nunique"),
    cancelled_orders=(
        "status",
        lambda x: (x == "cancelled").sum()
    )
).reset_index()

cancellation["cancellation_rate"] = (
    cancellation["cancelled_orders"]
    / cancellation["total_final_orders"]
)

features = features.merge(
    cancellation[["customer_id", "cancellation_rate"]],
    on="customer_id",
    how="left"
)


# =====================================================
# STEP 6: CALCULATE AVERAGE DELIVERY TIME
# =====================================================

delivery = historical_orders.groupby("customer_id").agg(
    avg_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

features = features.merge(
    delivery,
    on="customer_id",
    how="left"
)


# =====================================================
# STEP 7: FIND FAVOURITE PRODUCT CATEGORY
# =====================================================

category_orders = historical_orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

category_counts = (
    category_orders.groupby(["customer_id", "category"])
    .size()
    .reset_index(name="category_count")
)

favorite_category = (
    category_counts.sort_values(
        ["customer_id", "category_count", "category"],
        ascending=[True, False, True]
    )
    .drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favorite_category"})
)

features = features.merge(
    favorite_category,
    on="customer_id",
    how="left"
)


# =====================================================
# STEP 8: ADD CUSTOMER DETAILS AND TENURE
# =====================================================

customer_details = cohort[
    ["customer_id", "city", "age", "gender", "signup_date"]
].copy()

customer_details["tenure_days"] = (
    cutoff_date - customer_details["signup_date"]
).dt.days

features = features.merge(
    customer_details[
        ["customer_id", "city", "age", "gender", "tenure_days"]
    ],
    on="customer_id",
    how="left"
)


# =====================================================
# STEP 9: ADD LAPSE LABEL
# =====================================================

features = features.merge(
    cohort[["customer_id", "lapsed"]],
    on="customer_id",
    how="left"
)

print("\n" + "=" * 60)
print("STEP 9: FINAL FEATURE DATASET")
print("=" * 60)

print("Dataset shape:", features.shape)
print("\nFirst 5 rows:")
print(features.head())

print("\nMissing values:")
print(features.isnull().sum())

print("\nLapse distribution:")
print(features["lapsed"].value_counts())


# =====================================================
# STEP 10: SAVE DATASET
# =====================================================

output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

features.to_csv(
    output_dir / "lapse_dataset.csv",
    index=False
)

print("\nSaved: outputs/lapse_dataset.csv")


# =====================================================
# STEP 11: TRAIN TEST SPLIT
# =====================================================

y = features["lapsed"]

X = features.drop(
    columns=[
        "customer_id",
        "last_order_date",
        "lapsed"
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\n" + "=" * 60)
print("STEP 11: TRAIN TEST SPLIT")
print("=" * 60)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# =====================================================
# STEP 12: PREPROCESSING
# =====================================================

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "favorite_category",
    "city",
    "gender"
]

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using training rules
X_test_processed = preprocessor.transform(X_test)

print("\n" + "=" * 60)
print("STEP 12: PREPROCESSING")
print("=" * 60)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)


# =====================================================
# STEP 13: VERIFY SCALER
# =====================================================

frequency_index = numeric_features.index("frequency")

full_frequency_mean = features["frequency"].mean()

training_frequency_mean = (
    preprocessor.named_transformers_["num"].mean_[frequency_index]
)

print("\nFull cohort frequency mean:")
print(full_frequency_mean)

print("\nTraining frequency mean learned by scaler:")
print(training_frequency_mean)


# =====================================================
# STEP 14: FINAL VALIDATION
# =====================================================

assert len(features) == 1156
assert len(X_train) == 924
assert len(X_test) == 232

assert features["lapsed"].sum() == 443

assert not features.isnull().any().any()

assert "lapsed" not in X_train.columns
assert "customer_id" not in X_train.columns
assert "last_order_date" not in X_train.columns

assert set(X_train.columns) == set(
    numeric_features + categorical_features
)

print("\n" + "=" * 60)
print("ALL DATASET AND PREPROCESSING CHECKS PASSED!")
print("=" * 60)

# =====================================================
# STEP 15: BASELINE MODEL
# =====================================================

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Create baseline model
baseline_model = DummyClassifier(
    strategy="most_frequent"
)

# Train baseline model
baseline_model.fit(X_train_processed, y_train)

# Predict on test data
baseline_predictions = baseline_model.predict(X_test_processed)

# Evaluate baseline
print("\n" + "=" * 60)
print("STEP 15: BASELINE MODEL")
print("=" * 60)

print(
    "Accuracy:",
    round(accuracy_score(y_test, baseline_predictions), 4)
)

print(
    "Lapsed precision:",
    round(
        precision_score(
            y_test,
            baseline_predictions,
            pos_label=1,
            zero_division=0
        ),
        4
    )
)

print(
    "Lapsed recall:",
    round(
        recall_score(
            y_test,
            baseline_predictions,
            pos_label=1,
            zero_division=0
        ),
        4
    )
)

print(
    "Lapsed F1:",
    round(
        f1_score(
            y_test,
            baseline_predictions,
            pos_label=1,
            zero_division=0
        ),
        4
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, baseline_predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        baseline_predictions,
        zero_division=0
    )
)

# STEP 16: LOGISTIC REGRESSION WITH GRID SEARCH

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

print("\n" + "=" * 60)
print("STEP 16: LOGISTIC REGRESSION")
print("=" * 60)

# Create the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)

# Define the values of C to test
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100]
}

# Perform 5-fold cross-validation
logistic_grid = GridSearchCV(
    estimator=logistic_model,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# Train using training data only
logistic_grid.fit(X_train_processed, y_train)

# Get the best model
best_logistic_model = logistic_grid.best_estimator_

# Predict on the unchanged test data
logistic_predictions = best_logistic_model.predict(X_test_processed)

# Evaluate the model
print("Best Parameters:", logistic_grid.best_params_)
print("Best Cross-Validation F1:", round(logistic_grid.best_score_, 4))

print("\nTest Accuracy:",round(accuracy_score(y_test, logistic_predictions), 4))

print("Lapsed Precision:",round(precision_score(y_test, logistic_predictions,pos_label=1, zero_division=0), 4))

print("Lapsed Recall:",round(recall_score(y_test, logistic_predictions,pos_label=1, zero_division=0), 4))

print("Lapsed F1:",round(f1_score(y_test, logistic_predictions,pos_label=1, zero_division=0), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))

# STEP 17: RANDOM FOREST WITH GRID SEARCH

from sklearn.ensemble import RandomForestClassifier

print("\n" + "=" * 60)
print("STEP 17: RANDOM FOREST")
print("=" * 60)

# Create the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Define parameter combinations to test
rf_param_grid = {
    "max_depth": [3, 5, 8],
    "min_samples_leaf": [5, 10, 20]
}

# Perform 5-fold cross-validation
rf_grid = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# Train using training data only
rf_grid.fit(X_train_processed, y_train)

# Get the best Random Forest model
best_rf_model = rf_grid.best_estimator_

# Predict on unchanged test data
rf_predictions = best_rf_model.predict(X_test_processed)

# Evaluate the model
print("Best Parameters:", rf_grid.best_params_)
print("Best Cross-Validation F1:",round(rf_grid.best_score_, 4))

print("\nTest Accuracy:",round(accuracy_score(y_test, rf_predictions), 4))

print("Lapsed Precision:",round(precision_score(y_test, rf_predictions,pos_label=1, zero_division=0), 4))

print("Lapsed Recall:",round(recall_score(y_test, rf_predictions,pos_label=1, zero_division=0), 4))

print("Lapsed F1:",round(f1_score(y_test, rf_predictions,pos_label=1, zero_division=0), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
# =====================================================
# STEP 18: OVERSAMPLING THE TRAINING DATA
# =====================================================

from sklearn.utils import resample
import pandas as pd
from scipy.sparse import issparse

print("\n" + "=" * 60)
print("STEP 18: OVERSAMPLING")
print("=" * 60)

# Convert sparse matrix to a regular DataFrame
if issparse(X_train_processed):
    train_features = pd.DataFrame(X_train_processed.toarray())
else:
    train_features = pd.DataFrame(X_train_processed)

# Add target column
train_features["lapsed"] = y_train.reset_index(drop=True)

print("Before oversampling:")
print(train_features["lapsed"].value_counts())

# Separate majority and minority classes
majority = train_features[train_features["lapsed"] == 0]
minority = train_features[train_features["lapsed"] == 1]

# Oversample the minority class
minority_oversampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

# Combine and shuffle
balanced_train = pd.concat(
    [majority, minority_oversampled],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and target
X_train_balanced = balanced_train.drop(columns=["lapsed"])
y_train_balanced = balanced_train["lapsed"]

print("\nAfter oversampling:")
print(y_train_balanced.value_counts())

# Train Logistic Regression on balanced data
balanced_logistic_model = LogisticRegression(
    C=logistic_grid.best_params_["C"],
    max_iter=1000
)

balanced_logistic_model.fit(
    X_train_balanced,
    y_train_balanced
)

# Evaluate on the ORIGINAL test set
balanced_predictions = balanced_logistic_model.predict(X_test_processed)

print("\nBalanced Logistic Regression Results:")
print("Accuracy:", round(accuracy_score(y_test, balanced_predictions), 4))
print("Lapsed Precision:", round(precision_score(y_test, balanced_predictions, zero_division=0), 4))
print("Lapsed Recall:", round(recall_score(y_test, balanced_predictions, zero_division=0), 4))
print("Lapsed F1:", round(f1_score(y_test, balanced_predictions, zero_division=0), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, balanced_predictions))# =====================================================
# STEP 18: OVERSAMPLING THE TRAINING DATA
# =====================================================

from sklearn.utils import resample
import pandas as pd
from scipy.sparse import issparse

print("\n" + "=" * 60)
print("STEP 18: OVERSAMPLING")
print("=" * 60)

# Convert sparse matrix to a regular DataFrame
if issparse(X_train_processed):
    train_features = pd.DataFrame(X_train_processed.toarray())
else:
    train_features = pd.DataFrame(X_train_processed)

# Add target column
train_features["lapsed"] = y_train.reset_index(drop=True)

print("Before oversampling:")
print(train_features["lapsed"].value_counts())

# Separate majority and minority classes
majority = train_features[train_features["lapsed"] == 0]
minority = train_features[train_features["lapsed"] == 1]

# Oversample the minority class
minority_oversampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

# Combine and shuffle
balanced_train = pd.concat(
    [majority, minority_oversampled],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and target
X_train_balanced = balanced_train.drop(columns=["lapsed"])
y_train_balanced = balanced_train["lapsed"]

print("\nAfter oversampling:")
print(y_train_balanced.value_counts())

# Train Logistic Regression on balanced data
balanced_logistic_model = LogisticRegression(
    C=logistic_grid.best_params_["C"],
    max_iter=1000
)

balanced_logistic_model.fit(
    X_train_balanced,
    y_train_balanced
)

# Evaluate on the ORIGINAL test set
balanced_predictions = balanced_logistic_model.predict(X_test_processed)

print("\nBalanced Logistic Regression Results:")
print("Accuracy:", round(accuracy_score(y_test, balanced_predictions), 4))
print("Lapsed Precision:", round(precision_score(y_test, balanced_predictions, zero_division=0), 4))
print("Lapsed Recall:", round(recall_score(y_test, balanced_predictions, zero_division=0), 4))
print("Lapsed F1:", round(f1_score(y_test, balanced_predictions, zero_division=0), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, balanced_predictions))

# =====================================================
# STEP 19: MODEL INTERPRETATION
# =====================================================

import pandas as pd

print("\n" + "=" * 60)
print("STEP 19: MODEL INTERPRETATION")
print("=" * 60)

# Get feature names from the fitted preprocessing pipeline
feature_names = preprocessor.get_feature_names_out()

# -----------------------------------------
# A. RANDOM FOREST - TOP 5 FEATURE IMPORTANCE
# -----------------------------------------

rf_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": best_rf_model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 5 Random Forest Features:")
print(rf_importance.head(5).to_string(index=False))

# -----------------------------------------
# B. LOGISTIC REGRESSION - TOP 3 COEFFICIENTS
# -----------------------------------------

lr_coefficients = pd.DataFrame({
    "feature": feature_names,
    "coefficient": balanced_logistic_model.coef_[0]
})

# Sort by absolute coefficient magnitude
lr_coefficients["absolute_value"] = (
    lr_coefficients["coefficient"].abs()
)

lr_coefficients = lr_coefficients.sort_values(
    by="absolute_value",
    ascending=False
)

print("\nTop 3 Logistic Regression Coefficients:")
print(
    lr_coefficients[
        ["feature", "coefficient"]
    ].head(3).to_string(index=False)
)

# =====================================================
# STEP 20: SAVE AND VERIFY FINAL MODEL
# =====================================================

import joblib
import numpy as np
from pathlib import Path

print("\n" + "=" * 60)
print("STEP 20: SAVE AND VERIFY MODEL")
print("=" * 60)

# Select the final model
final_model = balanced_logistic_model

# Save model
model_path = Path("outputs/lapse_model.joblib")

joblib.dump(final_model, model_path)

print("Model saved:", model_path)

# Load the saved model
loaded_model = joblib.load(model_path)

# Generate predictions using original final model
original_predictions = final_model.predict(X_test_processed)

# Generate predictions using loaded model
loaded_predictions = loaded_model.predict(X_test_processed)

# Verify both predictions are identical
assert np.array_equal(
    original_predictions,
    loaded_predictions
), "Model verification failed!"

print("Model successfully reloaded.")
print("Original and loaded predictions match.")
print("MODEL VERIFICATION PASSED!")

# =====================================================
# STEP 21: GENERATE PREMIUM CUSTOMER RETENTION LIST
# =====================================================

print("\n" + "=" * 60)
print("STEP 21: PREMIUM CUSTOMER RETENTION LIST")
print("=" * 60)

# Load the original lapse dataset and customer segments
lapse_data = pd.read_csv("outputs/lapse_dataset.csv")
segments = pd.read_csv("outputs/customer_segments.csv")

# Recover the test customers using their original row indices
test_customers = lapse_data.loc[X_test.index].copy()

# Add Premium segment information
test_customers = test_customers.merge(
    segments[["customer_id", "is_premium"]],
    on="customer_id",
    how="left"
)

# Keep only Premium customers
premium_test = test_customers[
    test_customers["is_premium"] == True
].copy()

# Calculate probability of lapse (class 1)
premium_test["lapse_probability"] = (
    final_model.predict_proba(X_test_processed)[:, 1]
    [test_customers["is_premium"].to_numpy()]
)

# Sort by highest lapse probability
retention_list = premium_test.sort_values(
    by="lapse_probability",
    ascending=False
).head(10)

# Keep only the required columns
retention_list = retention_list[
    ["customer_id", "lapse_probability", "lapsed"]
].rename(columns={"lapsed": "actually_lapsed"})

# Save the retention list
retention_list.to_csv(
    "outputs/retention_list.csv",
    index=False
)

print("\nTop Premium Customers at Risk:")
print(retention_list.to_string(index=False))

print("\nNumber of Premium customers selected:",
      len(retention_list))

print("Actually lapsed customers:",
      retention_list["actually_lapsed"].sum())

print("\nSaved: outputs/retention_list.csv")

# Final validation
assert list(retention_list.columns) == [
    "customer_id",
    "lapse_probability",
    "actually_lapsed"
]

assert retention_list["lapse_probability"].between(0, 1).all()

print("\nRETENTION LIST CHECKS PASSED!")
