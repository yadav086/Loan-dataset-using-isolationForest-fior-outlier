import category_encoders as ce
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, TargetEncoder

# ==============================================================================
# 1. DATA LOADING & INGESTION
# ==============================================================================

# Read the raw source data file
df = pd.read_csv("loan_data.csv")

# Identify numerical column names, explicitly excluding the target column
# Note: Fix this to df.select_dtypes(...).drop(columns=['loan_status']).columns.to_list() if it crashes!
df_num = (
    df.select_dtypes(include="number")
    .drop("loan_status", axis=1)
    .columns.to_list()
)

# Identify non-numerical/categorical column names
df_cat = df.select_dtypes(exclude="number").columns.to_list()

# ==============================================================================
# 2. FEATURE-TARGET SPLITTING & TRAIN-TEST SPLIT
# ==============================================================================

# Reconstruct features DataFrame combining numerical and categorical columns
X = df[df_num + df_cat]

# Separate the target variable
y = df["loan_status"]

# Split data into training (70%) and testing (30%) subsets with reproducible state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# ==============================================================================
# 3. PREPROCESSING PIPELINE CONFIGURATION
# ==============================================================================

# Numerical pipeline: Apply Yeo-Johnson transformation to stabilize variance and normalize data
tranform_num_col = Pipeline(
    steps=[("Power", PowerTransformer(method="yeo-johnson"))]
)

# Categorical pipeline: Convert text labels into multi-column binary dummy indicators
tranform_cat_col = Pipeline(
    steps=[
        (
            "One_hot",
            OneHotEncoder(
                drop="first", handle_unknown="ignore", sparse_output=False
            ),
        )
    ]
)
# Alternative Option: Target encoding (commented out)
# tranform_cat_col = Pipeline(steps =[('Target', ce.TargetEncoder())])

# Bundle numerical and categorical transformations targeting specific column lists
preprocess = ColumnTransformer(
    transformers=[
        ("num", tranform_num_col, df_num),
        ("cat", tranform_cat_col, df_cat),
    ],
    remainder="drop",
    n_jobs=-1,
    verbose=True,
)

# ==============================================================================
# 4. MASTER MACHINE LEARNING PIPELINE
# ==============================================================================

# Define sequential execution: Preprocess -> Handle Class Imbalance -> Classify
model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        # SMOTEENN: Synthesizes minority samples and cleans noisy boundaries using Edited Nearest Neighbors
        ("smot", SMOTEENN(n_jobs=-1, random_state=42)),
        # Balanced Random Forest using cost-sensitive learning weights and an impurity split threshold
        (
            "Rf",
            RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True,
                max_features=6,
                class_weight="balanced",
                ccp_alpha=0.0001,
                criterion="gini",
            ),
        ),
    ]
)

# ==============================================================================
# 5. OUTLIER DETECTION (ISOLATION FOREST)
# ==============================================================================

# Fit processing rules on training data and transform it into a raw NumPy array
X_train_processed = preprocess.fit_transform(X_train)

# Transform test data using structural distributions learned from training data
X_test_processed = preprocess.transform(X_test)

# Instantiate anomaly detector targeting the most extreme 1% of data points
iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)

# Generate inlier (1) vs outlier (-1) predictions for both data subsets
train_label = iso.fit_predict(X_train_processed)
test_label = iso.predict(X_test_processed)

# Create boolean masks to separate clean rows from anomalies
mask = train_label == 1
mask_outlier = train_label == -1

mask_test = test_label == 1
mask_test_outlier = test_label == -1

# Filter original training data into clean and outlier subsets using masks
X_train_clean = X_train.loc[mask]
y_train_clean = y_train.loc[mask]

X_train_outlier = X_train.loc[mask_outlier]
y_train_outlier = y_train.loc[mask_outlier]

# Filter original testing data into clean and outlier subsets using masks
X_test_clean = X_test.loc[mask_test]
y_test_clean = y_test.loc[mask_test]

X_test_outlier = X_test.loc[mask_test_outlier]
y_test_outlier = y_test.loc[mask_test_outlier]

# ==============================================================================
# 6. MODEL TRAINING & PRODUCTION EVALUATION
# ==============================================================================

# Train the master pipeline (re-running transformers & SMOTEENN) on clean training data
model.fit(X_train_clean, y_train_clean)

# Generate hard classifications and positive class probabilities on uncleaned test data
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Display hold-out validation metrics
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Display Receiver Operating Characteristic (ROC) curves visually
RocCurveDisplay.from_predictions(y_test, y_pred_prob)

# ==============================================================================
# 7. TRAINING CROSS-VALIDATION
# ==============================================================================

# Calculate out-of-fold predictions to evaluate training stability across 10 slices
y_pred = cross_val_predict(
    model, X_train_clean, y_train_clean, cv=10, n_jobs=-1, verbose=True
)

# Extract and print raw confusion matrices based on training cross-validation
cm = confusion_matrix(y_train_clean, y_pred)
print(cm)

# ==============================================================================
# 8. FEATURE IMPORTANCE VISUALIZATION
# ==============================================================================

# Pull generated feature names (including one-hot variants) from transformer step
feature = model.named_steps["preprocess"].get_feature_names_out()

# Pull numeric importance weights assigned by the final Random Forest step
importance = model.named_steps["Rf"].feature_importances_

# Map extracted components into an evaluation DataFrame
df_top_feature = pd.DataFrame({"Feature": feature, "importance": importance})

# Isolate the first 6 records, sort them descending by weight, and index by text labels
# Warning: Using .head(6) BEFORE sorting takes the first 6 random items.
# Swap to .sort_values().head(6) if you want the actual top 6 features!
df_plot = (
    df_top_feature.head(6)
    .sort_values(ascending=False, by="importance")
    .set_index("Feature")
)

# Setup structural plot coordinates and sizes
plt.figure(figsize=(12, 6))
df_plot.plot(kind="bar", ax=plt.gca(), color="skyblue", edgecolor="black")

# Apply clean labels, title, and adjust angles to read feature names clearly
plt.title("Random Forest Feature Importances", fontsize=14, pad=15)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Importance Score", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Render out final graphic windows onto screen
plt.show()
