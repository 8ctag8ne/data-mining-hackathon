import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer

# ==============================================================================
# 1. SETUP AND DATA LOADING
# ==============================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

try:
    df_train = pd.read_csv("user_features.csv")      # Known churn
    df_predict = pd.read_csv("na_user_features.csv") # Unknown churn (NA)
    
    print(f"Training data shape: {df_train.shape}")
    print(f"Prediction data shape: {df_predict.shape}")

except FileNotFoundError:
    print("Error: Could not find input files.")
    exit()

# Define desired features
desired_features = [
    "answer_errors",
    "avg_event_interval_sec",
    "chat_opens",
    "chat_views",
    "dislikes",
    "error_rate",
    "likes",
    "messages_received",
    "messages_sent",
    "total_events",
    "time_to_first_message_sec",
    "onboarding_skips",
    "quiz_answers",
    "model_changes",
    "is_advanced",       # This caused the error
    "successful_purchase"
]

# ==============================================================================
# 2. DATA PATCHING (Fix missing columns)
# ==============================================================================
print("\n" + "=" * 70)
print("CHECKING & FIXING COLUMNS")
print("=" * 70)

def fix_missing_columns(df):
    # 1. Fix 'is_advanced' if missing
    if "is_advanced" not in df.columns:
        if "model_changes" in df.columns:
            print("  -> Creating missing 'is_advanced' from 'model_changes'")
            df["is_advanced"] = (df["model_changes"] > 0).astype(int)
        else:
            print("  -> 'model_changes' also missing. Setting 'is_advanced' to 0")
            df["is_advanced"] = 0
            
    # 2. Fix 'successful_purchase' if missing (critical feature)
    if "successful_purchase" not in df.columns:
        print("  -> 'successful_purchase' missing. Setting to 0 (Warning: Model may be weaker)")
        df["successful_purchase"] = 0
        
    return df

print("Patching Training Data:")
df_train = fix_missing_columns(df_train)

print("Patching Prediction Data:")
df_predict = fix_missing_columns(df_predict)

# Verify one last time which columns are actually available
available_features = [col for col in desired_features if col in df_train.columns]
print(f"\nFinal feature list used ({len(available_features)} features):")
print(available_features)

# ==============================================================================
# 3. PREPROCESSING
# ==============================================================================
print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)

# Extract X and y
X = df_train[available_features].copy()
y = df_train["is_churned"].astype(int)

X_na = df_predict[available_features].copy()

# Impute missing values (e.g., time_to_first_message_sec might be NaN)
imputer = SimpleImputer(strategy='constant', fill_value=-1)

X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=available_features)
X_na_imputed = pd.DataFrame(imputer.transform(X_na), columns=available_features)

print("✓ Missing values imputed (filled with -1)")

# Split into Train/Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================================================================
# 4. MODEL TRAINING
# ==============================================================================
print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("✓ Model trained successfully")

# ==============================================================================
# 5. EVALUATION
# ==============================================================================
print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

y_pred = rf_model.predict(X_val)
y_prob = rf_model.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
roc = roc_auc_score(y_val, y_prob)

print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC:  {roc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig("model_confusion_matrix.png")

# ==============================================================================
# 6. FEATURE IMPORTANCE
# ==============================================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 5 Drivers of Churn:")
for i in range(min(5, len(indices))):
    idx = indices[i]
    print(f"{i+1}. {available_features[idx]} ({importances[idx]:.4f})")

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(len(available_features)), importances[indices], align="center")
plt.xticks(range(len(available_features)), [available_features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig("feature_importance.png")

# ==============================================================================
# 7. PREDICTION & EXPORT
# ==============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Predictions for NA users
na_preds = rf_model.predict(X_na_imputed)
na_probs = rf_model.predict_proba(X_na_imputed)[:, 1]

results_df = df_predict[["user_id"]].copy()
results_df["predicted_is_churned"] = na_preds
results_df["churn_probability"] = na_probs

# Save
output_file = "churn_predictions_for_na_users.csv"
results_df.to_csv(output_file, index=False)

churned_cnt = results_df["predicted_is_churned"].sum()
print(f"Total NA users: {len(results_df)}")
print(f"Classified as CHURNED: {churned_cnt} ({(churned_cnt/len(results_df)*100):.1f}%)")
print(f"Classified as RETAINED: {len(results_df) - churned_cnt}")
print(f"\n✓ Saved predictions to: {output_file}")