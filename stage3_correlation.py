import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------------
# Завантаження фіч
# -------------------------------------
df = pd.read_csv("user_features.csv")

# Перелік фіч (усі, крім user_id і is_churned)
feature_cols = [
    col for col in df.columns
    if col not in ["user_id", "is_churned"]
]

results = []

def optimal_bins_fd(series, min_bins=3, max_bins=10):
    """Обчислює оптимальну кількість бінів за правилом Freedman–Diaconis."""
    data = series.dropna()

    if len(data) < 5:
        return min_bins  # замало даних для обчислення

    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25

    if iqr == 0:
        return min_bins  # всі значення однакові / занадто скупчено

    bin_width = 2 * iqr / (len(data) ** (1/3))
    if bin_width <= 0:
        return min_bins

    k = int((data.max() - data.min()) / bin_width)

    return max(min_bins, min(k, max_bins))

def compute_churn_bins(df, col):
    """Автоматичний підбір кількості бінів + обчислення churn rate."""
    data = df[col]

    bins = optimal_bins_fd(data)

    try:
        df["_bin"] = pd.qcut(data, bins, duplicates="drop")
    except ValueError:
        return None

    grouped = df.groupby("_bin")["is_churned"].mean()
    df.drop(columns="_bin", inplace=True)
    return grouped

# -------------------------------------
# Розрахунок
# -------------------------------------
for col in feature_cols:
    data = df[col]

    # Пропускаємо non-numeric фічі
    if data.dtype not in [np.float64, np.int64]:
        continue

    # Кореляція
    try:
        corr = df["is_churned"].corr(data)
    except Exception:
        corr = None

    # Churn rate у бінів
    bins = compute_churn_bins(df, col)

    if bins is not None:
        for interval, churn_rate in bins.items():
            results.append({
                "feature": col,
                "bin": str(interval),
                "churn_rate": churn_rate,
                "correlation": corr
            })
    else:
        results.append({
            "feature": col,
            "bin": "NO_BINS (all values identical or invalid)",
            "churn_rate": None,
            "correlation": corr
        })

# -------------------------------------
# Збереження таблиці
# -------------------------------------
out = pd.DataFrame(results)
out.to_csv("churn_dependency.csv", index=False)

print("\nГотово! Збережено churn_dependency.csv")
print("Приклад результатів:")
print(out.head(20))


# =====================================
#            ВІЗУАЛІЗАЦІЯ
# =====================================

os.makedirs("feature_charts", exist_ok=True)

for feature in out["feature"].unique():

    subset = out[out["feature"] == feature]
    corr_value = subset["correlation"].iloc[0]

    # ---- 1. Barplot кореляції ----
    plt.figure(figsize=(4, 4))
    sns.barplot(x=[corr_value], y=[feature])
    plt.title(f"Correlation with churn: {feature}\n({corr_value:.3f})")
    plt.xlabel("Correlation")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"feature_charts/{feature}_correlation.png")
    plt.close()

    # Пропуск графіка, якщо немає валідних бінів
    if subset["bin"].nunique() <= 1:
        continue

    # ---- 2. Churn-rate по інтервалах ----
    plt.figure(figsize=(8, 4))
    sns.barplot(data=subset, x="bin", y="churn_rate")
    plt.xticks(rotation=45)
    plt.title(f"Churn rate across bins: {feature}")
    plt.tight_layout()
    plt.savefig(f"feature_charts/{feature}_bins.png")
    plt.close()

print("\nГрафіки збережено у папку feature_charts/")



