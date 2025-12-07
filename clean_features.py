# clean_user_features.py
import pandas as pd
import numpy as np

INPUT_FILE = "user_features.csv"
OUTPUT_FILE = "user_features_cleaned.csv"

# -------------------------
# 1) Завантаження
# -------------------------
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {INPUT_FILE}")

# -------------------------
# 2) Видалення стовпців з однаковими значеннями
# -------------------------
n_before = len(df.columns)
cols_constant = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
df.drop(columns=cols_constant, inplace=True)
print(f"Removed {len(cols_constant)} constant columns: {cols_constant}")

# -------------------------
# 3) Видалення дублікатів стовпців (точні або пропорційні)
# -------------------------
def find_duplicate_columns(df):
    """Повертає список колонок, які дублюють інші або пропорційні їм."""
    duplicates = set()
    cols = df.columns.tolist()
    for i in range(len(cols)):
        if cols[i] in duplicates:
            continue
        col_i = df[cols[i]].astype(float)
        for j in range(i+1, len(cols)):
            if cols[j] in duplicates:
                continue
            col_j = df[cols[j]].astype(float)
            # Перевірка точного дублю
            if col_i.equals(col_j):
                duplicates.add(cols[j])
            else:
                # Перевірка пропорційності: x = a*y
                if col_j.abs().max() == 0 and col_i.abs().max() == 0:
                    # обидва стовпці нулі
                    duplicates.add(cols[j])
                elif (col_i != 0).any() and (col_j != 0).any():
                    ratio = col_i / col_j
                    if ratio.dropna().nunique() == 1:
                        duplicates.add(cols[j])
    return list(duplicates)

cols_duplicates = find_duplicate_columns(df)
df.drop(columns=cols_duplicates, inplace=True)
print(f"Removed {len(cols_duplicates)} duplicate/proportional columns: {cols_duplicates}")

# -------------------------
# 4) Збереження
# -------------------------
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved cleaned dataframe with {len(df.columns)} columns to {OUTPUT_FILE}")
