import pandas as pd

# Завантаження CSV
df = pd.read_csv("user_features.csv")

# Загальна кількість користувачів
total = len(df)

# Групування за комбінаціями
combo = (
    df.groupby(["is_churned", "is_advanced"])
      .size()
      .reset_index(name="count")
)

# Додавання відсотків
combo["percent"] = combo["count"] / total * 100

print(combo)