import pandas as pd
import numpy as np

df = pd.read_csv("valid.csv")

df["install_time"] = pd.to_datetime(df["install_time"])
df["event_time"] = pd.to_datetime(df["event_time"])
df["event_date"] = pd.to_datetime(df["event_date"])

import json


def parse_json_safe(x):
    if pd.isna(x) or x == "NA":
        return {}
    try:
        return json.loads(x)
    except:
        return {}


df["event_props"] = df["event_properties"].apply(parse_json_safe)


def safe_count(series, value):
    return (series == value).sum()


def count_events_containing(series, prefix):
    return series.str.endswith(prefix, na=False).sum()



features = df.groupby("user_id").apply(lambda g: pd.Series({

    "onboarding_skips": safe_count(g["event_type"], "quiz_question_onboarding_skip_tap"),
    "quiz_answers": g["event_props"].apply(lambda d: 1 if "answers" in d else 0).sum(),

    "total_events": len(g),

    "avg_event_interval_sec": g["event_time"].sort_values().diff().dt.total_seconds().dropna().mean(),

    # --- C. Chat behavior ---
    "chat_opens": safe_count(g["event_type"], "open_chat_tap"),
    "chat_views": safe_count(g["event_type"], "chat_screen_view"),
    "messages_sent": safe_count(g["event_type"], "send_message_tap"),
    "messages_received": safe_count(g["event_type"], "answer_received"),
    "answer_errors": safe_count(g["event_type"], "answer_error"),
    "likes": safe_count(g["event_type"], "answer_like_tap"),
    "dislikes": safe_count(g["event_type"], "answer_dislike_tap"),

    # --- D. Paywall ---
    "successful_purchase": int("sale_confirmation_success" in g["event_type"].values),

    # --- E. Model usage ---
    "model_changes": safe_count(g["event_type"], "advanced_model_tap"),

    # NEW: Чи використовував користувач advanced модель
    "is_advanced": int(safe_count(g["event_type"], "advanced_model_tap") > 0),

    # --- F. Temporal metrics ---

    "time_to_first_message_sec": (
        (g.loc[g["event_type"] == "send_message_tap", "event_time"].min()
         - g["install_time"].min()).total_seconds()
        if (g["event_type"] == "send_message_tap").any() else np.nan
    ),

    # --- G. Error/Friction ---
    # error_rate = помилки / відправлені повідомлення
    "error_rate": (
        safe_count(g["event_type"], "answer_error") / safe_count(g["event_type"], "send_message_tap")
        if safe_count(g["event_type"], "send_message_tap") > 0
        else 0
    ),

    # --- H. Satisfaction metrics ---
    # NEW: like/dislike ratio
    "like_dislike_ratio": (
        safe_count(g["event_type"], "answer_like_tap") / safe_count(g["event_type"], "answer_dislike_tap")
        if safe_count(g["event_type"], "answer_dislike_tap") > 0
        else (
            float('inf') if safe_count(g["event_type"], "answer_like_tap") > 0
            else 0  # Якщо немає ні likes, ні dislikes
        )
    ),

    # --- Churn (already validated) ---
    "is_churned": g["is_churned"].iloc[0]

})).reset_index()

# -------------------------
# 4. Обробка inf значень у like_dislike_ratio
# -------------------------
# Замінюємо inf на максимальне значення + 1 (користувачі з тільки likes)
max_finite_ratio = features.loc[features["like_dislike_ratio"] != float('inf'), "like_dislike_ratio"].max()

if pd.notna(max_finite_ratio):
    replacement_value = max_finite_ratio + 1
else:
    replacement_value = 100  # Якщо всі значення inf

features["like_dislike_ratio"] = features["like_dislike_ratio"].replace(float('inf'), replacement_value)

print(f"\n📊 Обробка like_dislike_ratio:")
print(f"   • Inf значень замінено на: {replacement_value}")
print(f"   • Діапазон значень: [{features['like_dislike_ratio'].min():.2f}, {features['like_dislike_ratio'].max():.2f}]")

# -------------------------
# 5. Зберегти результат
# -------------------------

features.to_csv("user_features.csv", index=False)

print(f"\n✓ Створено user_features.csv з {len(features)} користувачами.")
print("\n📋 Статистика по like_dislike_ratio:")
print(f"   • Середнє: {features['like_dislike_ratio'].mean():.2f}")
print(f"   • Медіана: {features['like_dislike_ratio'].median():.2f}")
print(f"   • Std: {features['like_dislike_ratio'].std():.2f}")

# Розподіл по категоріях
zero_ratio = (features['like_dislike_ratio'] == 0).sum()
low_ratio = ((features['like_dislike_ratio'] > 0) & (features['like_dislike_ratio'] < 1)).sum()
equal_ratio = (features['like_dislike_ratio'] == 1).sum()
high_ratio = ((features['like_dislike_ratio'] > 1) & (features['like_dislike_ratio'] < replacement_value)).sum()
only_likes = (features['like_dislike_ratio'] == replacement_value).sum()

print(f"\n📊 Розподіл користувачів:")
print(f"   • Немає реакцій (0): {zero_ratio} ({zero_ratio/len(features)*100:.1f}%)")
print(f"   • Більше dislikes (<1): {low_ratio} ({low_ratio/len(features)*100:.1f}%)")
print(f"   • Порівну (=1): {equal_ratio} ({equal_ratio/len(features)*100:.1f}%)")
print(f"   • Більше likes (>1): {high_ratio} ({high_ratio/len(features)*100:.1f}%)")
print(f"   • Тільки likes: {only_likes} ({only_likes/len(features)*100:.1f}%)")

print("\n📋 Приклади фіч:")
print(features[['user_id', 'likes', 'dislikes', 'like_dislike_ratio', 'error_rate', 'is_churned']].head(10))

