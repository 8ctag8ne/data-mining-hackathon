import pandas as pd
import numpy as np

# -------------------------
# 1. Завантаження валідного датасету
# -------------------------
df = pd.read_csv("churn_na.csv")

# Перетворення дат у datetime (вже після валідації)
df["install_time"] = pd.to_datetime(df["install_time"])
df["event_time"] = pd.to_datetime(df["event_time"])
df["event_date"] = pd.to_datetime(df["event_date"])

# Якщо event_properties — JSON у вигляді рядка
import json


def parse_json_safe(x):
    if pd.isna(x) or x == "NA":
        return {}
    try:
        return json.loads(x)
    except:
        return {}


df["event_props"] = df["event_properties"].apply(parse_json_safe)


# -------------------------
# 2. Корисні допоміжні функції
# -------------------------
def safe_count(series, value):
    return (series == value).sum()


def count_events_containing(series, prefix):
    return series.str.endswith(prefix, na=False).sum()


# -------------------------
# 3. Feature engineering по користувачу
# -------------------------

features = df.groupby("user_id").apply(lambda g: pd.Series({

    # --- A. Onboarding ---
    "onboarding_skips": safe_count(g["event_type"], "quiz_question_onboarding_skip_tap"),
    "quiz_answers": g["event_props"].apply(lambda d: 1 if "answers" in d else 0).sum(),

    # --- B. Engagement ---
    "total_events": len(g),

    # session-like intensity: середня різниця між подіями
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
    # ВИПРАВЛЕНО: error_rate = помилки / відправлені повідомлення
    "error_rate": (
        safe_count(g["event_type"], "answer_error") / safe_count(g["event_type"], "send_message_tap")
        if safe_count(g["event_type"], "send_message_tap") > 0
        else 0
    ),


})).reset_index()

# -------------------------
# 4. Зберегти результат
# -------------------------

features.to_csv("na_user_features.csv", index=False)

print("Створено user_features.csv з", len(features), "користувачами.")
print("\nПриклади фіч:")
print(features.head())

