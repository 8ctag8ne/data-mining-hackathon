import pandas as pd
import numpy as np

# -------------------------
# 1. Завантаження валідного датасету
# -------------------------
df = pd.read_csv("valid.csv")

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

#для роботи з онбордингом
def extract_answers(parsed):
    """Повертає list of answers або пустий list."""
    if not parsed:
        return []
    ans = parsed.get("answers")
    if not ans:
        return []

    # нормалізація
    if isinstance(ans, str):
        # split за комою або пробілом
        if "," in ans:
            ans = [x.strip() for x in ans.split(",")]
        else:
            ans = [x.strip() for x in ans.split()]
    elif isinstance(ans, list):
        ans = [str(x).strip() for x in ans]
    else:
        return []

    # фільтруємо пусті
    return [x for x in ans if x]


df["answers_list"] = df["event_props"].apply(extract_answers)
df["screen_name"] = df["event_props"].apply(lambda x: x.get("screen_name") if x else None)

def collect_user_answers(group):
    """Повертає dict з трьома ключами: goals, interests, assistance."""
    categories = {
        "goals": [],
        "interests": [],
        "assistance": []
    }

    for _, row in group.iterrows():
        screen = row["screen_name"]
        answers = row["answers_list"]

        if screen in categories and answers:
            categories[screen].extend(answers)

    # перетворити на унікальні, відсортовані значення
    return {
        "user_goals": ", ".join(sorted(set(categories["goals"]))),
        "user_interests": ", ".join(sorted(set(categories["interests"]))),
        "user_assistance": ", ".join(sorted(set(categories["assistance"])))
    }


user_answer_features = df.groupby("user_id").apply(collect_user_answers).apply(pd.Series).reset_index()

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

    # --- F. Temporal metrics ---

    "time_to_first_message_sec": (
        (g.loc[g["event_type"] == "send_message_tap", "event_time"].min()
         - g["install_time"].min()).total_seconds()
        if (g["event_type"] == "send_message_tap").any() else np.nan
    ),

    # --- G. Error/Friction ---
    "error_rate": safe_count(g["event_type"], "answer_error") / max(len(g), 1),

    # --- Churn (already validated) ---
    "is_churned": g["is_churned"].iloc[0]

})).reset_index()

features = features.merge(user_answer_features, on="user_id", how="left")

# -------------------------
# 4. Зберегти результат
# -------------------------

features.to_csv("user_features.csv", index=False)

print("Створено user_features.csv з", len(features), "користувачами.")
print("Приклади фіч:\n")
print(features.head())

