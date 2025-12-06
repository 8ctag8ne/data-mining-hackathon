import json
from collections import defaultdict

import pandas as pd
def parse_json(s):
    if pd.isna(s) or s == "":
        return {}
    try:
        return json.loads(s)
    except:
        # sometimes CSV escapes double quotes badly — fallback
        try:
            return json.loads(s.replace("''", '"').replace('""', '"'))
        except:
            return {}

# ---- Завантаження датасету ----
df = pd.read_csv(
    "valid.csv",
    parse_dates=["install_date", "install_time", "event_date", "event_time"],
    dtype={
        "user_id": "Int64",
        "event_counter": "Int64",
        "event_type": "string",
        "event_properties": "string",
        "is_churned": "Int64",
    }
)

# Привести дати/часи до timezone-aware (ваш CSV містить +00:00)
df["install_time"] = pd.to_datetime(df["install_time"], utc=True)
df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

# Розпарсити JSON-рядки
df["event_properties_parsed"] = df["event_properties"].apply(parse_json)

# Приклад доступу до вкладених полів
df["screen_name"] = df["event_properties_parsed"].apply(lambda x: x.get("screen_name"))
df["answers"] = df["event_properties_parsed"].apply(lambda x: x.get("answers"))

answers_dict = dict()

for row in df[["screen_name", "answers"]].to_dict("records"):
    answers = row.get("answers")
    screen = row.get("screen_name")

    if not answers or not screen:
        continue
    answers = answers.strip()

    # сплітинг
    if "," in answers:
        answers = answers.split(",")
    elif " " in answers:
        answers = answers.split(" ")

    if  not isinstance(answers, list):
        answers = [answers]


    answers = [ans.strip() for ans in answers]

    # ініціалізація словника
    if screen not in answers_dict:
        answers_dict[screen] = {}

    # підрахунок
    for ans in answers:
        if len(ans) > 0:
            answers_dict[screen][ans] = answers_dict[screen].get(ans, 0) + 1

for screen in answers_dict.keys():
    print(screen)
    for ans in answers_dict[screen]:
        print(f"    {ans}: {answers_dict[screen][ans]}")