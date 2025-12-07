import csv
import json
from datetime import datetime, timedelta, timezone

INPUT_FILE = "input.csv"
VALID_FILE = "valid.csv"
INVALID_FILE = "invalid.csv"
CHURN_NA_FILE = "churn_na.csv"
TRANSFORMED_FILE = "transformed.csv"

MIN_DATE = datetime(2025, 1, 1)
MAX_DATE = datetime(2025, 12, 4)

ALLOWED_KEYS = {"screen_name", "model_type", "answers"}

ALLOWED_EVENT_TYPES = {
    "quiz_question_onboarding_screen_view",
    "quiz_question_onboarding_skip_tap",
    "quiz_question_continue_tap",
    "subscription_screen_view",
    "sale_confirmation_success",
    "main_screen_view",
    "advanced_model_tap",
    "chat_screen_view",
    "open_chat_tap",
    "send_message_tap",
    "answer_received",
    "answer_error",
    "answer_like_tap",
    "answer_dislike_tap",
    "history_chat_tap",
    "history_close_tap",
    "history_screen_view",
    "history_tap",
}


def parse_datetime_with_offset(value):
    try:
        dt_part = value[:-6]
        offset_part = value[-6:]

        dt = datetime.strptime(dt_part, "%Y-%m-%d %H:%M:%S")

        sign = 1 if offset_part[0] == "+" else -1
        hours = int(offset_part[1:3])
        minutes = int(offset_part[4:6])
        offset = timezone(sign * timedelta(hours=hours, minutes=minutes))

        return dt.replace(tzinfo=offset)
    except Exception:
        return None


def is_valid_date(date_str):
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return MIN_DATE <= d <= MAX_DATE
    except:
        return False


def validate_event_properties(value):
    if value == "NA":
        return True

    try:
        obj = json.loads(value)
        if not isinstance(obj, dict):
            return False
        return set(obj.keys()).issubset(ALLOWED_KEYS)
    except:
        return False


def is_allowed_event_type(event_type):
    if event_type in ALLOWED_EVENT_TYPES:
        return True
    if event_type.endswith("_onboarding_screen_view"):
        return True
    if event_type.endswith("_onboarding_continue_tap"):
        return True
    return False


invalid_rows = []
valid_rows = []
churn_na_rows = []
transformed_rows = []

with open(INPUT_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        errors = []

        if not row["user_id"].isdigit() or int(row["user_id"]) <= 0:
            errors.append("invalid user_id")

        if not is_valid_date(row["install_date"]):
            errors.append("invalid install_date")

        if not is_valid_date(row["event_date"]):
            errors.append("invalid event_date")

        dt_install = parse_datetime_with_offset(row["install_time"])
        if not dt_install:
            errors.append("invalid install_time")

        dt_event = parse_datetime_with_offset(row["event_time"])
        if not dt_event:
            errors.append("invalid event_time")

        if not row["event_counter"].isdigit() or int(row["event_counter"]) <= 0:
            errors.append("invalid event_counter")

        if not validate_event_properties(row["event_properties"]):
            errors.append("invalid event_properties")

        if row["is_churned"] not in ("0", "1", "NA"):
            errors.append("invalid is_churned")

        if not is_allowed_event_type(row["event_type"]):
            errors.append("invalid event_type")

        if row["is_churned"] == "NA":
            churn_na_rows.append(row)
        elif errors:
            invalid_rows.append({"row": row, "errors": errors})
        else:
            valid_rows.append(row)

            if dt_install and dt_event:
                transformed_rows.append({
                    "user_id": row["user_id"],
                    "install_datetime": dt_install.isoformat(),
                    "event_datetime": dt_event.isoformat(),
                    "event_type": row["event_type"],
                    "event_counter": row["event_counter"],
                    "is_churned": row["is_churned"],
                    "event_properties": row["event_properties"]
                })


def write_csv(filename, rows, fieldnames):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if valid_rows:
    write_csv(VALID_FILE, valid_rows, valid_rows[0].keys())

if churn_na_rows:
    write_csv(CHURN_NA_FILE, churn_na_rows, churn_na_rows[0].keys())

if transformed_rows:
    write_csv(TRANSFORMED_FILE, transformed_rows, transformed_rows[0].keys())

if invalid_rows:
    fieldnames = list(invalid_rows[0]["row"].keys()) + ["errors"]
    flattened = []
    for item in invalid_rows:
        r = dict(item["row"])
        r["errors"] = ", ".join(item["errors"])
        flattened.append(r)
    write_csv(INVALID_FILE, flattened, fieldnames)


print(f"Валідних записів: {len(valid_rows)}")
print(f"Невалідних записів: {len(invalid_rows)}")
print(f"is_churned == NA: {len(churn_na_rows)}")

print("\nНевалідні записи:")
for item in invalid_rows:
    row = item["row"]
    reasons = ", ".join(item["errors"])

    row_str = ", ".join([f"{k}={v}" for k, v in row.items()])

    print(f"{row_str} | причини: {reasons}")












