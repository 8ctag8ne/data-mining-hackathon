import pandas as pd
import json
import matplotlib.pyplot as plt


# ======================================================================
# 1) FUNCTIONS
# ======================================================================

def parse_props(s):
    try:
        return json.loads(s)
    except:
        return {}


def prepare_answers(df):
    df = df.copy()
    df["props"] = df["event_properties"].apply(parse_props)
    df["screen"] = df["props"].apply(lambda x: x.get("screen_name"))
    df["answer"] = df["props"].apply(lambda x: x.get("answers"))
    # Split for interests
    df["answer"] = df.apply(
        lambda r: [i.strip() for i in r["answer"].split(",")]
        if r["screen"] == "interests" and isinstance(r["answer"], str)
        else r["answer"],
        axis=1
    )
    return df


def plot_comparison(all_dist, null_dist, title, fname, top_n=None):
    plt.figure(figsize=(12, 6))
    # Select top N (used only for goals)
    if top_n:
        all_dist = all_dist.head(top_n)
        null_dist = null_dist.reindex(all_dist.index).fillna(0)
    # Align indexes for both distributions
    idx = sorted(set(all_dist.index) | set(null_dist.index))
    all_vals = [all_dist.get(i, 0) for i in idx]
    null_vals = [null_dist.get(i, 0) for i in idx]
    x = range(len(idx))
    w = 0.35
    plt.bar([i - w / 2 for i in x], all_vals, width=w, label="All users")
    plt.bar([i + w / 2 for i in x], null_vals, width=w, label="NULL TTFM users")
    plt.xticks(x, idx, rotation=45, ha="right")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.show()


# ======================================================================
# 2) LOAD DATA
# ======================================================================

df = pd.read_csv("valid.csv")
uf = pd.read_csv("user_features.csv")

# Filter quiz "continue" events
answers_df = df[df["event_type"] == "quiz_question_continue_tap"].copy()
answers_df = prepare_answers(answers_df)

# Users with NULL TTFM
null_users = uf[uf["time_to_first_message_sec"].isna()]["user_id"].unique()
answers_null = df[df["user_id"].isin(null_users)]
answers_null = answers_null[answers_null["event_type"] == "quiz_question_continue_tap"]
answers_null = prepare_answers(answers_null)

# ======================================================================
# 3) CALCULATE OVERALL METRICS
# ======================================================================

print("=" * 70)
print("OVERALL METRICS")
print("=" * 70)

# Загальна кількість користувачів
total_users = uf["user_id"].nunique()

# Advanced users rate
advanced_users_count = uf["is_advanced"].sum()
advanced_rate_overall = (advanced_users_count / total_users * 100)

print(f"Total users: {total_users}")
print(f"Users who used advanced model: {advanced_users_count}")
print(f"Advanced users rate (%): {advanced_rate_overall:.2f}%")

# Error rate - загальний
total_messages = uf["messages_sent"].sum()
total_errors = uf["answer_errors"].sum()  # Глобальна для share
error_rate_overall = (total_errors / total_messages * 100) if total_messages > 0 else 0

print(f"Total messages sent: {total_messages}")
print(f"Total errors: {total_errors}")
print(f"Error rate (%): {error_rate_overall:.2f}%")
print()

# ======================================================================
# 4) CALCULATE METRICS BY INTEREST
# ======================================================================

print("=" * 70)
print("METRICS BY INTEREST")
print("=" * 70)

# Отримуємо всі інтереси
interests_df = answers_df[answers_df["screen"] == "interests"].copy()

# Створюємо розгорнутий DataFrame з окремими інтересами
interests_exploded = interests_df.explode("answer")
interests_exploded = interests_exploded[interests_exploded["answer"].notna()]

# Унікальні інтереси
unique_interests = interests_exploded["answer"].unique()

# Результати по кожному інтересу
interest_metrics = []

for interest in sorted(unique_interests):
    # Користувачі які вибрали цей інтерес
    users_with_interest = interests_exploded[
        interests_exploded["answer"] == interest
        ]["user_id"].unique()

    total_with_interest = len(users_with_interest)

    # Error rate серед цієї групи
    users_features_interest = uf[uf["user_id"].isin(users_with_interest)]
    total_messages_interest = users_features_interest["messages_sent"].sum()
    total_errors_interest = users_features_interest["answer_errors"].sum()
    error_rate_interest = (total_errors_interest / total_messages_interest * 100) if total_messages_interest > 0 else 0

    # Нова: Частка помилок від загальної
    error_share_interest = (total_errors_interest / total_errors * 100) if total_errors > 0 else 0

    interest_metrics.append({
        "Interest": interest,
        "Total Users": total_with_interest,
        "Total Messages": total_messages_interest,
        "Total Errors": total_errors_interest,
        "Error Rate (%)": round(error_rate_interest, 2),
        "Error Share (%)": round(error_share_interest, 2)
    })

    print(f"\nInterest: {interest}")
    print(f"  Total users with this interest: {total_with_interest}")
    print(f"  Total messages: {total_messages_interest}")
    print(f"  Total errors: {total_errors_interest}")
    print(f"  Error Rate%: {error_rate_interest:.2f}%")
    print(f"  Error Share%: {error_share_interest:.2f}%")

# Зберігаємо результати в CSV
interest_metrics_df = pd.DataFrame(interest_metrics)
interest_metrics_df = interest_metrics_df.sort_values("Error Share (%)", ascending=False)
interest_metrics_df.to_csv("interest_metrics.csv", index=False)

print("\n" + "=" * 70)
print("Interest metrics saved to: interest_metrics.csv")
print("=" * 70)

# ======================================================================
# 5) CALCULATE METRICS BY GOALS
# ======================================================================

print("\n" + "=" * 70)
print("METRICS BY GOALS")
print("=" * 70)

# Отримуємо всі goals
goals_df = answers_df[answers_df["screen"] == "goals"].copy()

# Створюємо розгорнутий DataFrame з окремими goals
# Goals також можуть бути множинними (розділені пробілами або комами)
goals_exploded_list = []
for _, row in goals_df.iterrows():
    answer = row["answer"]
    if pd.notna(answer):
        # Розділяємо по пробілах і комах
        goals_list = [g.strip() for g in str(answer).replace(",", " ").split()]
        for goal in goals_list:
            if goal:
                goals_exploded_list.append({
                    "user_id": row["user_id"],
                    "goal": goal
                })

goals_exploded = pd.DataFrame(goals_exploded_list)

# Унікальні goals
unique_goals = goals_exploded["goal"].unique()

# Результати по кожному goal
goals_metrics = []

for goal in sorted(unique_goals):
    # Користувачі які вибрали цей goal
    users_with_goal = goals_exploded[
        goals_exploded["goal"] == goal
        ]["user_id"].unique()

    total_with_goal = len(users_with_goal)

    # Error rate серед цієї групи
    users_features_goal = uf[uf["user_id"].isin(users_with_goal)]
    total_messages_goal = users_features_goal["messages_sent"].sum()
    total_errors_goal = users_features_goal["answer_errors"].sum()
    error_rate_goal = (total_errors_goal / total_messages_goal * 100) if total_messages_goal > 0 else 0

    # Нова: Частка помилок від загальної
    error_share_goal = (total_errors_goal / total_errors * 100) if total_errors > 0 else 0

    goals_metrics.append({
        "Goal": goal,
        "Total Users": total_with_goal,
        "Total Messages": total_messages_goal,
        "Total Errors": total_errors_goal,
        "Error Rate (%)": round(error_rate_goal, 2),
        "Error Share (%)": round(error_share_goal, 2)
    })

    print(f"\nGoal: {goal}")
    print(f"  Total users with this goal: {total_with_goal}")
    print(f"  Total messages: {total_messages_goal}")
    print(f"  Total errors: {total_errors_goal}")
    print(f"  Error Rate%: {error_rate_goal:.2f}%")
    print(f"  Error Share%: {error_share_goal:.2f}%")

# Зберігаємо результати в CSV
goals_metrics_df = pd.DataFrame(goals_metrics)
goals_metrics_df = goals_metrics_df.sort_values("Error Share (%)", ascending=False)
goals_metrics_df.to_csv("goals_metrics.csv", index=False)

print("\n" + "=" * 70)
print("Goals metrics saved to: goals_metrics.csv")
print("=" * 70)

# ======================================================================
# 6) CALCULATE METRICS BY ASSISTANCE TYPE
# ======================================================================

print("\n" + "=" * 70)
print("METRICS BY ASSISTANCE TYPE")
print("=" * 70)

# Отримуємо всі типи assistance
assistance_df = answers_df[answers_df["screen"] == "assistance"].copy()

# Унікальні типи assistance
unique_assistance = assistance_df["answer"].unique()

# Результати по кожному типу assistance
assistance_metrics = []

for assistance_type in sorted(unique_assistance):
    if pd.isna(assistance_type):
        continue

    # Користувачі які вибрали цей тип assistance
    users_with_assistance = assistance_df[
        assistance_df["answer"] == assistance_type
        ]["user_id"].unique()

    total_with_assistance = len(users_with_assistance)

    # Error rate серед цієї групи
    users_features_assistance = uf[uf["user_id"].isin(users_with_assistance)]
    total_messages_assistance = users_features_assistance["messages_sent"].sum()
    total_errors_assistance = users_features_assistance["answer_errors"].sum()
    error_rate_assistance = (
            total_errors_assistance / total_messages_assistance * 100) if total_messages_assistance > 0 else 0

    # Нова: Частка помилок від загальної
    error_share_assistance = (total_errors_assistance / total_errors * 100) if total_errors > 0 else 0

    assistance_metrics.append({
        "Assistance Type": assistance_type,
        "Total Users": total_with_assistance,
        "Total Messages": total_messages_assistance,
        "Total Errors": total_errors_assistance,
        "Error Rate (%)": round(error_rate_assistance, 2),
        "Error Share (%)": round(error_share_assistance, 2)
    })

    print(f"\nAssistance Type: {assistance_type}")
    print(f"  Total users with this type: {total_with_assistance}")
    print(f"  Total messages: {total_messages_assistance}")
    print(f"  Total errors: {total_errors_assistance}")
    print(f"  Error Rate%: {error_rate_assistance:.2f}%")
    print(f"  Error Share%: {error_share_assistance:.2f}%")

# Зберігаємо результати в CSV
assistance_metrics_df = pd.DataFrame(assistance_metrics)
assistance_metrics_df = assistance_metrics_df.sort_values("Error Share (%)", ascending=False)
assistance_metrics_df.to_csv("assistance_metrics.csv", index=False)

print("\n" + "=" * 70)
print("Assistance metrics saved to: assistance_metrics.csv")
print("=" * 70)

# ======================================================================
# 7) VISUALIZE METRICS BY GOALS
# ======================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Вибираємо топ-15 goals за кількістю користувачів для графіка
goals_metrics_top = goals_metrics_df.nlargest(15, "Total Users")

# Графік 1: Error Share% по goals (топ-15)
goals_sorted = goals_metrics_top.sort_values("Error Share (%)", ascending=True)
ax1.barh(goals_sorted["Goal"], goals_sorted["Error Share (%)"], color="skyblue", alpha=0.7)
ax1.set_xlabel("Error Share (%)")
ax1.set_title("Error Share% by Goals (Top 15)")
ax1.grid(axis='x', alpha=0.3)

# Графік 2: Error Rate% по goals (топ-15)
goals_sorted2 = goals_metrics_top.sort_values("Error Rate (%)", ascending=True)
ax2.barh(goals_sorted2["Goal"], goals_sorted2["Error Rate (%)"], color="lightcoral", alpha=0.7)
ax2.axvline(x=error_rate_overall, color='red', linestyle='--', linewidth=2,
            label=f'Overall Error Rate: {error_rate_overall:.2f}%')
ax2.set_xlabel("Error Rate (%)")
ax2.set_title("Error Rate% by Goals (Top 15)")
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("goals_metrics_comparison.png", dpi=200)
plt.show()

print("\nVisualization saved to: goals_metrics_comparison.png")

# ======================================================================
# 8) VISUALIZE METRICS BY INTEREST
# ======================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Графік 1: Error Share% по інтересах
interests_sorted = interest_metrics_df.sort_values("Error Share (%)", ascending=True)
ax1.barh(interests_sorted["Interest"], interests_sorted["Error Share (%)"], color="lightgreen", alpha=0.7)
ax1.set_xlabel("Error Share (%)")
ax1.set_title("Error Share% by Interest")
ax1.grid(axis='x', alpha=0.3)

# Графік 2: Error Rate% по інтересах
interests_sorted2 = interest_metrics_df.sort_values("Error Rate (%)", ascending=True)
ax2.barh(interests_sorted2["Interest"], interests_sorted2["Error Rate (%)"], color="salmon", alpha=0.7)
ax2.axvline(x=error_rate_overall, color='red', linestyle='--', linewidth=2,
            label=f'Overall Error Rate: {error_rate_overall:.2f}%')
ax2.set_xlabel("Error Rate (%)")
ax2.set_title("Error Rate% by Interest")
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("interest_metrics_comparison.png", dpi=200)
plt.show()

print("\nVisualization saved to: interest_metrics_comparison.png")

# ======================================================================
# 9) VISUALIZE METRICS BY ASSISTANCE TYPE
# ======================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Графік 1: Error Share% по типах assistance
assistance_sorted = assistance_metrics_df.sort_values("Error Share (%)", ascending=True)
ax1.barh(assistance_sorted["Assistance Type"], assistance_sorted["Error Share (%)"], color="plum", alpha=0.7)
ax1.set_xlabel("Error Share (%)")
ax1.set_title("Error Share% by Assistance Type")
ax1.grid(axis='x', alpha=0.3)

# Графік 2: Error Rate% по типах assistance
assistance_sorted2 = assistance_metrics_df.sort_values("Error Rate (%)", ascending=True)
ax2.barh(assistance_sorted2["Assistance Type"], assistance_sorted2["Error Rate (%)"], color="peachpuff", alpha=0.7)
ax2.axvline(x=error_rate_overall, color='red', linestyle='--', linewidth=2,
            label=f'Overall Error Rate: {error_rate_overall:.2f}%')
ax2.set_xlabel("Error Rate (%)")
ax2.set_title("Error Rate% by Assistance Type")
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("assistance_metrics_comparison.png", dpi=200)
plt.show()

print("\nVisualization saved to: assistance_metrics_comparison.png")

# ======================================================================
# 10) DISTRIBUTIONS — ALL USERS
# ======================================================================

goals_all = answers_df[answers_df["screen"] == "goals"]["answer"].value_counts()
assistance_all = answers_df[answers_df["screen"] == "assistance"]["answer"].value_counts()
interests_all = answers_df[answers_df["screen"] == "interests"]["answer"].explode().value_counts()

goals_all.to_csv("goals_all.csv")
assistance_all.to_csv("assistance_all.csv")
interests_all.to_csv("interests_all.csv")

print("Saved ALL distributions.")

# ======================================================================
# 11) DISTRIBUTIONS — NULL USERS
# ======================================================================

goals_null = answers_null[answers_null["screen"] == "goals"]["answer"].value_counts()
assistance_null = answers_null[answers_null["screen"] == "assistance"]["answer"].value_counts()
interests_null = answers_null[answers_null["screen"] == "interests"]["answer"].explode().value_counts()

goals_null.to_csv("goals_null.csv")
assistance_null.to_csv("assistance_null.csv")
interests_null.to_csv("interests_null.csv")

print("Saved NULL USER distributions.")

# ======================================================================
# 12) PLOT COMPARISONS
# ======================================================================

# Goals — only top-10
plot_comparison(
    goals_all,
    goals_null,
    title="GOALS — Comparison (Top 10)",
    fname="goals_comparison.png",
    top_n=10
)

# Assistance
plot_comparison(
    assistance_all,
    assistance_null,
    title="ASSISTANCE — Comparison",
    fname="assistance_comparison.png"
)

# Interests
plot_comparison(
    interests_all,
    interests_null,
    title="INTERESTS — Comparison",
    fname="interests_comparison.png"
)

print("All comparison plots generated.")

