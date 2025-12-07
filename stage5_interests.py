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

# Користувачі які купили підписку (є подія sale_confirmation_success)
users_with_purchase = df[df["event_type"] == "sale_confirmation_success"]["user_id"].unique()

# Загальна кількість користувачів
total_users = uf["user_id"].nunique()

# CR% - Conversion Rate
cr_overall = (len(users_with_purchase) / total_users) * 100

print(f"Total users: {total_users}")
print(f"Users with purchase: {len(users_with_purchase)}")
print(f"CR% (Conversion to subscription): {cr_overall:.2f}%")

# Unsubscribe rate - серед тих хто купив, скільки відписались (is_churned=1)
# Користувачі які купили і є в user_features
purchased_users_features = uf[uf["user_id"].isin(users_with_purchase)]

users_purchased_count = len(purchased_users_features)
users_churned_count = purchased_users_features["is_churned"].sum()

unsubscribe_rate_overall = (users_churned_count / users_purchased_count * 100) if users_purchased_count > 0 else 0

print(f"Users who churned (among purchasers): {users_churned_count}")
print(f"Unsubscribe rate (%): {unsubscribe_rate_overall:.2f}%")
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

    # З цих користувачів, хто купив підписку
    purchased_with_interest = [u for u in users_with_interest if u in users_with_purchase]
    count_purchased = len(purchased_with_interest)

    # CR для цього інтересу
    cr_interest = (count_purchased / total_with_interest * 100) if total_with_interest > 0 else 0

    # З тих хто купив, хто відписався (is_churned=1)
    if count_purchased > 0:
        churned_with_interest = uf[
            (uf["user_id"].isin(purchased_with_interest)) &
            (uf["is_churned"] == 1)
            ]
        count_churned = len(churned_with_interest)
        unsubscribe_rate_interest = (count_churned / count_purchased * 100)
    else:
        count_churned = 0
        unsubscribe_rate_interest = 0

    interest_metrics.append({
        "Interest": interest,
        "Total Users": total_with_interest,
        "Purchased": count_purchased,
        "CR (%)": round(cr_interest, 2),
        "Churned": count_churned,
        "Unsubscribe Rate (%)": round(unsubscribe_rate_interest, 2)
    })

    print(f"\nInterest: {interest}")
    print(f"  Total users with this interest: {total_with_interest}")
    print(f"  Users who purchased: {count_purchased}")
    print(f"  CR%: {cr_interest:.2f}%")
    print(f"  Users who churned: {count_churned}")
    print(f"  Unsubscribe Rate%: {unsubscribe_rate_interest:.2f}%")

# Зберігаємо результати в CSV
interest_metrics_df = pd.DataFrame(interest_metrics)
interest_metrics_df = interest_metrics_df.sort_values("CR (%)", ascending=False)
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

    # З цих користувачів, хто купив підписку
    purchased_with_goal = [u for u in users_with_goal if u in users_with_purchase]
    count_purchased = len(purchased_with_goal)

    # CR для цього goal
    cr_goal = (count_purchased / total_with_goal * 100) if total_with_goal > 0 else 0

    # З тих хто купив, хто відписався (is_churned=1)
    if count_purchased > 0:
        churned_with_goal = uf[
            (uf["user_id"].isin(purchased_with_goal)) &
            (uf["is_churned"] == 1)
            ]
        count_churned = len(churned_with_goal)
        unsubscribe_rate_goal = (count_churned / count_purchased * 100)
    else:
        count_churned = 0
        unsubscribe_rate_goal = 0

    goals_metrics.append({
        "Goal": goal,
        "Total Users": total_with_goal,
        "Purchased": count_purchased,
        "CR (%)": round(cr_goal, 2),
        "Churned": count_churned,
        "Unsubscribe Rate (%)": round(unsubscribe_rate_goal, 2)
    })

    print(f"\nGoal: {goal}")
    print(f"  Total users with this goal: {total_with_goal}")
    print(f"  Users who purchased: {count_purchased}")
    print(f"  CR%: {cr_goal:.2f}%")
    print(f"  Users who churned: {count_churned}")
    print(f"  Unsubscribe Rate%: {unsubscribe_rate_goal:.2f}%")

# Зберігаємо результати в CSV
goals_metrics_df = pd.DataFrame(goals_metrics)
goals_metrics_df = goals_metrics_df.sort_values("CR (%)", ascending=False)
goals_metrics_df.to_csv("goals_metrics.csv", index=False)

print("\n" + "=" * 70)
print("Goals metrics saved to: goals_metrics.csv")
print("=" * 70)

# ======================================================================
# 7) CALCULATE METRICS BY ASSISTANCE TYPE
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

    # З цих користувачів, хто купив підписку
    purchased_with_assistance = [u for u in users_with_assistance if u in users_with_purchase]
    count_purchased = len(purchased_with_assistance)

    # CR для цього типу assistance
    cr_assistance = (count_purchased / total_with_assistance * 100) if total_with_assistance > 0 else 0

    # З тих хто купив, хто відписався (is_churned=1)
    if count_purchased > 0:
        churned_with_assistance = uf[
            (uf["user_id"].isin(purchased_with_assistance)) &
            (uf["is_churned"] == 1)
            ]
        count_churned = len(churned_with_assistance)
        unsubscribe_rate_assistance = (count_churned / count_purchased * 100)
    else:
        count_churned = 0
        unsubscribe_rate_assistance = 0

    assistance_metrics.append({
        "Assistance Type": assistance_type,
        "Total Users": total_with_assistance,
        "Purchased": count_purchased,
        "CR (%)": round(cr_assistance, 2),
        "Churned": count_churned,
        "Unsubscribe Rate (%)": round(unsubscribe_rate_assistance, 2)
    })

    print(f"\nAssistance Type: {assistance_type}")
    print(f"  Total users with this type: {total_with_assistance}")
    print(f"  Users who purchased: {count_purchased}")
    print(f"  CR%: {cr_assistance:.2f}%")
    print(f"  Users who churned: {count_churned}")
    print(f"  Unsubscribe Rate%: {unsubscribe_rate_assistance:.2f}%")

# Зберігаємо результати в CSV
assistance_metrics_df = pd.DataFrame(assistance_metrics)
assistance_metrics_df = assistance_metrics_df.sort_values("CR (%)", ascending=False)
assistance_metrics_df.to_csv("assistance_metrics.csv", index=False)

print("\n" + "=" * 70)
print("Assistance metrics saved to: assistance_metrics.csv")
print("=" * 70)

# ======================================================================
# 8) VISUALIZE METRICS BY GOALS
# ======================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Вибираємо топ-15 goals за кількістю користувачів для графіка
goals_metrics_top = goals_metrics_df.nlargest(15, "Total Users")

# Графік 1: CR% по goals (топ-15)
goals_sorted = goals_metrics_top.sort_values("CR (%)", ascending=True)
ax1.barh(goals_sorted["Goal"], goals_sorted["CR (%)"], color="mediumpurple", alpha=0.7)
ax1.axvline(x=cr_overall, color='red', linestyle='--', linewidth=2, label=f'Overall CR: {cr_overall:.2f}%')
ax1.set_xlabel("Conversion Rate (%)")
ax1.set_title("CR% by Goals (Top 15)")
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Графік 2: Unsubscribe Rate% по goals (топ-15)
goals_sorted2 = goals_metrics_top.sort_values("Unsubscribe Rate (%)", ascending=True)
ax2.barh(goals_sorted2["Goal"], goals_sorted2["Unsubscribe Rate (%)"], color="lightsalmon", alpha=0.7)
ax2.axvline(x=unsubscribe_rate_overall, color='red', linestyle='--', linewidth=2,
            label=f'Overall Unsubscribe: {unsubscribe_rate_overall:.2f}%')
ax2.set_xlabel("Unsubscribe Rate (%)")
ax2.set_title("Unsubscribe Rate% by Goals (Top 15)")
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("goals_metrics_comparison.png", dpi=200)
plt.show()

print("\nVisualization saved to: goals_metrics_comparison.png")

# ======================================================================
# 9) VISUALIZE METRICS BY INTEREST
# ======================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Графік 1: CR% по інтересах
interests_sorted = interest_metrics_df.sort_values("CR (%)", ascending=True)
ax1.barh(interests_sorted["Interest"], interests_sorted["CR (%)"], color="steelblue", alpha=0.7)
ax1.axvline(x=cr_overall, color='red', linestyle='--', linewidth=2, label=f'Overall CR: {cr_overall:.2f}%')
ax1.set_xlabel("Conversion Rate (%)")
ax1.set_title("CR% by Interest")
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Графік 2: Unsubscribe Rate% по інтересах
interests_sorted2 = interest_metrics_df.sort_values("Unsubscribe Rate (%)", ascending=True)
ax2.barh(interests_sorted2["Interest"], interests_sorted2["Unsubscribe Rate (%)"], color="coral", alpha=0.7)
ax2.axvline(x=unsubscribe_rate_overall, color='red', linestyle='--', linewidth=2,
            label=f'Overall Unsubscribe: {unsubscribe_rate_overall:.2f}%')
ax2.set_xlabel("Unsubscribe Rate (%)")
ax2.set_title("Unsubscribe Rate% by Interest")
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("interest_metrics_comparison.png", dpi=200)
plt.show()

print("\nVisualization saved to: interest_metrics_comparison.png")

# ======================================================================
# 10) VISUALIZE METRICS BY ASSISTANCE TYPE
# ======================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Графік 1: CR% по типах assistance
assistance_sorted = assistance_metrics_df.sort_values("CR (%)", ascending=True)
ax1.barh(assistance_sorted["Assistance Type"], assistance_sorted["CR (%)"], color="mediumseagreen", alpha=0.7)
ax1.axvline(x=cr_overall, color='red', linestyle='--', linewidth=2, label=f'Overall CR: {cr_overall:.2f}%')
ax1.set_xlabel("Conversion Rate (%)")
ax1.set_title("CR% by Assistance Type")
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Графік 2: Unsubscribe Rate% по типах assistance
assistance_sorted2 = assistance_metrics_df.sort_values("Unsubscribe Rate (%)", ascending=True)
ax2.barh(assistance_sorted2["Assistance Type"], assistance_sorted2["Unsubscribe Rate (%)"], color="sandybrown",
         alpha=0.7)
ax2.axvline(x=unsubscribe_rate_overall, color='red', linestyle='--', linewidth=2,
            label=f'Overall Unsubscribe: {unsubscribe_rate_overall:.2f}%')
ax2.set_xlabel("Unsubscribe Rate (%)")
ax2.set_title("Unsubscribe Rate% by Assistance Type")
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("assistance_metrics_comparison.png", dpi=200)
plt.show()

print("\nVisualization saved to: assistance_metrics_comparison.png")

# ======================================================================
# 11) DISTRIBUTIONS — ALL USERS
# ======================================================================

goals_all = answers_df[answers_df["screen"] == "goals"]["answer"].value_counts()
assistance_all = answers_df[answers_df["screen"] == "assistance"]["answer"].value_counts()
interests_all = answers_df[answers_df["screen"] == "interests"]["answer"].explode().value_counts()

goals_all.to_csv("goals_all.csv")
assistance_all.to_csv("assistance_all.csv")
interests_all.to_csv("interests_all.csv")

print("Saved ALL distributions.")

# ======================================================================
# 12) DISTRIBUTIONS — NULL USERS
# ======================================================================

goals_null = answers_null[answers_null["screen"] == "goals"]["answer"].value_counts()
assistance_null = answers_null[answers_null["screen"] == "assistance"]["answer"].value_counts()
interests_null = answers_null[answers_null["screen"] == "interests"]["answer"].explode().value_counts()

goals_null.to_csv("goals_null.csv")
assistance_null.to_csv("assistance_null.csv")
interests_null.to_csv("interests_null.csv")

print("Saved NULL USER distributions.")

# ======================================================================
# 13) PLOT COMPARISONS
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


