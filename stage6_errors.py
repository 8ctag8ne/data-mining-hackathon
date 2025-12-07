import pandas as pd
import json
import matplotlib.pyplot as plt


# ======================================================================
# 1) FUNCTIONS
# ======================================================================

def parse_props(s):
    """Парсинг JSON властивостей події"""
    try:
        return json.loads(s)
    except:
        return {}


# ======================================================================
# 2) LOAD DATA
# ======================================================================

print("=" * 70)
print("MODEL ERROR ANALYSIS")
print("=" * 70)

df = pd.read_csv("valid.csv")

# Парсимо event_properties для отримання model_type
df["props"] = df["event_properties"].apply(parse_props)
df["model_type"] = df["props"].apply(lambda x: x.get("model_type"))

# ======================================================================
# 3) FILTER EVENTS WITH MODEL TYPE
# ======================================================================

# Відбираємо тільки події, що мають model_type (basic або advanced)
model_events = df[df["model_type"].notna()].copy()

print(f"\nTotal events with model_type: {len(model_events)}")
print(f"Total users with model events: {model_events['user_id'].nunique()}")

# ======================================================================
# 4) ANALYSIS 1: DISTRIBUTION OF answer_error BY MODEL TYPE
# ======================================================================

print("\n" + "=" * 70)
print("ANALYSIS 1: Distribution of answer_error events by model type")
print("=" * 70)

# Відбираємо всі answer_error події
error_events = model_events[model_events["event_type"] == "answer_error"]

print(f"\nTotal answer_error events: {len(error_events)}")

# Розподіл помилок по типах моделей
errors_by_model = error_events["model_type"].value_counts()
errors_by_model_pct = (errors_by_model / errors_by_model.sum() * 100)

print(f"\nDistribution of answer_error by model type:")
for model, count in errors_by_model.items():
    pct = errors_by_model_pct[model]
    print(f"  {model}: {count} errors ({pct:.2f}%)")

# Зберігаємо в CSV
errors_distribution = pd.DataFrame({
    'Model Type': errors_by_model.index,
    'Error Count': errors_by_model.values,
    'Percentage (%)': errors_by_model_pct.values.round(2)
})
errors_distribution.to_csv("errors_by_model_distribution.csv", index=False)

# ======================================================================
# 5) ANALYSIS 2: ERROR RATE FOR BASIC MODEL
# ======================================================================

print("\n" + "=" * 70)
print("ANALYSIS 2: Error rate for BASIC model")
print("=" * 70)

# ВИПРАВЛЕНО: рахуємо тільки send_message_tap події
basic_messages = model_events[
    (model_events["model_type"] == "basic") &
    (model_events["event_type"] == "send_message_tap")
    ]
basic_errors = model_events[
    (model_events["model_type"] == "basic") &
    (model_events["event_type"] == "answer_error")
    ]

total_basic_messages = len(basic_messages)
errors_basic = len(basic_errors)
error_rate_basic = (errors_basic / total_basic_messages * 100) if total_basic_messages > 0 else 0

print(f"\nTotal BASIC model messages sent: {total_basic_messages}")
print(f"BASIC model answer_error events: {errors_basic}")
print(f"Error rate for BASIC model: {error_rate_basic:.2f}%")

# ======================================================================
# 6) ANALYSIS 3: ERROR RATE FOR ADVANCED MODEL
# ======================================================================

print("\n" + "=" * 70)
print("ANALYSIS 3: Error rate for ADVANCED model")
print("=" * 70)

# ВИПРАВЛЕНО: рахуємо тільки send_message_tap події
advanced_messages = model_events[
    (model_events["model_type"] == "advanced") &
    (model_events["event_type"] == "send_message_tap")
    ]
advanced_errors = model_events[
    (model_events["model_type"] == "advanced") &
    (model_events["event_type"] == "answer_error")
    ]

total_advanced_messages = len(advanced_messages)
errors_advanced = len(advanced_errors)
error_rate_advanced = (errors_advanced / total_advanced_messages * 100) if total_advanced_messages > 0 else 0

print(f"\nTotal ADVANCED model messages sent: {total_advanced_messages}")
print(f"ADVANCED model answer_error events: {errors_advanced}")
print(f"Error rate for ADVANCED model: {error_rate_advanced:.2f}%")

# ======================================================================
# 7) SUMMARY TABLE
# ======================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

total_messages = total_basic_messages + total_advanced_messages
total_errors = errors_basic + errors_advanced

summary_data = {
    'Model Type': ['basic', 'advanced', 'TOTAL'],
    'Messages Sent': [total_basic_messages, total_advanced_messages, total_messages],
    'Error Events': [errors_basic, errors_advanced, total_errors],
    'Error Rate (%)': [
        round(error_rate_basic, 2),
        round(error_rate_advanced, 2),
        round((total_errors / total_messages * 100) if total_messages > 0 else 0, 2)
    ],
    'Share of All Errors (%)': [
        round(errors_by_model_pct.get('basic', 0), 2),
        round(errors_by_model_pct.get('advanced', 0), 2),
        100.0
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Зберігаємо зведену таблицю
summary_df.to_csv("model_error_summary.csv", index=False)

# ======================================================================
# 8) VISUALIZATIONS
# ======================================================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Графік 1: Розподіл помилок по моделях (абсолютні значення)
colors_abs = ['#e74c3c', '#3498db']
ax1.bar(errors_by_model.index, errors_by_model.values, color=colors_abs, alpha=0.7)
ax1.set_ylabel("Number of Errors")
ax1.set_title("Total answer_error Events by Model Type", fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for i, (model, count) in enumerate(errors_by_model.items()):
    ax1.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')

# Графік 2: Розподіл помилок по моделях (%)
ax2.pie(errors_by_model_pct.values, labels=errors_by_model_pct.index,
        autopct='%1.1f%%', colors=colors_abs, startangle=90)
ax2.set_title("Distribution of answer_error Events (%)", fontsize=12, fontweight='bold')

# Графік 3: Error Rate по моделях
error_rates = [error_rate_basic, error_rate_advanced]
model_names = ['basic', 'advanced']
colors_rate = ['#e74c3c', '#3498db']
bars = ax3.bar(model_names, error_rates, color=colors_rate, alpha=0.7)
ax3.set_ylabel("Error Rate (%)")
ax3.set_title("Error Rate by Model Type\n(errors / messages sent)", fontsize=12, fontweight='bold')
ax3.set_ylim(0, max(error_rates) * 1.2 if max(error_rates) > 0 else 1)
ax3.grid(axis='y', alpha=0.3)
for i, (model, rate) in enumerate(zip(model_names, error_rates)):
    ax3.text(i, rate, f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold')

# Графік 4: Порівняння messages sent vs помилки
x = range(len(model_names))
width = 0.35
ax4.bar([i - width / 2 for i in x], [total_basic_messages, total_advanced_messages],
        width=width, label='Messages Sent', color='lightgray', alpha=0.7)
ax4.bar([i + width / 2 for i in x], [errors_basic, errors_advanced],
        width=width, label='Error Events', color=['#e74c3c', '#3498db'], alpha=0.7)
ax4.set_ylabel("Number of Events")
ax4.set_title("Messages Sent vs Error Events by Model", fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(model_names)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("model_error_analysis.png", dpi=200, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("✓ Analysis complete!")
print("=" * 70)
print("\nGenerated files:")
print("  - errors_by_model_distribution.csv")
print("  - model_error_summary.csv")
print("  - model_error_analysis.png")

# ======================================================================
# 9) DETAILED EVENT TYPE BREAKDOWN
# ======================================================================

print("\n" + "=" * 70)
print("DETAILED: Event types breakdown by model")
print("=" * 70)

# Розподіл типів подій для кожної моделі
for model in ['basic', 'advanced']:
    model_data = model_events[model_events["model_type"] == model]
    event_types = model_data["event_type"].value_counts()

    print(f"\n{model.upper()} model - Event types:")
    for event_type, count in event_types.head(10).items():
        pct = count / len(model_data) * 100
        print(f"  {event_type}: {count} ({pct:.2f}%)")

    # Зберігаємо детальний breakdown
    event_breakdown = pd.DataFrame({
        'Event Type': event_types.index,
        'Count': event_types.values,
        'Percentage (%)': (event_types.values / len(model_data) * 100).round(2)
    })
    event_breakdown.to_csv(f"{model}_event_breakdown.csv", index=False)

print("\n✓ Detailed breakdowns saved to:")
print("  - basic_event_breakdown.csv")
print("  - advanced_event_breakdown.csv")

# ======================================================================
# 10) ADDITIONAL ANALYSIS: Messages per User
# ======================================================================

print("\n" + "=" * 70)
print("ADDITIONAL: Messages per user by model")
print("=" * 70)

for model in ['basic', 'advanced']:
    model_messages = model_events[
        (model_events["model_type"] == model) &
        (model_events["event_type"] == "send_message_tap")
        ]
    model_errors = model_events[
        (model_events["model_type"] == model) &
        (model_events["event_type"] == "answer_error")
        ]

    users_count = model_messages["user_id"].nunique()
    avg_messages = len(model_messages) / users_count if users_count > 0 else 0
    avg_errors = len(model_errors) / users_count if users_count > 0 else 0

    print(f"\n{model.upper()} model:")
    print(f"  Users who sent messages: {users_count}")
    print(f"  Avg messages per user: {avg_messages:.2f}")
    print(f"  Avg errors per user: {avg_errors:.2f}")