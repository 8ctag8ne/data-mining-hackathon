import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

# --- 1. Завантаження ---
df = pd.read_csv("user_features.csv")

print("=" * 70)
print("ОБРОБКА NULL ЗНАЧЕНЬ У time_to_first_message_sec")
print("=" * 70)

# Перевірка наявності NULL значень
null_count = df["time_to_first_message_sec"].isna().sum()
print(f"\nКількість NULL значень: {null_count}")

if null_count > 0:
    # Знаходимо максимальне значення серед ненульових записів
    max_ttfm = df["time_to_first_message_sec"].max()
    print(f"Максимальне значення time_to_first_message_sec: {max_ttfm:.2f}")

    # Заміняємо NULL на 2*MAX
    replacement_value = 2 * max_ttfm
    print(f"NULL значення будуть замінені на: {replacement_value:.2f}")

    df["time_to_first_message_sec"].fillna(replacement_value, inplace=True)
    print(f"✓ Замінено {null_count} NULL значень")
else:
    print("✓ NULL значення відсутні")

# Зберігаємо max_ttfm для використання з новими користувачами
max_ttfm_value = df["time_to_first_message_sec"].max()

# --- 2. Найважливіші фічі ---
features = [
    "answer_errors",
    "avg_event_interval_sec",
    "chat_opens",
    "chat_views",
    "dislikes",
    "error_rate",
    "likes",
    "messages_received",
    "messages_sent",
    "total_events",
    "time_to_first_message_sec",
    "like_dislike_ratio"
]

X = df[features].copy()

print(f"\nФічі для кластеризації: {len(features)}")
print(f"Кількість користувачів: {len(X)}")

# --- 3. Масштабування ---
print("\n" + "=" * 70)
print("МАСШТАБУВАННЯ ДАНИХ")
print("=" * 70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✓ Дані відмасштабовані")

# --- 4. Elbow-графік ---
print("\n" + "=" * 70)
print("ELBOW METHOD")
print("=" * 70)

inertias = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"k={k}: inertia={km.inertia_:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, marker="o", linewidth=2, markersize=8)
plt.title("Elbow Method", fontsize=14, fontweight='bold')
plt.xlabel("Number of Clusters (k)", fontsize=12)
plt.ylabel("Inertia", fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig("elbow_plot.png", dpi=200, bbox_inches='tight')
plt.show()

# --- 5. Silhouette Score ---
print("\n" + "=" * 70)
print("SILHOUETTE SCORE")
print("=" * 70)

sil_scores = {}

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores[k] = score
    print(f"k={k}: silhouette={score:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\n✓ Best k by silhouette score: {best_k} (score={sil_scores[best_k]:.4f})")

# Візуалізація Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(list(sil_scores.keys()), list(sil_scores.values()),
         marker="o", linewidth=2, markersize=8, color='green')
plt.title("Silhouette Score by Number of Clusters", fontsize=14, fontweight='bold')
plt.xlabel("Number of Clusters (k)", fontsize=12)
plt.ylabel("Silhouette Score", fontsize=12)
plt.grid(True, alpha=0.3)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
plt.legend()
plt.savefig("silhouette_plot.png", dpi=200, bbox_inches='tight')
plt.show()

# --- 6. Кластеризація ---
print("\n" + "=" * 70)
print(f"КЛАСТЕРИЗАЦІЯ (k={best_k})")
print("=" * 70)

model = KMeans(n_clusters=best_k, random_state=42)
df["cluster"] = model.fit_predict(X_scaled)

cluster_counts = df["cluster"].value_counts().sort_index()
print("\nРозподіл користувачів по кластерах:")
for cluster_id, count in cluster_counts.items():
    print(f"  Cluster {cluster_id}: {count} користувачів ({count / len(df) * 100:.1f}%)")

# --- 7. PCA для візуалізації ---
print("\n" + "=" * 70)
print("PCA ВІЗУАЛІЗАЦІЯ")
print("=" * 70)

pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
df["pca1"] = coords[:, 0]
df["pca2"] = coords[:, 1]

explained_var = pca.explained_variance_ratio_
print(f"PCA компоненти пояснюють {sum(explained_var) * 100:.1f}% варіації")
print(f"  PC1: {explained_var[0] * 100:.1f}%")
print(f"  PC2: {explained_var[1] * 100:.1f}%")

# Візуалізація кластерів у PCA просторі
plt.figure(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, best_k))

for i in range(best_k):
    cluster_data = df[df["cluster"] == i]
    plt.scatter(cluster_data["pca1"], cluster_data["pca2"],
                c=[colors[i]], label=f"Cluster {i}", alpha=0.6, s=50)

plt.xlabel(f"PC1 ({explained_var[0] * 100:.1f}% variance)", fontsize=12)
plt.ylabel(f"PC2 ({explained_var[1] * 100:.1f}% variance)", fontsize=12)
plt.title("User Clusters (PCA Visualization)", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("clusters_pca.png", dpi=200, bbox_inches='tight')
plt.show()

# --- 8. Опис кластерів ---
print("\n" + "=" * 70)
print("ПРОФІЛІ КЛАСТЕРІВ")
print("=" * 70)

cluster_profile = df.groupby("cluster")[features].mean()
cluster_size = df.groupby("cluster").size().rename("count")
cluster_churn = df.groupby("cluster")["is_churned"].mean().rename("churn_rate")

# Обчислюємо avg_ratio (likes / dislikes) по кластерах
cluster_avg_ratio = df.groupby("cluster").apply(
    lambda g: g["likes"].sum() / g["dislikes"].sum() if g["dislikes"].sum() > 0 else (
        float('inf') if g["likes"].sum() > 0 else 0
    )
).rename("avg_ratio")

# Замінюємо inf на максимальне значення + 1
max_finite_avg_ratio = cluster_avg_ratio[cluster_avg_ratio != float('inf')].max()
if pd.notna(max_finite_avg_ratio) and max_finite_avg_ratio > 0:
    replacement_avg_ratio = max_finite_avg_ratio + 1
else:
    replacement_avg_ratio = 100
cluster_avg_ratio = cluster_avg_ratio.replace(float('inf'), replacement_avg_ratio)

cluster_info = pd.concat([cluster_size, cluster_churn, cluster_avg_ratio, cluster_profile], axis=1)
cluster_info.to_csv("cluster_profiles.csv")

print("\nCluster profiles:")
print(cluster_info.round(2))

# --- 9. Додаткова статистика ---
print("\n" + "=" * 70)
print("ДОДАТКОВА СТАТИСТИКА")
print("=" * 70)

# Churn rate по кластерах
print("\nChurn rate по кластерах:")
for cluster_id in sorted(df["cluster"].unique()):
    cluster_data = df[df["cluster"] == cluster_id]
    churn_rate = cluster_data["is_churned"].mean()
    churn_count = cluster_data["is_churned"].sum()
    print(f"  Cluster {cluster_id}: {churn_rate * 100:.1f}% ({churn_count}/{len(cluster_data)})")

# Advanced users по кластерах
print("\nAdvanced users по кластерах:")
for cluster_id in sorted(df["cluster"].unique()):
    cluster_data = df[df["cluster"] == cluster_id]
    advanced_rate = cluster_data["is_advanced"].mean()
    advanced_count = cluster_data["is_advanced"].sum()
    print(f"  Cluster {cluster_id}: {advanced_rate * 100:.1f}% ({advanced_count}/{len(cluster_data)})")

# --- 10. Збереження класифікації ---
df.to_csv("clustered_users.csv", index=False)

print("\n" + "=" * 70)
print("✓ ГОТОВО!")
print("=" * 70)
print("\nЗбережені файли:")
print("  - clustered_users.csv (користувачі з номерами кластерів)")
print("  - cluster_profiles.csv (середні значення фіч по кластерах)")
print("  - elbow_plot.png (графік методу ліктя)")
print("  - silhouette_plot.png (графік silhouette score)")
print("  - clusters_pca.png (візуалізація кластерів)")

# --- 11. ЗБЕРЕЖЕННЯ МОДЕЛІ ТА SCALER ---
print("\n" + "=" * 70)
print("ЗБЕРЕЖЕННЯ МОДЕЛІ")
print("=" * 70)

# Зберігаємо модель, scaler та max_ttfm_value
with open("clustering_model.pkl", "wb") as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'features': features,
        'max_ttfm': max_ttfm_value,
        'best_k': best_k
    }, f)

print("✓ Модель збережена у файл clustering_model.pkl")

# =============================================================================
# --- 12. КЛАСИФІКАЦІЯ НОВИХ КОРИСТУВАЧІВ ---
# =============================================================================

print("\n" + "=" * 70)
print("КЛАСИФІКАЦІЯ НОВИХ КОРИСТУВАЧІВ")
print("=" * 70)

try:
    # Завантаження нових користувачів
    new_users_df = pd.read_csv("na_user_features.csv")
    print(f"\n✓ Завантажено {len(new_users_df)} нових користувачів")

    # Обробка NULL значень у time_to_first_message_sec
    null_count_new = new_users_df["time_to_first_message_sec"].isna().sum()

    if null_count_new > 0:
        print(f"\nОбробка {null_count_new} NULL значень у нових користувачів")
        replacement_value_new = 2 * max_ttfm_value
        new_users_df["time_to_first_message_sec"].fillna(replacement_value_new, inplace=True)
        print(f"✓ NULL значення замінені на: {replacement_value_new:.2f}")

    # Вибираємо тільки потрібні фічі
    X_new = new_users_df[features].copy()

    # Перевірка на наявність всіх необхідних фіч
    missing_features = set(features) - set(X_new.columns)
    if missing_features:
        print(f"\n⚠ ПОМИЛКА: Відсутні фічі: {missing_features}")
    else:
        # Масштабування нових даних використовуючи той самий scaler
        X_new_scaled = scaler.transform(X_new)
        print("✓ Нові дані відмасштабовані")

        # Класифікація нових користувачів
        new_users_df["cluster"] = model.predict(X_new_scaled)
        print("✓ Класифікація виконана")

        # Визначаємо які кластери мають високий churn rate
        print("\n" + "=" * 70)
        print("ПЕРЕДБАЧЕННЯ CHURN ДЛЯ НОВИХ КОРИСТУВАЧІВ")
        print("=" * 70)

        # Отримуємо churn rate кожного кластера з навчальних даних
        cluster_churn_rates = df.groupby("cluster")["is_churned"].mean()
        print("\nChurn rate по кластерах (з навчальних даних):")
        for cluster_id in sorted(cluster_churn_rates.index):
            rate = cluster_churn_rates[cluster_id]
            print(f"  Cluster {cluster_id}: {rate * 100:.1f}%")

        # Присвоюємо передбачення churn на основі кластера
        new_users_df["predicted_churn_probability"] = new_users_df["cluster"].map(cluster_churn_rates)
        new_users_df["predicted_churned"] = (new_users_df["predicted_churn_probability"] > 0.5).astype(int)

        # Статистика по кластерах для нових користувачів
        print("\nРозподіл нових користувачів по кластерах:")
        new_cluster_counts = new_users_df["cluster"].value_counts().sort_index()
        for cluster_id, count in new_cluster_counts.items():
            percentage = count / len(new_users_df) * 100
            churn_prob = cluster_churn_rates[cluster_id] * 100
            risk_level = "🔴 ВИСОКИЙ" if churn_prob > 70 else "🟡 СЕРЕДНІЙ" if churn_prob > 40 else "🟢 НИЗЬКИЙ"
            print(
                f"  Cluster {cluster_id}: {count} користувачів ({percentage:.1f}%) - {risk_level} ризик churn ({churn_prob:.1f}%)")

        # Загальна статистика передбачень
        predicted_churned_count = new_users_df["predicted_churned"].sum()
        predicted_churned_pct = predicted_churned_count / len(new_users_df) * 100

        print("\n" + "=" * 70)
        print("📊 ЗАГАЛЬНА СТАТИСТИКА ПЕРЕДБАЧЕНЬ")
        print("=" * 70)
        print(f"\nВсього нових користувачів: {len(new_users_df)}")
        print(f"Передбачено churned: {predicted_churned_count} ({predicted_churned_pct:.1f}%)")
        print(f"Передбачено active: {len(new_users_df) - predicted_churned_count} ({100 - predicted_churned_pct:.1f}%)")

        # Топ користувачів з найвищим ризиком churn
        print("\n" + "=" * 70)
        print("⚠️ ТОП-10 КОРИСТУВАЧІВ З НАЙВИЩИМ РИЗИКОМ CHURN")
        print("=" * 70)

        top_risk_users = new_users_df.nlargest(10, 'predicted_churn_probability')

        if 'user_id' in new_users_df.columns:
            display_cols = ['user_id', 'cluster', 'predicted_churn_probability', 'predicted_churned']
        else:
            display_cols = ['cluster', 'predicted_churn_probability', 'predicted_churned']
            top_risk_users = top_risk_users.reset_index()
            display_cols = ['index'] + display_cols

        print("\n" + top_risk_users[display_cols].to_string(index=False))

        # Рекомендації по кластерах
        print("\n" + "=" * 70)
        print("💡 РЕКОМЕНДАЦІЇ ПО РОБОТІ З КЛАСТЕРАМИ")
        print("=" * 70)

        for cluster_id in sorted(new_users_df["cluster"].unique()):
            cluster_data = new_users_df[new_users_df["cluster"] == cluster_id]
            churn_prob = cluster_churn_rates[cluster_id] * 100
            count = len(cluster_data)

            print(f"\n🔹 Cluster {cluster_id} ({count} користувачів, churn risk: {churn_prob:.1f}%):")

            if churn_prob > 70:
                print("   🔴 КРИТИЧНИЙ РИЗИК - Термінові дії:")
                print("      • Персоналізовані пропозиції та знижки")
                print("      • Пріоритетна підтримка")
                print("      • Дослідження причин незадоволення")
            elif churn_prob > 40:
                print("   🟡 СЕРЕДНІЙ РИЗИК - Превентивні заходи:")
                print("      • Engagement campaigns")
                print("      • Покращення onboarding")
                print("      • Збір feedback")
            else:
                print("   🟢 НИЗЬКИЙ РИЗИК - Підтримка активності:")
                print("      • Стандартна комунікація")
                print("      • Upsell можливості")
                print("      • Community building")

        # Додавання координат PCA для візуалізації
        coords_new = pca.transform(X_new_scaled)
        new_users_df["pca1"] = coords_new[:, 0]
        new_users_df["pca2"] = coords_new[:, 1]

        # Збереження класифікованих нових користувачів
        new_users_df.to_csv("classified_new_users.csv", index=False)
        print("\n✓ Класифіковані нові користувачі збережені у classified_new_users.csv")

        # --- Візуалізація нових користувачів на фоні існуючих кластерів ---
        print("\n" + "=" * 70)
        print("ВІЗУАЛІЗАЦІЯ НОВИХ КОРИСТУВАЧІВ")
        print("=" * 70)

        plt.figure(figsize=(14, 10))

        # Спочатку малюємо існуючі кластери (напівпрозорі)
        for i in range(best_k):
            cluster_data = df[df["cluster"] == i]
            plt.scatter(cluster_data["pca1"], cluster_data["pca2"],
                        c=[colors[i]], label=f"Cluster {i} (existing)",
                        alpha=0.3, s=30, edgecolors='none')

        # Потім малюємо нових користувачів (яскраві)
        for i in range(best_k):
            new_cluster_data = new_users_df[new_users_df["cluster"] == i]
            if len(new_cluster_data) > 0:
                plt.scatter(new_cluster_data["pca1"], new_cluster_data["pca2"],
                            c=[colors[i]], label=f"Cluster {i} (NEW)",
                            alpha=0.9, s=100, marker='*', edgecolors='black', linewidths=1)

        plt.xlabel(f"PC1 ({explained_var[0] * 100:.1f}% variance)", fontsize=12)
        plt.ylabel(f"PC2 ({explained_var[1] * 100:.1f}% variance)", fontsize=12)
        plt.title("New Users Classification (PCA Visualization)", fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("new_users_classification.png", dpi=200, bbox_inches='tight')
        plt.show()

        print("✓ Візуалізація збережена у new_users_classification.png")

        # --- Порівняння профілів нових та існуючих користувачів ---
        print("\n" + "=" * 70)
        print("ПОРІВНЯННЯ ПРОФІЛІВ")
        print("=" * 70)

        new_cluster_profile = new_users_df.groupby("cluster")[features].mean()

        print("\nСередні значення фіч для НОВИХ користувачів:")
        print(new_cluster_profile.round(2))

        print("\nПорівняння з існуючими користувачами:")
        for cluster_id in sorted(new_users_df["cluster"].unique()):
            print(f"\n--- Cluster {cluster_id} ---")
            if cluster_id in cluster_profile.index:
                comparison = pd.DataFrame({
                    'Existing': cluster_profile.loc[cluster_id],
                    'New': new_cluster_profile.loc[cluster_id],
                    'Diff %': ((new_cluster_profile.loc[cluster_id] - cluster_profile.loc[cluster_id]) /
                               (cluster_profile.loc[cluster_id] + 0.001) * 100)
                }).round(2)
                print(comparison)
            else:
                print(f"Кластер {cluster_id} не був у навчальних даних")

        # Збереження порівняльного аналізу
        comparison_summary = pd.DataFrame({
            'existing_users': df['cluster'].value_counts().sort_index(),
            'new_users': new_users_df['cluster'].value_counts().reindex(
                df['cluster'].unique(), fill_value=0
            )
        })
        comparison_summary['percentage_change'] = (
            (comparison_summary['new_users'] / comparison_summary['existing_users'] * 100)
        ).round(1)

        print("\n" + "=" * 70)
        print("ФІНАЛЬНА СТАТИСТИКА")
        print("=" * 70)
        print("\nРозподіл користувачів:")
        print(comparison_summary)

        comparison_summary.to_csv("classification_comparison.csv")
        print("\n✓ Порівняльний аналіз збережено у classification_comparison.csv")

except FileNotFoundError:
    print("\n⚠ Файл na_user_features.csv не знайдено")
    print("Пропускаємо класифікацію нових користувачів")
except Exception as e:
    print(f"\n⚠ Помилка при класифікації нових користувачів: {e}")

# =============================================================================
# --- 13. ОЦІНКА ЯКОСТІ МОДЕЛІ ---
# =============================================================================

print("\n" + "=" * 70)
print("ОЦІНКА ЯКОСТІ КЛАСТЕРИЗАЦІЇ")
print("=" * 70)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

# --- 13.1. Внутрішні метрики кластеризації (unsupervised) ---
print("\n🔍 ВНУТРІШНІ МЕТРИКИ ЯКОСТІ КЛАСТЕРІВ:")
print("-" * 70)

# Silhouette Score (вже обчислювали раніше, але покажемо ще раз)
silhouette = silhouette_score(X_scaled, df["cluster"])
print(f"Silhouette Score: {silhouette:.4f}")
print(f"  Діапазон: [-1, 1], краще > 0.5")
print(f"  Інтерпретація: {'✓ Добре' if silhouette > 0.5 else '⚠ Середньо' if silhouette > 0.25 else '✗ Погано'}")

# Davies-Bouldin Index (менше = краще)
davies_bouldin = davies_bouldin_score(X_scaled, df["cluster"])
print(f"\nDavies-Bouldin Index: {davies_bouldin:.4f}")
print(f"  Діапазон: [0, ∞], краще < 1.0")
print(f"  Інтерпретація: {'✓ Добре' if davies_bouldin < 1.0 else '⚠ Середньо' if davies_bouldin < 2.0 else '✗ Погано'}")

# Calinski-Harabasz Score (більше = краще)
calinski = calinski_harabasz_score(X_scaled, df["cluster"])
print(f"\nCalinski-Harabasz Score: {calinski:.2f}")
print(f"  Діапазон: [0, ∞], краще > 100")
print(f"  Інтерпретація: {'✓ Добре' if calinski > 100 else '⚠ Середньо' if calinski > 50 else '✗ Погано'}")

# Inertia (сума квадратів відстаней до центроїдів)
inertia = model.inertia_
print(f"\nInertia: {inertia:.2f}")
print(f"  Менше = краще (але залежить від розміру даних)")

# --- 13.2. Зовнішні метрики (supervised) - якщо є ground truth ---
print("\n" + "=" * 70)
print("ЗОВНІШНІ МЕТРИКИ (з використанням is_churned як proxy)")
print("=" * 70)

# Використаємо is_churned як ground truth для оцінки
# Це не ідеально, але дає уявлення про здатність кластерів виявляти churn
print("\n⚠ ВАЖЛИВО: Ці метрики використовують 'is_churned' як приблизний")
print("   ground truth. Кластеризація є unsupervised, тому це лише орієнтовні оцінки.")

# Створюємо "псевдо-класифікацію": кластери з churn > 50% = "churned"
cluster_churn_rates = df.groupby("cluster")["is_churned"].mean()
high_churn_clusters = cluster_churn_rates[cluster_churn_rates > 0.5].index.tolist()

df["predicted_churn"] = df["cluster"].isin(high_churn_clusters).astype(int)
y_true = df["is_churned"]
y_pred = df["predicted_churn"]

print(f"\nКластери з високим churn rate (>{50}%): {high_churn_clusters}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("\n📊 Confusion Matrix:")
print("                Predicted")
print("                Not Churn  Churn")
print(f"Actual Not Churn    {conf_matrix[0, 0]:<6}  {conf_matrix[0, 1]:<6}")
print(f"Actual Churn        {conf_matrix[1, 0]:<6}  {conf_matrix[1, 1]:<6}")

# Accuracy, Precision, Recall, F1
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='binary', zero_division=0
)

print("\n📈 МЕТРИКИ КЛАСИФІКАЦІЇ (churn prediction):")
print("-" * 70)
print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"  - З усіх передбачених churned, скільки насправді churned")
print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
print(f"  - З усіх справжніх churned, скільки ми виявили")
print(f"F1-Score:  {f1:.4f}")
print(f"  - Гармонійне середнє precision і recall")

# Детальний звіт
print("\n📋 ДЕТАЛЬНИЙ CLASSIFICATION REPORT:")
print("-" * 70)
print(classification_report(y_true, y_pred,
                            target_names=['Not Churned', 'Churned'],
                            zero_division=0))

# --- 13.3. Порівняння з базовою моделлю (baseline) ---
print("\n" + "=" * 70)
print("ПОРІВНЯННЯ З BASELINE")
print("=" * 70)

# Baseline 1: Завжди передбачати most frequent class
most_frequent_class = y_true.mode()[0]
baseline_pred_1 = np.full(len(y_true), most_frequent_class)
baseline_acc_1 = accuracy_score(y_true, baseline_pred_1)

# Baseline 2: Випадкове передбачення
np.random.seed(42)
baseline_pred_2 = np.random.randint(0, 2, len(y_true))
baseline_acc_2 = accuracy_score(y_true, baseline_pred_2)

print(f"\nBaseline 1 (most frequent class): {baseline_acc_1:.4f} ({baseline_acc_1 * 100:.2f}%)")
print(f"Baseline 2 (random prediction):   {baseline_acc_2:.4f} ({baseline_acc_2 * 100:.2f}%)")
print(f"Our Model:                         {accuracy:.4f} ({accuracy * 100:.2f}%)")

improvement_1 = ((accuracy - baseline_acc_1) / baseline_acc_1 * 100) if baseline_acc_1 > 0 else 0
improvement_2 = ((accuracy - baseline_acc_2) / baseline_acc_2 * 100) if baseline_acc_2 > 0 else 0

print(f"\n📊 Покращення відносно baseline 1: {improvement_1:+.2f}%")
print(f"📊 Покращення відносно baseline 2: {improvement_2:+.2f}%")

# --- 13.4. Збереження всіх метрик ---
metrics_summary = pd.DataFrame({
    'Metric': [
        'Silhouette Score',
        'Davies-Bouldin Index',
        'Calinski-Harabasz Score',
        'Inertia',
        'Churn Accuracy',
        'Churn Precision',
        'Churn Recall',
        'Churn F1-Score',
        'Baseline Accuracy (most frequent)',
        'Baseline Accuracy (random)',
        'Improvement vs Baseline 1 (%)',
        'Improvement vs Baseline 2 (%)'
    ],
    'Value': [
        silhouette,
        davies_bouldin,
        calinski,
        inertia,
        accuracy,
        precision,
        recall,
        f1,
        baseline_acc_1,
        baseline_acc_2,
        improvement_1,
        improvement_2
    ]
})

metrics_summary.to_csv("model_performance_metrics.csv", index=False)
print("\n✓ Метрики збережено у model_performance_metrics.csv")

# --- 13.6. Візуалізація метрик ---
print("\n" + "=" * 70)
print("ВІЗУАЛІЗАЦІЯ МЕТРИК")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix для Churn
ax1 = axes[0, 0]
im1 = ax1.imshow(conf_matrix, cmap='Blues', aspect='auto')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Not Churned', 'Churned'])
ax1.set_yticklabels(['Not Churned', 'Churned'])
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix (Churn)', fontweight='bold')

# Додаємо числа в клітинки
for i in range(2):
    for j in range(2):
        text = ax1.text(j, i, conf_matrix[i, j],
                        ha="center", va="center", color="black", fontsize=14)

plt.colorbar(im1, ax=ax1)

# 2. Метрики класифікації
ax2 = axes[0, 1]
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
churn_values = [accuracy, precision, recall, f1]

x = np.arange(len(metrics_to_plot))
bars = ax2.bar(x, churn_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])

ax2.set_xlabel('Metrics')
ax2.set_ylabel('Score')
ax2.set_title('Churn Prediction Performance', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
ax2.set_ylim([0, 1])
ax2.grid(axis='y', alpha=0.3)

# Додаємо значення на стовпчики
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Внутрішні метрики кластеризації
ax3 = axes[1, 0]
internal_metrics = ['Silhouette\nScore', 'Davies-Bouldin\nIndex', 'Calinski-Harabasz\nScore (×0.01)']
internal_values = [silhouette, davies_bouldin, calinski / 100]  # Масштабуємо CH для візуалізації

bars3 = ax3.bar(internal_metrics, internal_values, color=['green', 'orange', 'purple'])
ax3.set_ylabel('Score')
ax3.set_title('Internal Clustering Metrics', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Додаємо значення
for i, bar in enumerate(bars3):
    height = bar.get_height()
    actual_value = [silhouette, davies_bouldin, calinski][i]
    ax3.text(bar.get_x() + bar.get_width() / 2., height,
             f'{actual_value:.2f}',
             ha='center', va='bottom', fontsize=10)

# 4. Порівняння з Baseline
ax4 = axes[1, 1]
comparison_labels = ['Baseline\n(most frequent)', 'Baseline\n(random)', 'Our Model']
comparison_values = [baseline_acc_1, baseline_acc_2, accuracy]
colors_comp = ['lightgray', 'lightgray', 'green']

bars4 = ax4.bar(comparison_labels, comparison_values, color=colors_comp)
ax4.set_ylabel('Accuracy')
ax4.set_title('Model vs Baseline Comparison', fontweight='bold')
ax4.set_ylim([0, 1])
ax4.grid(axis='y', alpha=0.3)

# Додаємо значення
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("model_performance_analysis.png", dpi=200, bbox_inches='tight')
plt.show()

print("✓ Візуалізація збережена у model_performance_analysis.png")

# --- 13.7. Фінальний звіт ---
print("\n" + "=" * 70)
print("📊 ФІНАЛЬНИЙ ЗВІТ ПРО ЯКІСТЬ МОДЕЛІ")
print("=" * 70)

print(f"""
🎯 ЗАГАЛЬНА ОЦІНКА КЛАСТЕРИЗАЦІЇ:
{'=' * 70}
Модель створила {best_k} кластери з наступними характеристиками:

📌 ВНУТРІШНІ МЕТРИКИ (якість розділення):
   • Silhouette Score: {silhouette:.4f} {'✓ Добре' if silhouette > 0.5 else '⚠ Середньо'}
   • Davies-Bouldin Index: {davies_bouldin:.4f} {'✓ Добре' if davies_bouldin < 1.0 else '⚠ Середньо'}
   • Calinski-Harabasz Score: {calinski:.2f} {'✓ Добре' if calinski > 100 else '⚠ Середньо'}

📌 ПРОГНОЗУВАННЯ CHURN:
   • Accuracy: {accuracy * 100:.2f}% (на {improvement_1:+.1f}% краще baseline)
   • Precision: {precision * 100:.2f}% (точність виявлення churned users)
   • Recall: {recall * 100:.2f}% (повнота виявлення churned users)
   • F1-Score: {f1:.4f}

💡 ВИСНОВОК:
   Кластеризація {'успішно' if silhouette > 0.5 and davies_bouldin < 1.0 else 'помірно'} розділяє користувачів на сегменти.
   Модель {'значно' if improvement_1 > 10 else 'помірно' if improvement_1 > 0 else 'не'} перевершує baseline підходи.
   {'Висока точність (precision) і відмінний recall вказують на надійність виявлення churn.' if precision > 0.8 and recall > 0.9 else ''}
""")

print("\n" + "=" * 70)
print("✓ ВСЕ ЗАВДАННЯ ВИКОНАНО!")
print("=" * 70)


