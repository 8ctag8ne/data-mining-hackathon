import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import umap
import hdbscan
import warnings

warnings.filterwarnings('ignore')

# Налаштування для кращої візуалізації
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class UserSegmentation:
    def __init__(self, filepath):
        """Ініціалізація класу для сегментації користувачів"""
        self.filepath = filepath
        self.df_original = None
        self.df_cleaned = None
        self.df_features = None
        self.df_normalized = None
        self.umap_embedding = None
        self.hdbscan_model = None
        self.kmeans_model = None
        self.optimal_k = None
        self.scaler = RobustScaler()
        self.n_clusters_found = None

    def load_and_clean_data(self):
        """Крок 1: Завантаження та очищення датасету"""
        print("=" * 60)
        print("КРОК 1: Завантаження та очищення датасету")
        print("=" * 60)

        # Завантаження даних
        self.df_original = pd.read_csv(self.filepath)
        print(f"✓ Завантажено {len(self.df_original)} записів")
        print(f"✓ Кількість колонок: {len(self.df_original.columns)}")

        self.df_cleaned = self.df_original.copy()

        # Видалення дублікатів
        duplicates = self.df_cleaned.duplicated().sum()
        self.df_cleaned = self.df_cleaned.drop_duplicates()
        print(f"✓ Видалено дублікатів: {duplicates}")

        # Обробка пропущених значень
        missing_before = self.df_cleaned.isnull().sum().sum()
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns

        # Спеціальна обробка time_to_first_message_sec
        no_messages = self.df_cleaned['time_to_first_message_sec'].isnull()
        max_time = self.df_cleaned['time_to_first_message_sec'].max()
        self.df_cleaned.loc[no_messages, 'time_to_first_message_sec'] = max_time * 2

        for col in numeric_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                self.df_cleaned[col].fillna(self.df_cleaned[col].median(), inplace=True)

        print(f"✓ Заповнено пропущених значень: {missing_before}")

        # -------------------------------------------------------------------
        # 🟩  MULTI-HOT ENCODING З АГРЕГАЦІЄЮ У КАТЕГОРІЇ
        # -------------------------------------------------------------------
        print("✓ Виконується категоризація goals / interests / assistance...")

        def split_values(s):
            if pd.isna(s) or s == "":
                return []
            return [x.strip().lower() for x in str(s).split(",") if x.strip()]

        # нормалізуємо списки
        self.df_cleaned["goals_list"] = self.df_cleaned["user_goals"].apply(split_values)
        self.df_cleaned["interests_list"] = self.df_cleaned["user_interests"].apply(split_values)
        self.df_cleaned["assistance_list"] = self.df_cleaned["user_assistance"].apply(split_values)

        # -----------------------------------------
        # КАТЕГОРІЇ ДЛЯ АГРЕГАЦІЇ
        # -----------------------------------------

        GOALS_CATEGORIES = {
            "goals_creative": ["content-creation", "essay-writing", "cooking"],
            "goals_professional": ["coding-assistance", "business-purposes"],
            "goals_personal_dev": ["education", "mental-health"],
            "goals_social_entertain": ["social-media", "entertainment"]
        }

        ASSISTANCE_CATEGORIES = {
            "assist_detailed": ["step-by-step", "detailed-explanation", "answer-with-explanation"],
            "assist_concise": ["direct-answer", "simplified-explanation"]
        }

        INTEREST_CATEGORIES = {
            "interest_creative_arts": ["arts", "music", "writing", "dance", "movies"],
            "interest_practical": ["diy", "cooking", "gardening"],
            "interest_outdoor": ["sports", "outdoor", "travel"],
            "interest_intellectual": ["reading", "learning", "history"],
            "interest_business_tech": ["business", "technology"],
            "interest_lifestyle": ["fashion", "family", "animals"]
        }

        # -----------------------------------------
        # ФУНКЦІЯ ДЛЯ СТВОРЕННЯ БІНАРНИХ ОЗНАК
        # -----------------------------------------

        def assign_categories(df, source_col, categories):
            for new_col, group_values in categories.items():
                df[new_col] = df[source_col].apply(
                    lambda lst: int(any(item in lst for item in group_values))
                )
            return df

        # застосування
        self.df_cleaned = assign_categories(self.df_cleaned, "goals_list", GOALS_CATEGORIES)
        self.df_cleaned = assign_categories(self.df_cleaned, "assistance_list", ASSISTANCE_CATEGORIES)
        self.df_cleaned = assign_categories(self.df_cleaned, "interests_list", INTEREST_CATEGORIES)

        print(f"✓ Створено {len(GOALS_CATEGORIES)} категорій goals")
        print(f"✓ Створено {len(ASSISTANCE_CATEGORIES)} категорій assistance")
        print(f"✓ Створено {len(INTEREST_CATEGORIES)} категорій interests")

        # видаляємо сирі колонки
        self.df_cleaned.drop(columns=[
            "goals_list", "interests_list", "assistance_list",
            "user_goals", "user_interests", "user_assistance"
        ], inplace=True)

        # -------------------------------------------------------------------

        len_before = len(self.df_cleaned)

        print(f"✓ Видалено викидів (outliers): {len_before - len(self.df_cleaned)}")
        print(f"✓ Залишилось записів після очищення: {len(self.df_cleaned)}\n")

    def feature_selection(self):
        """Крок 2: Відбір релевантних ознак"""
        print("=" * 60)
        print("КРОК 2: Відбір релевантних ознак")
        print("=" * 60)

        self.df_features = self.df_cleaned.copy()

        # Видалення ідентифікаторів та нерелевантних колонок
        columns_to_remove = ['user_id', 'model_changes']
        self.df_features = self.df_features.drop(columns=columns_to_remove, errors='ignore')
        print(f"✓ Видалено ідентифікатори: {columns_to_remove}")

        # Видалення колонок з низькою варіативністю (std < 0.01)
        numeric_cols = self.df_features.select_dtypes(include=[np.number]).columns
        low_variance_cols = []

        for col in numeric_cols:
            if self.df_features[col].std() < 0.01:
                low_variance_cols.append(col)

        if low_variance_cols:
            self.df_features = self.df_features.drop(columns=low_variance_cols)
            print(f"✓ Видалено колонок з низькою варіативністю: {low_variance_cols}")
        else:
            print("✓ Колонок з низькою варіативністю не знайдено")

        # Видалення високо корельованих ознак
        correlation_matrix = self.df_features.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        high_corr_cols = []
        for column in upper_triangle.columns:
            if any(upper_triangle[column] > 0.9):
                high_corr_cols.append(column)

        if high_corr_cols:
            self.df_features = self.df_features.drop(columns=high_corr_cols)
            print(f"✓ Видалено високо корельованих колонок (r > 0.9): {high_corr_cols}")
        else:
            print("✓ Високо корельованих колонок не знайдено")

        print(f"✓ Залишилось ознак для кластеризації: {len(self.df_features.columns)}")
        print(f"  Ознаки: {list(self.df_features.columns)}\n")

    def normalize_data(self):
        """Крок 3: Нормалізація даних"""
        print("=" * 60)
        print("КРОК 3: Нормалізація даних")
        print("=" * 60)

        self.df_normalized = pd.DataFrame(
            self.scaler.fit_transform(self.df_features),
            columns=self.df_features.columns,
            index=self.df_features.index
        )

        print("✓ Дані нормалізовано за допомогою MinMaxScaler")
        print(f"  Середнє значення після нормалізації: {self.df_normalized.mean().mean():.4f}")
        print(f"  Стандартне відхилення після нормалізації: {self.df_normalized.std().mean():.4f}\n")

    def find_optimal_clusters(self, max_k=10):
        """Крок 4: UMAP зменшення розмірності + HDBSCAN + аналіз K-Means"""
        print("=" * 60)
        print("КРОК 4: UMAP + HDBSCAN + оптимізація кластерів")
        print("=" * 60)

        # UMAP для зменшення розмірності
        print("Виконується UMAP зменшення розмірності...")
        umap_model = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=5,  # Більше компонент для кращого збереження структури
            metric='euclidean',
            random_state=42
        )
        self.umap_embedding = umap_model.fit_transform(self.df_normalized)
        print(f"✓ UMAP завершено: {self.df_normalized.shape[1]} ознак → {self.umap_embedding.shape[1]} компонент\n")

        # HDBSCAN для автоматичного визначення кластерів
        print("Виконується HDBSCAN кластеризація...")
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=30,
            min_samples=10,
            metric='euclidean',
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom'
        )
        hdbscan_labels = self.hdbscan_model.fit_predict(self.umap_embedding)

        # Аналіз результатів HDBSCAN
        n_clusters_hdbscan = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
        n_noise = list(hdbscan_labels).count(-1)

        print(f"✓ HDBSCAN завершено")
        print(f"  Знайдено кластерів: {n_clusters_hdbscan}")
        print(f"  Шумових точок (outliers): {n_noise} ({n_noise / len(hdbscan_labels) * 100:.1f}%)")

        if n_clusters_hdbscan > 0:
            valid_labels = hdbscan_labels[hdbscan_labels != -1]
            valid_data = self.umap_embedding[hdbscan_labels != -1]
            if len(valid_labels) > 0:
                hdbscan_silhouette = silhouette_score(valid_data, valid_labels)
                print(f"  Silhouette Score (без шуму): {hdbscan_silhouette:.4f}\n")

        # K-Means на UMAP embedding для порівняння
        print("Обчислення метрик K-Means на UMAP embedding...")
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.umap_embedding)

            inertias.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(self.umap_embedding, labels)
            silhouette_scores.append(silhouette_avg)

            db_score = davies_bouldin_score(self.umap_embedding, labels)
            davies_bouldin_scores.append(db_score)

            ch_score = calinski_harabasz_score(self.umap_embedding, labels)
            calinski_harabasz_scores.append(ch_score)

            print(f"  k={k}: Silhouette={silhouette_avg:.4f}, Inertia={kmeans.inertia_:.2f}")

        # Визначення оптимального k
        self.optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"\n✓ Оптимальна кількість кластерів (K-Means за Silhouette): {self.optimal_k}")
        print(f"  Максимальний Silhouette Score: {max(silhouette_scores):.4f}")
        print(f"  HDBSCAN рекомендує: {n_clusters_hdbscan} кластерів\n")

        return K_range, inertias, silhouette_scores, davies_bouldin_scores, calinski_harabasz_scores, n_clusters_hdbscan

    def create_clusters(self, n_clusters=None, method='kmeans'):
        """Крок 5: Створення фінальних кластерів"""
        print("=" * 60)
        print("КРОК 5: Створення кластерів")
        print("=" * 60)

        if method == 'hdbscan':
            print("Використовується HDBSCAN кластеризація...")
            labels = self.hdbscan_model.labels_
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            # Присвоюємо шумові точки до найближчого кластера
            if n_noise > 0:
                print(f"⚠ Знайдено {n_noise} шумових точок, присвоюються до найближчого кластера...")
                noise_mask = labels == -1
                if noise_mask.sum() > 0:
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=1)
                    nn.fit(self.umap_embedding[~noise_mask])
                    _, indices = nn.kneighbors(self.umap_embedding[noise_mask])
                    labels[noise_mask] = labels[~noise_mask][indices.flatten()]

            self.df_features['cluster'] = labels
            self.n_clusters_found = n_clusters_found

            print(f"✓ HDBSCAN кластеризація завершена")
            print(f"  Знайдено кластерів: {n_clusters_found}")

        else:  # kmeans
            if n_clusters is None:
                n_clusters = self.optimal_k
                print(f"Використовується оптимальна кількість кластерів: {n_clusters}")
            else:
                print(f"Використовується задана кількість кластерів: {n_clusters}")

            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = self.kmeans_model.fit_predict(self.umap_embedding)
            self.df_features['cluster'] = labels
            self.n_clusters_found = n_clusters

            print(f"✓ K-Means кластеризація завершена")

        # Метрики
        silhouette_avg = silhouette_score(self.umap_embedding, labels)
        db_score = davies_bouldin_score(self.umap_embedding, labels)
        ch_score = calinski_harabasz_score(self.umap_embedding, labels)

        print(f"  Silhouette Score: {silhouette_avg:.4f}")
        print(f"  Davies-Bouldin Index: {db_score:.4f} (нижче = краще)")
        print(f"  Calinski-Harabasz Score: {ch_score:.2f} (вище = краще)\n")

    def get_centroids(self):
        """Крок 6: Отримання центроїдів кластерів"""
        print("=" * 60)
        print("КРОК 6: Центроїди кластерів")
        print("=" * 60)

        # Обчислюємо центроїди як середні значення в оригінальному просторі
        centroids_list = []
        for cluster in sorted(self.df_features['cluster'].unique()):
            cluster_mask = self.df_features['cluster'] == cluster
            cluster_data = self.df_features[cluster_mask].drop(columns=['cluster'])
            centroid = cluster_data.mean()
            centroids_list.append(centroid)

        centroids_df = pd.DataFrame(centroids_list)
        centroids_df.index = [f"Кластер {i}" for i in range(len(centroids_df))]

        # Денормалізація
        centroids_denorm = self.scaler.inverse_transform(centroids_df)
        centroids_df = pd.DataFrame(
            centroids_denorm,
            columns=centroids_df.columns,
            index=centroids_df.index
        )

        print("Центроїди кластерів (денормалізовані значення):")
        print(centroids_df.round(2))
        print()

        # Збереження центроїдів у CSV
        centroids_df.to_csv('cluster_centroids.csv')
        print("✓ Центроїди збережено у 'cluster_centroids.csv'")

        # Збереження центроїдів у TXT
        with open('cluster_centroids.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ЦЕНТРОЇДИ КЛАСТЕРІВ КОРИСТУВАЧІВ (UMAP + HDBSCAN/K-Means)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Дата створення: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Кількість кластерів: {len(centroids_df)}\n")
            f.write(f"Кількість ознак: {len(centroids_df.columns)}\n\n")

            for idx, row in centroids_df.iterrows():
                f.write(f"{'─' * 80}\n")
                f.write(f"{idx}\n")
                f.write(f"{'─' * 80}\n")
                for col, val in row.items():
                    f.write(f"  {col:.<40} {val:>12.4f}\n")
                f.write("\n")

        print("✓ Центроїди збережено у 'cluster_centroids.txt'\n")

        return centroids_df

    def plot_metrics(self, K_range, inertias, silhouette_scores, davies_bouldin_scores, calinski_harabasz_scores):
        """Крок 7: Візуалізація метрик"""
        print("=" * 60)
        print("КРОК 7: Візуалізація метрик")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Elbow Method
        axes[0, 0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Кількість кластерів (k)', fontsize=12)
        axes[0, 0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        axes[0, 0].set_title('Elbow Method (K-Means on UMAP)', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(K_range)

        # Silhouette Score
        axes[0, 1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].axvline(x=self.optimal_k, color='green', linestyle='--',
                           label=f'Оптимальне k={self.optimal_k}', linewidth=2)
        axes[0, 1].set_xlabel('Кількість кластерів (k)', fontsize=12)
        axes[0, 1].set_ylabel('Silhouette Score', fontsize=12)
        axes[0, 1].set_title('Silhouette Score Method', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].set_xticks(K_range)

        # Davies-Bouldin Index
        axes[1, 0].plot(K_range, davies_bouldin_scores, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Кількість кластерів (k)', fontsize=12)
        axes[1, 0].set_ylabel('Davies-Bouldin Index (Lower is Better)', fontsize=12)
        axes[1, 0].set_title('Davies-Bouldin Index', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(K_range)

        # Calinski-Harabasz Score
        axes[1, 1].plot(K_range, calinski_harabasz_scores, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Кількість кластерів (k)', fontsize=12)
        axes[1, 1].set_ylabel('Calinski-Harabasz Score (Higher is Better)', fontsize=12)
        axes[1, 1].set_title('Calinski-Harabasz Score', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xticks(K_range)

        plt.tight_layout()
        plt.savefig('cluster_metrics.png', dpi=300, bbox_inches='tight')
        print("✓ Графіки збережено у файл 'cluster_metrics.png'")
        plt.show()
        print()

    def visualize_clusters(self):
        """Крок 8: Візуалізація кластерів на UMAP embedding"""
        print("=" * 60)
        print("КРОК 8: Візуалізація кластерів (UMAP)")
        print("=" * 60)

        # Використовуємо перші 2 компоненти UMAP для візуалізації
        # Якщо є 5 компонент, створюємо окремий 2D UMAP для візуалізації
        if self.umap_embedding.shape[1] > 2:
            print("Створення 2D UMAP для візуалізації...")
            umap_2d = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                metric='euclidean',
                random_state=42
            )
            embedding_2d = umap_2d.fit_transform(self.df_normalized)
        else:
            embedding_2d = self.umap_embedding

        print(f"✓ Використовується 2D UMAP проекція для візуалізації\n")

        # Створення графіку
        plt.figure(figsize=(14, 10))

        clusters = self.df_features['cluster'].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))

        for cluster, color in zip(sorted(clusters), colors):
            cluster_data = embedding_2d[self.df_features['cluster'] == cluster]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                        c=[color], label=f'Кластер {cluster}',
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        # Додавання центроїдів (в UMAP просторі)
        for cluster in sorted(clusters):
            cluster_mask = self.df_features['cluster'] == cluster
            centroid_2d = embedding_2d[cluster_mask].mean(axis=0)
            plt.scatter(centroid_2d[0], centroid_2d[1],
                        c='black', marker='X', s=300,
                        edgecolors='yellow', linewidth=2, zorder=5)

        plt.xlabel('UMAP Component 1', fontsize=12)
        plt.ylabel('UMAP Component 2', fontsize=12)
        plt.title('Візуалізація кластерів користувачів (UMAP)', fontsize=14, fontweight='bold')
        plt.legend(loc='best', ncol=2)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('clusters_umap.png', dpi=300, bbox_inches='tight')
        print("✓ Графік збережено у файл 'clusters_umap.png'")
        plt.show()
        print()

    def cluster_statistics(self, centroids_df):
        """Крок 9: Статистика по кластерах"""
        print("=" * 60)
        print("КРОК 9: Статистика по кластерах")
        print("=" * 60)

        total_users = len(self.df_features)

        print(f"Загальна кількість користувачів: {total_users}\n")

        for cluster in sorted(self.df_features['cluster'].unique()):
            cluster_size = len(self.df_features[self.df_features['cluster'] == cluster])
            cluster_percentage = (cluster_size / total_users) * 100

            print(f"{'─' * 60}")
            print(f"КЛАСТЕР {cluster}")
            print(f"{'─' * 60}")
            print(f"Кількість користувачів: {cluster_size}")
            print(f"Відсоток від загальної кількості: {cluster_percentage:.2f}%")
            print(f"\nЦентроїд кластера {cluster}:")
            print(centroids_df.iloc[cluster].round(2))
            print()

        # Загальна статистика по ознаках в кластерах
        print(f"{'=' * 60}")
        print("ПОРІВНЯЛЬНА СТАТИСТИКА ПО КЛАСТЕРАХ")
        print(f"{'=' * 60}")

        cluster_stats = self.df_features.groupby('cluster').mean()
        print("\nСередні значення ознак по кластерах:")
        print(cluster_stats.round(2))
        print()

        return cluster_stats


# Основна функція для запуску всього процесу
def main():
    """Головна функція для виконання сегментації користувачів"""

    # Шлях до файлу (змініть на свій)
    filepath = 'user_features.csv'  # Замініть на шлях до вашого файлу

    # Створення об'єкта для сегментації
    segmentation = UserSegmentation(filepath)

    # Виконання всіх кроків
    segmentation.load_and_clean_data()
    segmentation.df_cleaned.to_csv('cleaned.csv')
    segmentation.feature_selection()
    segmentation.normalize_data()

    K_range, inertias, silhouette_scores, db_scores, ch_scores, n_hdbscan = segmentation.find_optimal_clusters(max_k=20)

    # Запит методу та кількості кластерів від користувача
    print("=" * 60)
    print(f"HDBSCAN знайшов {n_hdbscan} кластерів автоматично")
    print(f"K-Means рекомендує {segmentation.optimal_k} кластерів (за Silhouette Score)")
    print()
    method_input = input("Виберіть метод (1=HDBSCAN, 2=K-Means, Enter=K-Means): ").strip()

    if method_input == '1':
        method = 'hdbscan'
        print("\n✓ Використовується HDBSCAN")
        segmentation.create_clusters(method='hdbscan')
    else:
        method = 'kmeans'
        user_input = input(f"Введіть кількість кластерів (Enter для {segmentation.optimal_k}): ")
        n_clusters = int(user_input) if user_input.strip() else None
        print(f"\n✓ Використовується K-Means з {n_clusters if n_clusters else segmentation.optimal_k} кластерами")
        segmentation.create_clusters(n_clusters, method='kmeans')

    print("=" * 60)
    print()

    centroids_df = segmentation.get_centroids()
    segmentation.plot_metrics(K_range, inertias, silhouette_scores, db_scores, ch_scores)
    segmentation.visualize_clusters()
    segmentation.cluster_statistics(centroids_df)

    print("=" * 60)
    print("СЕГМЕНТАЦІЯ ЗАВЕРШЕНА УСПІШНО!")
    print("=" * 60)
    print("\nФайли збережено:")
    print("  • cluster_metrics.png - графіки метрик (4 метрики)")
    print("  • clusters_umap.png - UMAP візуалізація кластерів")
    print("  • cluster_centroids.csv - центроїди у CSV форматі")
    print("  • cluster_centroids.txt - центроїди у текстовому форматі")
    print("\nМетоди використані:")
    print("  • UMAP для зменшення розмірності")
    print("  • HDBSCAN для автоматичного визначення кластерів")
    print("  • K-Means на UMAP embedding для стабільних сегментів")
    print("  • MinMaxScaler для нормалізації")
    print("  • 4 метрики якості кластеризації")


if __name__ == "__main__":
    main()