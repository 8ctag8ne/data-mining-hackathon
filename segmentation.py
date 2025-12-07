import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import umap
import hdbscan
import warnings
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')


class UserSegmentation:
    def __init__(self,
                 filepath,
                 out_dir='segmentation_outputs',
                 remove_outliers=True,
                 outlier_iqr=1.5,
                 random_state=42):
        self.filepath = filepath
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.df_original = None
        self.df_cleaned = None
        self.df_features = None
        self.df_normalized = None

        self.umap_model = None
        self.umap_embedding = None

        self.hdbscan_model = None
        self.kmeans_model = None

        self.optimal_k = None
        self.scaler = MinMaxScaler()
        self.n_clusters_found = None

        self.remove_outliers = remove_outliers
        self.outlier_iqr = outlier_iqr
        self.random_state = random_state

    # -----------------------
    # Utility functions
    # -----------------------
    @staticmethod
    def _safe_normalize_text(s: str):
        if pd.isna(s) or s == '':
            return []
        # нормалізація: lower, replace нелітеральні символи на "-",
        # розділення комами, ; або | або /
        s = re.sub(r"[\/;|]+", ",", str(s))
        items = [re.sub(r'[^a-z0-9\- ]', '', x.strip().lower()).replace(' ', '-') for x in s.split(',') if x.strip()]
        return [it for it in items if it]

    def _assign_categories(self, df, source_col, categories):
        # categories: dict[new_col] = list_of_keywords
        # для кожного елементу — перевіряємо чи хоча б один keyword є підрядком
        for new_col, keywords in categories.items():
            keys_norm = [k.lower().replace(' ', '-').strip() for k in keywords]
            df[new_col] = df[source_col].apply(
                lambda lst: int(any(any(k in item for k in keys_norm) for item in lst))
            )
        return df

    def _remove_outliers_iqr(self, df, numeric_cols):
        # IQR method
        df_out = df.copy()
        mask = pd.Series(True, index=df.index)
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            low = q1 - self.outlier_iqr * iqr
            high = q3 + self.outlier_iqr * iqr
            mask &= df[col].between(low, high, inclusive='both')
        removed = (~mask).sum()
        if removed > 0:
            df_out = df_out[mask]
        return df_out, removed

    # -----------------------
    # Step 1: Load & clean
    # -----------------------
    def load_and_clean_data(self):
        print("=" * 60)
        print("КРОК 1: Завантаження та очищення датасету")
        print("=" * 60)

        self.df_original = pd.read_csv(self.filepath)
        print(f"✓ Завантажено {len(self.df_original)} записів, {len(self.df_original.columns)} колонок")

        self.df_cleaned = self.df_original.copy()

        # Дублі
        duplicates = self.df_cleaned.duplicated().sum()
        if duplicates:
            self.df_cleaned = self.df_cleaned.drop_duplicates()
        print(f"✓ Видалено дублікатів: {duplicates}")

        # Numeric columns
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns.tolist()

        # Спеціальна обробка time_to_first_message_sec
        if 'time_to_first_message_sec' in self.df_cleaned.columns:
            t_col = 'time_to_first_message_sec'
            if self.df_cleaned[t_col].isna().all():
                # якщо все NaN — підставимо медіану інших часів або фіксоване велике значення
                fallback = 3600 * 24  # 1 day, як приклад
                print("⚠ Усі значення time_to_first_message_sec пусті — підставлено fallback (1 day).")
                self.df_cleaned[t_col] = self.df_cleaned[t_col].fillna(fallback)
            else:
                max_time = self.df_cleaned[t_col].max(skipna=True)
                # якщо max_time NaN (невірно), використати 99-й перцентиль
                if pd.isna(max_time):
                    max_time = self.df_cleaned[t_col].quantile(0.99)
                no_messages = self.df_cleaned[t_col].isnull()
                self.df_cleaned.loc[no_messages, t_col] = float(max_time) * 2.0
        else:
            print("⚠ Колонки 'time_to_first_message_sec' немає у датасеті — пропускаємо спеціальну обробку.")

        # Заповнення числових полів медианою
        missing_before = self.df_cleaned.isnull().sum().sum()
        for col in numeric_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                self.df_cleaned[col].fillna(self.df_cleaned[col].median(), inplace=True)
        print(f"✓ Заповнено пропущених значень (числові): {missing_before}")

        # MULTI-HOT ENCODING З АГРЕГАЦІЄЮ У КАТЕГОРІЇ
        print("✓ Виконується категоризація goals / interests / assistance...")

        # Перевіряємо наявність колонок
        for col in ['user_goals', 'user_interests', 'user_assistance']:
            if col not in self.df_cleaned.columns:
                self.df_cleaned[col] = np.nan
                print(f"  ⚠ Колонка {col} відсутня — створено порожню.")

        self.df_cleaned["goals_list"] = self.df_cleaned["user_goals"].apply(self._safe_normalize_text)
        self.df_cleaned["interests_list"] = self.df_cleaned["user_interests"].apply(self._safe_normalize_text)
        self.df_cleaned["assistance_list"] = self.df_cleaned["user_assistance"].apply(self._safe_normalize_text)

        GOALS_CATEGORIES = {
            "goals_creative": ["content-creation", "essay-writing", "cooking", "content creation"],
            "goals_professional": ["coding-assistance", "business-purposes", "coding"],
            "goals_personal_dev": ["education", "mental-health", "self-improvement"],
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

        # assign categories
        self.df_cleaned = self._assign_categories(self.df_cleaned, "goals_list", GOALS_CATEGORIES)
        self.df_cleaned = self._assign_categories(self.df_cleaned, "assistance_list", ASSISTANCE_CATEGORIES)
        self.df_cleaned = self._assign_categories(self.df_cleaned, "interests_list", INTEREST_CATEGORIES)

        # drop raw cols
        drop_cols = ["goals_list", "interests_list", "assistance_list",
                     "user_goals", "user_interests", "user_assistance"]
        for c in drop_cols:
            if c in self.df_cleaned.columns:
                self.df_cleaned.drop(columns=[c], inplace=True)

        # Видалення викидів (IQR) — опційно
        len_before = len(self.df_cleaned)
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        if self.remove_outliers and len(numeric_cols) > 0:
            self.df_cleaned, removed = self._remove_outliers_iqr(self.df_cleaned, numeric_cols)
            print(f"✓ Видалено викидів (IQR): {removed}")
        else:
            print("✓ Видалення викидів пропущене (remove_outliers=False або відсутні числові колонки).")

        print(f"✓ Залишилось записів після очищення: {len(self.df_cleaned)}\n")

    # -----------------------
    # Step 2: Feature selection
    # -----------------------
    def feature_selection(self, drop_low_variance_threshold=0.01, high_corr_threshold=0.9):
        print("=" * 60)
        print("КРОК 2: Відбір релевантних ознак")
        print("=" * 60)

        self.df_features = self.df_cleaned.copy()

        # remove identifiers
        columns_to_remove = ['user_id', 'model_changes']
        self.df_features = self.df_features.drop(columns=columns_to_remove, errors='ignore')
        print(f"✓ Видалено ідентифікатори (якщо були): {columns_to_remove}")

        numeric_cols = self.df_features.select_dtypes(include=[np.number]).columns.tolist()

        # low variance
        low_variance_cols = [col for col in numeric_cols if self.df_features[col].std() < drop_low_variance_threshold]
        if low_variance_cols:
            self.df_features.drop(columns=low_variance_cols, inplace=True)
            print(f"✓ Видалено колонок з низькою варіативністю: {low_variance_cols}")
        else:
            print("✓ Колонок з низькою варіативністю не знайдено")

        # high correlation
        numeric_cols = self.df_features.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr = self.df_features[numeric_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > high_corr_threshold)]
            if to_drop:
                self.df_features.drop(columns=to_drop, inplace=True)
                print(f"✓ Видалено високо корельованих колонок (r>{high_corr_threshold}): {to_drop}")
            else:
                print("✓ Високо корельованих колонок не знайдено")
        else:
            print("✓ Недостатньо числових колонок для перевірки кореляції")

        print(f"✓ Залишилось ознак для кластеризації: {len(self.df_features.columns)}")
        print(f"  Ознаки: {list(self.df_features.columns)}\n")

    # -----------------------
    # Step 3: Normalize
    # -----------------------
    def normalize_data(self):
        print("=" * 60)
        print("КРОК 3: Нормалізація даних")
        print("=" * 60)

        # Зберігаємо лише числові колонки в scaler'і
        numeric_cols = self.df_features.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("Немає числових колонок для нормалізації.")

        self.df_features = self.df_features[numeric_cols]  # відкидаємо нечислові (якщо є)
        self.df_normalized = pd.DataFrame(self.scaler.fit_transform(self.df_features),
                                          columns=self.df_features.columns,
                                          index=self.df_features.index)
        print("✓ Дані нормалізовано за допомогою MinMaxScaler")
        print(f"  Середнє після нормалізації: {self.df_normalized.mean().mean():.4f}")
        print()

    # -----------------------
    # Step 4: UMAP + HDBSCAN + KMeans metrics
    # -----------------------
    def find_optimal_clusters(self, umap_n_neighbors=15, umap_min_dist=0.1, umap_n_components=5, max_k=10,
                              hdbscan_min_cluster_size=30, hdbscan_min_samples=10):
        print("=" * 60)
        print("КРОК 4: UMAP + HDBSCAN + оптимізація кластерів")
        print("=" * 60)

        # UMAP
        print("Виконується UMAP зменшення розмірності...")
        self.umap_model = umap.UMAP(n_neighbors=umap_n_neighbors,
                                    min_dist=umap_min_dist,
                                    n_components=umap_n_components,
                                    metric='euclidean',
                                    random_state=self.random_state)
        self.umap_embedding = self.umap_model.fit_transform(self.df_normalized)
        print(f"✓ UMAP завершено: {self.df_normalized.shape[1]} ознак → {self.umap_embedding.shape[1]} компонент\n")

        # HDBSCAN
        print("Виконується HDBSCAN кластеризація...")
        self.hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
                                             min_samples=hdbscan_min_samples,
                                             metric='euclidean',
                                             cluster_selection_method='eom')
        hdbscan_labels = self.hdbscan_model.fit_predict(self.umap_embedding)

        n_clusters_hdbscan = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
        n_noise = (hdbscan_labels == -1).sum()
        print(f"✓ HDBSCAN завершено — знайдено {n_clusters_hdbscan} кластерів, шумових точок: {n_noise} ({n_noise/len(hdbscan_labels)*100:.1f}%)")

        if n_clusters_hdbscan >= 1 and (hdbscan_labels != -1).sum() >= 2:
            try:
                valid_labels = hdbscan_labels[hdbscan_labels != -1]
                valid_data = self.umap_embedding[hdbscan_labels != -1]
                hdb_sil = silhouette_score(valid_data, valid_labels)
                print(f"  Silhouette (HDBSCAN без шуму): {hdb_sil:.4f}")
            except Exception as e:
                print("  ⚠ Не вдалося обчислити silhouette для HDBSCAN:", e)

        # KMeans metrics on UMAP embedding
        print("\nОбчислення метрик K-Means на UMAP embedding...")
        inertias, silhouette_scores, db_scores, ch_scores = [], [], [], []
        K_range = list(range(2, max_k + 1))
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(self.umap_embedding)

            inertias.append(kmeans.inertia_)
            try:
                s = silhouette_score(self.umap_embedding, labels)
            except Exception:
                s = -1.0
            silhouette_scores.append(s)

            try:
                db = davies_bouldin_score(self.umap_embedding, labels)
            except Exception:
                db = np.nan
            db_scores.append(db)

            try:
                ch = calinski_harabasz_score(self.umap_embedding, labels)
            except Exception:
                ch = np.nan
            ch_scores.append(ch)

            print(f"  k={k}: Silhouette={s:.4f}, Inertia={kmeans.inertia_:.2f}")

        # optimal k by silhouette (safe)
        if any([s > -1 for s in silhouette_scores]):
            self.optimal_k = K_range[int(np.nanargmax(silhouette_scores))]
        else:
            self.optimal_k = K_range[0]
        print(f"\n✓ Оптимальна кількість кластерів (K-Means за Silhouette): {self.optimal_k}")
        print(f"  HDBSCAN рекомендує: {n_clusters_hdbscan} кластерів\n")

        return K_range, inertias, silhouette_scores, db_scores, ch_scores, n_clusters_hdbscan

    # -----------------------
    # Step 5: Create clusters
    # -----------------------
    def create_clusters(self, n_clusters=None, method='kmeans'):
        print("=" * 60)
        print("КРОК 5: Створення кластерів")
        print("=" * 60)

        if method == 'hdbscan':
            if self.hdbscan_model is None:
                raise RuntimeError("HDBSCAN модель не натренована. Викличте find_optimal_clusters() спочатку.")
            labels = self.hdbscan_model.labels_.copy()
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()

            # присвоєння шумових точок до найближчих
            if n_noise > 0:
                print(f"⚠ Знайдено {n_noise} шумових точок — присвоюємо до найближчого кластера...")
                noise_mask = labels == -1
                if noise_mask.sum() > 0:
                    nn = NearestNeighbors(n_neighbors=1)
                    nn.fit(self.umap_embedding[~noise_mask])
                    _, indices = nn.kneighbors(self.umap_embedding[noise_mask])
                    labels[noise_mask] = labels[~noise_mask][indices.flatten()]

            self.df_features['cluster'] = labels
            self.n_clusters_found = n_clusters_found
            print(f"✓ HDBSCAN кластеризація завершена — знайдено {n_clusters_found} кластерів")

        else:
            if n_clusters is None:
                n_clusters = self.optimal_k
            print(f"✓ Виконуємо KMeans з n_clusters={n_clusters}")
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = self.kmeans_model.fit_predict(self.umap_embedding)
            self.df_features['cluster'] = labels
            self.n_clusters_found = n_clusters
            print("✓ KMeans завершено")

        # метрики (перевірки)
        if len(set(labels)) >= 2 and len(labels) > 1:
            try:
                sil = silhouette_score(self.umap_embedding, labels)
            except Exception:
                sil = np.nan
            try:
                db = davies_bouldin_score(self.umap_embedding, labels)
            except Exception:
                db = np.nan
            try:
                ch = calinski_harabasz_score(self.umap_embedding, labels)
            except Exception:
                ch = np.nan
            print(f"  Silhouette: {sil:.4f}  Davies-Bouldin: {db:.4f}  Calinski-Harabasz: {ch:.2f}\n")
        else:
            print("  ⚠ Недостатньо кластерів / зразків для обчислення метрик\n")

    # -----------------------
    # Step 6: Centroids
    # -----------------------
    def get_centroids(self):
        print("=" * 60)
        print("КРОК 6: Центроїди кластерів")
        print("=" * 60)

        if 'cluster' not in self.df_features.columns:
            raise RuntimeError("Колонка 'cluster' не знайдена. Створіть кластери спочатку.")

        centroids_list = []
        cluster_labels = sorted(self.df_features['cluster'].unique())
        for cluster in cluster_labels:
            mask = self.df_features['cluster'] == cluster
            cluster_data = self.df_features.loc[mask].drop(columns=['cluster'])
            centroid = cluster_data.mean()
            centroids_list.append(centroid)

        centroids_df = pd.DataFrame(centroids_list, index=[f"Кластер {c}" for c in cluster_labels])

        # Денормалізація: перевірка порядку колонок перед inverse_transform
        cols = centroids_df.columns.tolist()
        scaler_cols = list(self.df_normalized.columns)
        if cols != scaler_cols:
            # якщо порядок відрізняється, переставимо
            centroids_df = centroids_df[scaler_cols]

        centroids_denorm = self.scaler.inverse_transform(centroids_df)
        centroids_df = pd.DataFrame(centroids_denorm, columns=scaler_cols, index=centroids_df.index)

        # Збереження
        csv_path = os.path.join(self.out_dir, 'cluster_centroids.csv')
        txt_path = os.path.join(self.out_dir, 'cluster_centroids.txt')
        centroids_df.round(4).to_csv(csv_path)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ЦЕНТРОЇДИ КЛАСТЕРІВ\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Дата: {pd.Timestamp.now()}\n")
            f.write(f"Кількість кластерів: {len(centroids_df)}\n\n")
            for idx, row in centroids_df.iterrows():
                f.write(f"{'-'*40}\n{idx}\n{'-'*40}\n")
                for col, val in row.items():
                    f.write(f"{col:.<40} {val:>12.4f}\n")
                f.write("\n")

        print(f"✓ Центроїди збережено: {csv_path}, {txt_path}\n")
        return centroids_df

    # -----------------------
    # Step 7: Plot metrics
    # -----------------------
    def plot_metrics(self, K_range, inertias, silhouette_scores, davies_bouldin_scores, calinski_harabasz_scores):
        print("=" * 60)
        print("КРОК 7: Візуалізація метрик")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes[0, 0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Elbow Method (KMeans on UMAP)')
        axes[0, 0].set_xticks(K_range)

        axes[0, 1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        if self.optimal_k:
            axes[0, 1].axvline(x=self.optimal_k, color='green', linestyle='--',
                               label=f'Оптимальне k={self.optimal_k}', linewidth=2)
            axes[0, 1].legend()
        axes[0, 1].set_title('Silhouette Score')

        axes[1, 0].plot(K_range, davies_bouldin_scores, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Davies-Bouldin Index')

        axes[1, 1].plot(K_range, calinski_harabasz_scores, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Calinski-Harabasz Score')

        plt.tight_layout()
        path = os.path.join(self.out_dir, 'cluster_metrics.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✓ Графіки збережено у {path}")
        plt.show()

    # -----------------------
    # Step 8: Visualize clusters
    # -----------------------
    def visualize_clusters(self, use_existing_umap=True):
        print("=" * 60)
        print("КРОК 8: Візуалізація кластерів (UMAP)")
        print("=" * 60)

        if use_existing_umap and self.umap_embedding is not None and self.umap_embedding.shape[1] >= 2:
            emb2d = self.umap_embedding[:, :2]
        else:
            # тренуємо окремий 2D UMAP на тих самих нормалізованих даних
            umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=self.random_state)
            emb2d = umap_2d.fit_transform(self.df_normalized)

        clusters = self.df_features['cluster'].unique()
        plt.figure(figsize=(14, 10))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))

        for cluster, color in zip(sorted(clusters), colors):
            mask = self.df_features['cluster'] == cluster
            data = emb2d[mask.values]
            plt.scatter(data[:, 0], data[:, 1], label=f'Кластер {cluster}', alpha=0.6, s=50, edgecolors='black')

        # центроїди (UMAP-простір)
        for cluster in sorted(clusters):
            mask = self.df_features['cluster'] == cluster
            centroid = emb2d[mask.values].mean(axis=0)
            plt.scatter(centroid[0], centroid[1], c='black', marker='X', s=200, edgecolors='yellow', linewidth=2, zorder=5)

        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.title('Візуалізація кластерів (UMAP)')
        plt.legend(loc='best', ncol=2)
        plt.grid(alpha=0.3)
        path = os.path.join(self.out_dir, 'clusters_umap.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✓ Графік збережено у {path}")
        plt.show()

    # -----------------------
    # Step 9: Cluster statistics
    # -----------------------
    def cluster_statistics(self, centroids_df):
        print("=" * 60)
        print("КРОК 9: Статистика по кластерах")
        print("=" * 60)

        total = len(self.df_features)
        print(f"Загальна кількість користувачів: {total}\n")

        stats = {}
        for cluster in sorted(self.df_features['cluster'].unique()):
            mask = self.df_features['cluster'] == cluster
            size = mask.sum()
            pct = 100.0 * size / total if total else 0
            print(f"Кластер {cluster}: розмір={size}, відсоток={pct:.2f}%")
            print("Центроїд (денормалізований):")
            print(centroids_df.loc[f"Кластер {cluster}"].round(2))
            print()
            stats[cluster] = {'size': int(size), 'pct': pct}

        cluster_stats = self.df_features.groupby('cluster').mean().round(3)
        print("Середні значення ознак по кластерах (normalized space):")
        print(cluster_stats)
        return cluster_stats


# -----------------------
# Example main runner
# -----------------------
def main():
    filepath = 'user_features_2.csv'  # змініть під свій шлях
    seg = UserSegmentation(filepath, out_dir='seg_out', remove_outliers=True)

    seg.load_and_clean_data()
    seg.df_cleaned.to_csv(os.path.join(seg.out_dir, 'cleaned.csv'), index=False)

    seg.feature_selection()
    seg.normalize_data()

    K_range, inertias, sils, dbs, chs, n_hdbscan = seg.find_optimal_clusters(max_k=20)
    print(f"HDBSCAN знайшов {n_hdbscan} кластерів, K-Means рекомендує {seg.optimal_k}")

    # Default: KMeans (можна змінити на 'hdbscan')
    seg.create_clusters(method='kmeans', n_clusters=seg.optimal_k)

    centroids = seg.get_centroids()
    seg.plot_metrics(K_range, inertias, sils, dbs, chs)
    seg.visualize_clusters()
    seg.cluster_statistics(centroids)

    print("СЕГМЕНТАЦІЯ ЗАВЕРШИЛАСЬ. Артефакти у папці:", seg.out_dir)


if __name__ == "__main__":
    main()
