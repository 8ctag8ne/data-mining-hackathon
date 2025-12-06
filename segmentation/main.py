import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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
        self.kmeans_model = None
        self.optimal_k = None
        self.scaler = MinMaxScaler()
        
    def load_and_clean_data(self):
        """Крок 1: Завантаження та очищення датасету"""
        print("="*60)
        print("КРОК 1: Завантаження та очищення датасету")
        print("="*60)
        
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

        no_messages = self.df_cleaned['time_to_first_message_sec'].isnull()
        max_time = self.df_cleaned['time_to_first_message_sec'].max()

        self.df_cleaned.loc[no_messages, 'time_to_first_message_sec'] = max_time * 2

        # self.df_cleaned = self.df_cleaned.dropna(subset=['time_to_first_message_sec'])
        
        for col in numeric_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                self.df_cleaned[col].fillna(self.df_cleaned[col].median(), inplace=True)
        
        print(f"✓ Заповнено пропущених значень: {missing_before}")
        
        # Видалення викидів за допомогою IQR
        mask = np.ones(len(self.df_cleaned), dtype=bool)
        len_before = len(self.df_cleaned)

        # for col in numeric_cols:
        #     if col not in ['user_id', 'is_churned']:
        #         Q1 = self.df_cleaned[col].quantile(0.25)
        #         Q3 = self.df_cleaned[col].quantile(0.75)
        #         IQR = Q3 - Q1
        #         before = len(self.df_cleaned)
        #         lower = Q1 - 1.5*IQR
        #         upper = Q3 + 1.5*IQR
        #
        #         mask &= (self.df_cleaned[col] >= lower) & (self.df_cleaned[col] <= upper)
        #
        # self.df_cleaned = self.df_cleaned[mask]
        
        print(f"✓ Видалено викидів (outliers): {len_before - len(self.df_cleaned)}")
        print(f"✓ Залишилось записів після очищення: {len(self.df_cleaned)}\n")
        
    def feature_selection(self):
        """Крок 2: Відбір релевантних ознак"""
        print("="*60)
        print("КРОК 2: Відбір релевантних ознак")
        print("="*60)
        
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
        print("="*60)
        print("КРОК 3: Нормалізація даних")
        print("="*60)
        
        self.df_normalized = pd.DataFrame(
            self.scaler.fit_transform(self.df_features),
            columns=self.df_features.columns,
            index=self.df_features.index
        )
        
        print("✓ Дані нормалізовано за допомогою StandardScaler")
        print(f"  Середнє значення після нормалізації: {self.df_normalized.mean().mean():.4f}")
        print(f"  Стандартне відхилення після нормалізації: {self.df_normalized.std().mean():.4f}\n")
        
    def find_optimal_clusters(self, max_k=10):
        """Крок 4: Визначення оптимальної кількості кластерів"""
        print("="*60)
        print("КРОК 4: Визначення оптимальної кількості кластерів")
        print("="*60)
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_k + 1)
        
        print("Обчислення метрик для різної кількості кластерів...")
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.df_normalized)
            inertias.append(kmeans.inertia_)
            
            silhouette_avg = silhouette_score(self.df_normalized, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
            print(f"  k={k}: Silhouette Score = {silhouette_avg:.4f}, Inertia = {kmeans.inertia_:.2f}")
        
        # Знаходження оптимального k за silhouette score
        self.optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"\n✓ Оптимальна кількість кластерів (за Silhouette Score): {self.optimal_k}")
        print(f"  Максимальний Silhouette Score: {max(silhouette_scores):.4f}\n")
        
        return K_range, inertias, silhouette_scores
    
    def create_clusters(self, n_clusters=None):
        """Крок 5: Створення кластерів"""
        print("="*60)
        print("КРОК 5: Створення кластерів")
        print("="*60)
        
        if n_clusters is None:
            n_clusters = self.optimal_k
            print(f"Використовується оптимальна кількість кластерів: {n_clusters}")
        else:
            print(f"Використовується задана кількість кластерів: {n_clusters}")
        
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df_features['cluster'] = self.kmeans_model.fit_predict(self.df_normalized)
        
        silhouette_avg = silhouette_score(self.df_normalized, self.kmeans_model.labels_)
        print(f"✓ Кластеризація завершена")
        print(f"  Silhouette Score: {silhouette_avg:.4f}\n")
        
    def get_centroids(self):
        """Крок 6: Отримання центроїдів кластерів"""
        print("="*60)
        print("КРОК 6: Центроїди кластерів")
        print("="*60)
        
        # Денормалізація центроїдів
        centroids_normalized = self.kmeans_model.cluster_centers_
        centroids = self.scaler.inverse_transform(centroids_normalized)
        
        centroids_df = pd.DataFrame(
            centroids,
            columns=self.df_features.columns[:-1],  # без колонки 'cluster'
            index=[f"Кластер {i}" for i in range(len(centroids))]
        )
        
        print("Центроїди кластерів (денормалізовані значення):")
        print(centroids_df.round(2))
        print()
        
        # Збереження центроїдів у CSV
        centroids_df.to_csv('cluster_centroids.csv')
        print("✓ Центроїди збережено у 'cluster_centroids.csv'")
        
        # Збереження центроїдів у TXT (зручний для читання формат)
        with open('cluster_centroids.txt', 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ЦЕНТРОЇДИ КЛАСТЕРІВ КОРИСТУВАЧІВ\n")
            f.write("="*80 + "\n\n")
            f.write(f"Дата створення: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Кількість кластерів: {len(centroids_df)}\n")
            f.write(f"Кількість ознак: {len(centroids_df.columns)}\n\n")
            
            for idx, row in centroids_df.iterrows():
                f.write(f"{'─'*80}\n")
                f.write(f"{idx}\n")
                f.write(f"{'─'*80}\n")
                for col, val in row.items():
                    f.write(f"  {col:.<40} {val:>12.4f}\n")
                f.write("\n")
        
        print("✓ Центроїди збережено у 'cluster_centroids.txt'\n")
        
        return centroids_df
    
    def plot_metrics(self, K_range, inertias, silhouette_scores):
        """Крок 7: Візуалізація метрик"""
        print("="*60)
        print("КРОК 7: Візуалізація метрик")
        print("="*60)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow Method
        axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Кількість кластерів (k)', fontsize=12)
        axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(K_range)
        axes[0].set_ylim(bottom=0)  # Починаємо з нуля
        
        # Silhouette Score
        axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        axes[1].axvline(x=self.optimal_k, color='green', linestyle='--', 
                       label=f'Оптимальне k={self.optimal_k}', linewidth=2)
        axes[1].set_xlabel('Кількість кластерів (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Score Method', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_xticks(K_range)
        axes[1].set_ylim(bottom=0)  # Починаємо з нуля
        
        plt.tight_layout()
        plt.savefig('cluster_metrics.png', dpi=300, bbox_inches='tight')
        print("✓ Графіки збережено у файл 'cluster_metrics.png'")
        plt.show()
        print()
        
    def visualize_clusters(self):
        """Крок 8: PCA візуалізація кластерів"""
        print("="*60)
        print("КРОК 8: Візуалізація кластерів (PCA)")
        print("="*60)
        
        # PCA для зменшення до 2 вимірів
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(self.df_normalized)
        
        explained_variance = pca.explained_variance_ratio_
        print(f"✓ PCA завершено")
        print(f"  Пояснена дисперсія PC1: {explained_variance[0]:.2%}")
        print(f"  Пояснена дисперсія PC2: {explained_variance[1]:.2%}")
        print(f"  Загальна пояснена дисперсія: {sum(explained_variance):.2%}\n")
        
        # Створення графіку
        plt.figure(figsize=(12, 8))
        
        clusters = self.df_features['cluster'].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
        
        for cluster, color in zip(sorted(clusters), colors):
            cluster_data = pca_features[self.df_features['cluster'] == cluster]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                       c=[color], label=f'Кластер {cluster}', 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Додавання центроїдів
        centroids_pca = pca.transform(self.kmeans_model.cluster_centers_)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                   c='black', marker='X', s=300, 
                   edgecolors='yellow', linewidth=2, 
                   label='Центроїди', zorder=5)
        
        plt.xlabel(f'Головна компонента 1 ({explained_variance[0]:.1%} дисперсії)', fontsize=12)
        plt.ylabel(f'Головна компонента 2 ({explained_variance[1]:.1%} дисперсії)', fontsize=12)
        plt.title('Візуалізація кластерів користувачів (PCA)', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('clusters_pca.png', dpi=300, bbox_inches='tight')
        print("✓ Графік збережено у файл 'clusters_pca.png'")
        plt.show()
        print()
        
    def cluster_statistics(self, centroids_df):
        """Крок 9: Статистика по кластерах"""
        print("="*60)
        print("КРОК 9: Статистика по кластерах")
        print("="*60)
        
        total_users = len(self.df_features)
        
        print(f"Загальна кількість користувачів: {total_users}\n")
        
        for cluster in sorted(self.df_features['cluster'].unique()):
            cluster_size = len(self.df_features[self.df_features['cluster'] == cluster])
            cluster_percentage = (cluster_size / total_users) * 100
            
            print(f"{'─'*60}")
            print(f"КЛАСТЕР {cluster}")
            print(f"{'─'*60}")
            print(f"Кількість користувачів: {cluster_size}")
            print(f"Відсоток від загальної кількості: {cluster_percentage:.2f}%")
            print(f"\nЦентроїд кластера {cluster}:")
            print(centroids_df.iloc[cluster].round(2))
            print()
        
        # Загальна статистика по ознаках в кластерах
        print(f"{'='*60}")
        print("ПОРІВНЯЛЬНА СТАТИСТИКА ПО КЛАСТЕРАХ")
        print(f"{'='*60}")
        
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
    segmentation.feature_selection()
    segmentation.normalize_data()
    
    K_range, inertias, silhouette_scores = segmentation.find_optimal_clusters(max_k=10)
    
    # Запит кількості кластерів від користувача
    print("="*60)
    user_input = input(f"Введіть бажану кількість кластерів (Enter для {segmentation.optimal_k}): ")
    n_clusters = int(user_input) if user_input.strip() else None
    print("="*60)
    print()
    
    segmentation.create_clusters(n_clusters)
    centroids_df = segmentation.get_centroids()
    segmentation.plot_metrics(K_range, inertias, silhouette_scores)
    segmentation.visualize_clusters()
    segmentation.cluster_statistics(centroids_df)
    
    print("="*60)
    print("СЕГМЕНТАЦІЯ ЗАВЕРШЕНА УСПІШНО!")
    print("="*60)
    print("\nФайли збережено:")
    print("  • cluster_metrics.png - графіки метрик")
    print("  • clusters_pca.png - візуалізація кластерів")
    print("  • cluster_centroids.csv - центроїди у CSV форматі")
    print("  • cluster_centroids.txt - центроїди у текстовому форматі")


if __name__ == "__main__":
    main()