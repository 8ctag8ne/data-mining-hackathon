import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
import warnings

warnings.filterwarnings('ignore')

# Налаштування візуалізації
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ChurnPredictionModel:
    def __init__(self, filepath, use_cleaned=False, use_categories=True):
        """
        Ініціалізація моделі передбачення churn

        Parameters:
        -----------
        filepath : str
            Шлях до CSV файлу
        use_cleaned : bool
            True - використати частково оброблений датасет (з категоріями)
            False - використати оригінальний датасет
        use_categories : bool
            True - групувати ознаки за категоріями (менше ознак)
            False - multi-hot encoding для кожного унікального значення (більше ознак)
        """
        self.filepath = filepath
        self.use_cleaned = use_cleaned
        self.use_categories = use_categories
        self.df_original = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = RobustScaler()
        self.model = None
        self.feature_importance = None

    def load_and_prepare_data(self):
        """Крок 1: Завантаження та підготовка даних"""
        print("=" * 70)
        print("КРОК 1: Завантаження та підготовка даних")
        print("=" * 70)

        # Завантаження
        self.df_original = pd.read_csv(self.filepath)
        print(f"✓ Завантажено {len(self.df_original)} записів")
        print(f"✓ Кількість колонок: {len(self.df_original.columns)}")

        self.df_processed = self.df_original.copy()

        # Фільтрація: залишити тільки користувачів з successful_purchase = 1
        if 'successful_purchase' in self.df_processed.columns:
            records_before = len(self.df_processed)
            self.df_processed = self.df_processed[self.df_processed['successful_purchase'] == 1].copy()
            records_after = len(self.df_processed)
            print(f"✓ Відфільтровано користувачів з successful_purchase = 1")
            print(f"  Залишилось: {records_after} з {records_before} ({records_after / records_before * 100:.1f}%)")

            # Видалення колонки successful_purchase (більше не потрібна)
            self.df_processed.drop(columns=['successful_purchase'], inplace=True)
            self.df_processed.drop(columns=['answer_errors', 'messages_received'], inplace=True)

            print(f"✓ Видалено колонку 'successful_purchase' (всі значення = 1)")
        else:
            print("⚠️  Колонка 'successful_purchase' не знайдена, пропускаємо фільтрацію")

        if 'likes' in self.df_processed.columns and 'dislikes' in self.df_processed.columns:
            print("✓ Створення колонок like_rate та dislike_rate...")

            likes = self.df_processed['likes']
            dislikes = self.df_processed['dislikes']
            total = likes + dislikes

            # Уникаємо ділення на 0
            self.df_processed['like_rate'] = likes / total.replace(0, np.nan)
            self.df_processed['dislike_rate'] = dislikes / total.replace(0, np.nan)

            # Заповнення NaN у випадках total = 0
            self.df_processed['like_rate'].fillna(0, inplace=True)
            self.df_processed['dislike_rate'].fillna(0, inplace=True)

            # Видалення старих колонок
            self.df_processed.drop(columns=['likes', 'dislikes'], inplace=True)

            print("  ✓ Колонки 'likes' та 'dislikes' видалено")
            print("  ✓ Нові колонки 'like_rate' та 'dislike_rate' додано")
        else:
            print("⚠️  Колонки 'likes'/'dislikes' не знайдено — пропускаємо створення rate-фіч")

        # Видалення дублікатів
        duplicates = self.df_processed.duplicated().sum()
        self.df_processed = self.df_processed.drop_duplicates()
        print(f"✓ Видалено дублікатів: {duplicates}")

        # Перевірка балансу класів
        churn_dist = self.df_processed['is_churned'].value_counts()
        churn_ratio = churn_dist[1] / len(self.df_processed) * 100
        print(f"\n📊 Розподіл класів:")
        print(f"   Churned (1): {churn_dist.get(1, 0)} ({churn_ratio:.2f}%)")
        print(f"   Active (0): {churn_dist.get(0, 0)} ({100 - churn_ratio:.2f}%)")

        if churn_ratio < 30 or churn_ratio > 70:
            print(f"   ⚠️  Дисбаланс класів виявлено! Буде застосовано class_weight='balanced'")

        # Якщо не використовується cleaned версія, виконати категоризацію
        if not self.use_cleaned:
            print("\n✓ Виконується категоризація goals/interests/assistance...")
            self._categorize_features()
        else:
            print("✓ Використовується вже оброблений датасет з категоріями")

        # Обробка пропущених значень
        self._handle_missing_values()

        print(f"✓ Фінальна кількість записів: {len(self.df_processed)}\n")

    def _categorize_features(self):
        """Категоризація текстових ознак (якщо використовується оригінальний датасет)"""

        def split_values(s):
            if pd.isna(s) or s == "":
                return []
            return [x.strip().lower() for x in str(s).split(",") if x.strip()]

        # Обробка текстових колонок, якщо вони є
        text_cols = ['user_goals', 'user_interests', 'user_assistance']

        for col in text_cols:
            if col in self.df_processed.columns:
                self.df_processed[f"{col}_list"] = self.df_processed[col].apply(split_values)

        # Категорії
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

        def assign_categories(df, source_col, categories):
            if f"{source_col}_list" in df.columns:
                for new_col, group_values in categories.items():
                    df[new_col] = df[f"{source_col}_list"].apply(
                        lambda lst: int(any(item in lst for item in group_values))
                    )
            return df

        self.df_processed = assign_categories(self.df_processed, "user_goals", GOALS_CATEGORIES)
        self.df_processed = assign_categories(self.df_processed, "user_assistance", ASSISTANCE_CATEGORIES)
        self.df_processed = assign_categories(self.df_processed, "user_interests", INTEREST_CATEGORIES)

        # Видалення сирих текстових колонок
        cols_to_drop = [col for col in text_cols + [f"{c}_list" for c in text_cols]
                        if col in self.df_processed.columns]
        self.df_processed.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    def _handle_missing_values(self):
        """Обробка пропущених значень"""
        missing_before = self.df_processed.isnull().sum().sum()

        if missing_before > 0:
            print(f"⚠️  Знайдено пропущених значень: {missing_before}")

            # Спеціальна обробка time_to_first_message_sec
            if 'time_to_first_message_sec' in self.df_processed.columns:
                no_messages = self.df_processed['time_to_first_message_sec'].isnull()
                if no_messages.sum() > 0:
                    max_time = self.df_processed['time_to_first_message_sec'].max()
                    self.df_processed.loc[no_messages, 'time_to_first_message_sec'] = max_time * 2
                    print(f"   • time_to_first_message_sec: заповнено {no_messages.sum()} значень")

            # Заповнення медіаною для числових колонок
            numeric_cols = self.df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df_processed[col].isnull().sum() > 0:
                    self.df_processed[col].fillna(self.df_processed[col].median(), inplace=True)

            print(f"✓ Всі пропущені значення оброблено")
        else:
            print("✓ Пропущених значень не виявлено")

    def feature_engineering(self):
        """Крок 2: Feature engineering та відбір ознак"""
        print("=" * 70)
        print("КРОК 2: Feature Engineering та відбір ознак")
        print("=" * 70)

        # Видалення нерелевантних колонок
        cols_to_remove = ['user_id']
        if 'Unnamed: 0' in self.df_processed.columns:
            cols_to_remove.append('Unnamed: 0')

        self.df_processed.drop(columns=cols_to_remove, inplace=True, errors='ignore')
        print(f"✓ Видалено ідентифікатори: {cols_to_remove}")

        # Відділення цільової змінної
        if 'is_churned' not in self.df_processed.columns:
            raise ValueError("Колонка 'is_churned' відсутня в датасеті!")

        X = self.df_processed.drop(columns=['is_churned'])
        y = self.df_processed['is_churned']

        # Видалення колонок з низькою варіативністю
        low_variance_cols = []
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].std() < 0.01:
                low_variance_cols.append(col)

        if low_variance_cols:
            X = X.drop(columns=low_variance_cols)
            print(f"✓ Видалено колонок з низькою варіативністю: {low_variance_cols}")

        # Видалення високо корельованих ознак (щоб уникнути мультиколінеарності)
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        high_corr_cols = []
        for column in upper_triangle.columns:
            if any(upper_triangle[column] > 0.95):
                high_corr_cols.append(column)

        if high_corr_cols:
            X = X.drop(columns=high_corr_cols)
            print(f"✓ Видалено високо корельованих колонок (r > 0.95): {high_corr_cols}")

        print(f"\n✓ Фінальна кількість ознак: {len(X.columns)}")
        print(f"  Ознаки: {list(X.columns)}\n")

        return X, y

    def split_and_scale(self, X, y, test_size=0.2, random_state=42):
        """Крок 3: Поділ даних та масштабування"""
        print("=" * 70)
        print("КРОК 3: Поділ даних та нормалізація")
        print("=" * 70)

        # Розділення на train/test з стратифікацією
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"✓ Train set: {len(self.X_train)} записів")
        print(f"✓ Test set: {len(self.X_test)} записів")
        print(f"✓ Співвідношення train/test: {(1 - test_size) * 100:.0f}% / {test_size * 100:.0f}%")

        # Нормалізація (RobustScaler стійкий до викидів)
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )

        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )

        print(f"✓ Дані нормалізовано за допомогою RobustScaler\n")

    def train_model(self, max_iter=1000, solver='lbfgs', class_weight='balanced'):
        """Крок 4: Тренування моделі логістичної регресії"""
        print("=" * 70)
        print("КРОК 4: Тренування моделі")
        print("=" * 70)

        print(f"🚀 Параметри моделі:")
        print(f"   • Solver: {solver} (швидкий для малих/середніх датасетів)")
        print(f"   • Max iterations: {max_iter}")
        print(f"   • Class weight: {class_weight} (компенсує дисбаланс класів)")
        print(f"   • Penalty: L2 (ridge regression, уникає overfitting)\n")

        # Створення та тренування моделі
        self.model = LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
            random_state=42,
            penalty='l2',
            C=1.0  # Inverse of regularization strength
        )

        import time
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time

        print(f"✓ Модель натреновано за {training_time:.4f} секунд")

        # Витягнення важливості ознак
        self.feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'coefficient': self.model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)

        print(f"✓ Найважливіші ознаки (топ-10):")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"   {row['feature']:.<45} {row['coefficient']:>8.4f}")
        print()

    def cross_validate(self, cv=5):
        """Крок 5: Крос-валідація"""
        print("=" * 70)
        print("КРОК 5: Крос-валідація")
        print("=" * 70)

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Оцінка за різними метриками
        cv_scores = {
            'accuracy': cross_val_score(self.model, self.X_train, self.y_train,
                                        cv=skf, scoring='accuracy'),
            'precision': cross_val_score(self.model, self.X_train, self.y_train,
                                         cv=skf, scoring='precision'),
            'recall': cross_val_score(self.model, self.X_train, self.y_train,
                                      cv=skf, scoring='recall'),
            'f1': cross_val_score(self.model, self.X_train, self.y_train,
                                  cv=skf, scoring='f1'),
            'roc_auc': cross_val_score(self.model, self.X_train, self.y_train,
                                       cv=skf, scoring='roc_auc')
        }

        print(f"📊 Результати {cv}-fold крос-валідації:")
        for metric, scores in cv_scores.items():
            print(f"   {metric.upper():.<20} {scores.mean():.4f} (±{scores.std():.4f})")
        print()

        return cv_scores

    def evaluate_model(self):
        """Крок 6: Оцінка моделі на тестовій вибірці"""
        print("=" * 70)
        print("КРОК 6: Оцінка моделі на тестових даних")
        print("=" * 70)

        # Передбачення
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Метрики
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        print(f"📈 Метрики на тестовій вибірці:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}\n")

        # Детальний звіт
        print("📋 Детальний звіт класифікації:")
        print(classification_report(self.y_test, y_pred,
                                    target_names=['Active (0)', 'Churned (1)']))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("🔲 Confusion Matrix:")
        print(f"   True Negatives:  {cm[0, 0]:>5}")
        print(f"   False Positives: {cm[0, 1]:>5}")
        print(f"   False Negatives: {cm[1, 0]:>5}")
        print(f"   True Positives:  {cm[1, 1]:>5}\n")

        return y_pred, y_pred_proba, accuracy, f1, roc_auc, cm

    def plot_results(self, y_pred, y_pred_proba, cm):
        """Крок 7: Візуалізація результатів"""
        print("=" * 70)
        print("КРОК 7: Візуалізація результатів")
        print("=" * 70)

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
        ax1.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        ax1.set_xticklabels(['Active', 'Churned'])
        ax1.set_yticklabels(['Active', 'Churned'])

        # 2. ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC AUC = {roc_auc:.4f}')
        ax2.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve', fontweight='bold', fontsize=12)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        ax3 = fig.add_subplot(gs[0, 2])
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        ax3.plot(recall, precision, 'g-', linewidth=2)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve', fontweight='bold', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # 4. Feature Correlation with Churn (топ-15)
        ax4 = fig.add_subplot(gs[1, :])

        # Обчислення кореляції між ознаками та is_churned
        feature_correlations = []
        for feature in self.X_train.columns:
            # Об'єднуємо train та test для повної картини
            all_X = pd.concat([self.X_train, self.X_test])
            all_y = pd.concat([self.y_train, self.y_test])

            correlation = all_X[feature].corr(all_y)
            feature_correlations.append({
                'feature': feature,
                'correlation': correlation,
                'coefficient': self.model.coef_[0][list(self.X_train.columns).index(feature)]
            })

        # Сортування за абсолютною кореляцією
        corr_df = pd.DataFrame(feature_correlations).sort_values('correlation', key=abs, ascending=False).head(15)

        # Інвертуємо порядок для відображення (найсильніший вгорі)
        corr_df = corr_df.iloc[::-1]

        # Кольори: червоний = підвищує churn (негатив), зелений = знижує churn (позитив)
        colors = ['#d32f2f' if x > 0 else '#388e3c' for x in corr_df['correlation']]

        # Створення барів
        bars = ax4.barh(range(len(corr_df)), corr_df['correlation'], color=colors, alpha=0.75, edgecolor='black',
                        linewidth=0.8)

        # Налаштування осей
        ax4.set_yticks(range(len(corr_df)))
        ax4.set_yticklabels(corr_df['feature'], fontsize=10)
        ax4.set_xlabel('Correlation with Churn (negative ←  |  →  positive)', fontsize=11)
        ax4.set_title('Top 15 Features: Correlation with Churn Risk', fontweight='bold', fontsize=13)

        # Вертикальна лінія на нулі (вирівняна по центру барів)
        ax4.axvline(x=0, color='#424242', linestyle='-', linewidth=1.5, zorder=0)

        # Горизонтальні лінії сітки на рівні кожного бару (не середини)
        for i in range(len(corr_df) + 1):
            ax4.axhline(y=i - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)

        # Додавання значень кореляції на барах
        for i, (idx, row) in enumerate(corr_df.iterrows()):
            value = row['correlation']
            x_pos = value + (0.01 if value > 0 else -0.01)
            ha = 'left' if value > 0 else 'right'
            ax4.text(x_pos, i, f'{value:.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')

        # Легенда
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d32f2f', alpha=0.75, edgecolor='black', label='Increases Churn Risk'),
            Patch(facecolor='#388e3c', alpha=0.75, edgecolor='black', label='Decreases Churn Risk')
        ]
        ax4.legend(handles=legend_elements, loc='lower right', fontsize=10)

        # Встановлення межі осі X симетрично
        max_abs_corr = corr_df['correlation'].abs().max()
        ax4.set_xlim(-max_abs_corr * 1.15, max_abs_corr * 1.15)

        # 5. Predicted Probability Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(y_pred_proba[self.y_test == 0], bins=30, alpha=0.6,
                 label='Active (0)', color='blue', edgecolor='black')
        ax5.hist(y_pred_proba[self.y_test == 1], bins=30, alpha=0.6,
                 label='Churned (1)', color='red', edgecolor='black')
        ax5.set_xlabel('Predicted Probability')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Predicted Probability Distribution', fontweight='bold', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Class Distribution
        ax6 = fig.add_subplot(gs[2, 1])
        class_counts = [sum(self.y_test == 0), sum(self.y_test == 1)]
        ax6.bar(['Active (0)', 'Churned (1)'], class_counts,
                color=['blue', 'red'], alpha=0.7, edgecolor='black')
        ax6.set_ylabel('Count')
        ax6.set_title('Test Set Class Distribution', fontweight='bold', fontsize=12)
        ax6.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(class_counts):
            ax6.text(i, v + 5, str(v), ha='center', fontweight='bold')

        # 7. Prediction Distribution
        ax7 = fig.add_subplot(gs[2, 2])
        pred_counts = [sum(y_pred == 0), sum(y_pred == 1)]
        ax7.bar(['Active (0)', 'Churned (1)'], pred_counts,
                color=['blue', 'red'], alpha=0.7, edgecolor='black')
        ax7.set_ylabel('Count')
        ax7.set_title('Predicted Class Distribution', fontweight='bold', fontsize=12)
        ax7.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(pred_counts):
            ax7.text(i, v + 5, str(v), ha='center', fontweight='bold')

        plt.savefig('churn_prediction_results.png', dpi=300, bbox_inches='tight')
        print("✓ Візуалізації збережено у 'churn_prediction_results.png'")
        plt.show()
        print()

    def plot_feature_correlation_matrix(self, save_path="feature_correlation_matrix.png", figsize=(16, 14)):
        """
        Будує heatmap кореляційної матриці фіч (всі числові фічі після препроцесингу)
        і зберігає у окремий файл.
        """

        if self.X_train is None or self.X_test is None:
            raise ValueError("Дані не підготовлені. Спочатку викликай split_and_scale().")

        # Об’єднуємо train + test для повнішої матриці
        all_X = pd.concat([self.X_train, self.X_test])

        # Обчислюємо кореляцію
        corr = all_X.corr()

        # Малюємо
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr,
            cmap="coolwarm",
            annot=False,
            cbar=True,
            square=True,
            linewidths=0.5,
            linecolor="gray"
        )
        plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Зберігаємо
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"✓ Кореляційна матриця фіч збережена у файл «{save_path}»")

    def save_model(self, model_path='churn_model.pkl'):
        """Збереження моделі"""
        import pickle

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.X_train.columns.tolist(),
            'feature_importance': self.feature_importance
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Модель збережено у '{model_path}'")

    def predict_new_user(self, user_data):
        """Передбачення для нового користувача"""
        # Переконатися, що всі ознаки присутні
        user_df = pd.DataFrame([user_data])
        user_df = user_df[self.X_train.columns]

        # Нормалізація
        user_scaled = self.scaler.transform(user_df)

        # Передбачення
        prediction = self.model.predict(user_scaled)[0]
        probability = self.model.predict_proba(user_scaled)[0]

        return {
            'prediction': 'Churned' if prediction == 1 else 'Active',
            'churn_probability': probability[1],
            'active_probability': probability[0]
        }


def main():
    """Головна функція"""

    print("\n" + "=" * 70)
    print("🎯 МОДЕЛЬ ПЕРЕДБАЧЕННЯ CHURN КОРИСТУВАЧІВ")
    print("=" * 70 + "\n")

    # Налаштування
    filepath = 'user_features.csv'  # Або 'cleaned.csv' для обробленого
    use_cleaned = False  # True якщо використовуєте cleaned.csv
    use_categories = True  # False для multi-hot encoding кожного значення

    print(f"⚙️  Налаштування:")
    print(f"   Файл: {filepath}")
    print(f"   Режим категорій: {'Групування' if use_categories else 'Multi-hot encoding'}")
    print()

    # Ініціалізація
    model = ChurnPredictionModel(filepath, use_cleaned=use_cleaned, use_categories=use_categories)

    # Виконання етапів
    model.load_and_prepare_data()
    X, y = model.feature_engineering()
    model.split_and_scale(X, y, test_size=0.2)
    model.train_model(max_iter=1000, solver='lbfgs', class_weight='balanced')
    cv_scores = model.cross_validate(cv=5)
    y_pred, y_pred_proba, accuracy, f1, roc_auc, cm = model.evaluate_model()
    model.plot_results(y_pred, y_pred_proba, cm)
    model.plot_feature_correlation_matrix()

    # Збереження моделі
    model.save_model('churn_model.pkl')

    # Збереження feature importance
    model.feature_importance.to_csv('feature_importance.csv', index=False)
    print("✓ Feature importance збережено у 'feature_importance.csv'\n")

    print("=" * 70)
    print("✅ МОДЕЛЮВАННЯ ЗАВЕРШЕНО УСПІШНО!")
    print("=" * 70)
    print("\nФайли збережено:")
    print("  • churn_prediction_results.png - візуалізації (7 графіків)")
    print("  • churn_model.pkl - навчена модель")
    print("  • feature_importance.csv - важливість ознак")
    print("\nМетрики:")
    print(f"  • Accuracy:  {accuracy:.4f}")
    print(f"  • F1-Score:  {f1:.4f}")
    print(f"  • ROC-AUC:   {roc_auc:.4f}")

    # Приклад використання для нового користувача
    print("\n" + "=" * 70)
    print("📝 ПРИКЛАД ПЕРЕДБАЧЕННЯ ДЛЯ НОВОГО КОРИСТУВАЧА")
    print("=" * 70)

    example_user = {col: X.iloc[0][col] for col in X.columns}
    result = model.predict_new_user(example_user)

    print(f"\nРезультат передбачення:")
    print(f"  • Статус: {result['prediction']}")
    print(f"  • Ймовірність churn: {result['churn_probability']:.2%}")
    print(f"  • Ймовірність активності: {result['active_probability']:.2%}")


if __name__ == "__main__":
    main()