import pickle
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class ChurnPredictor:
    """
    Клас для використання навченої моделі передбачення churn.
    Завантажує модель з файлу та робить передбачення для нових користувачів.
    """

    def __init__(self, model_path='churn_model.pkl'):
        """
        Ініціалізація предиктора

        Parameters:
        -----------
        model_path : str
            Шлях до файлу збереженої моделі (.pkl)
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_importance = None
        self._load_model()

    def _load_model(self):
        """Завантаження моделі з файлу"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.feature_importance = model_data.get('feature_importance', None)

            print(f"✓ Модель успішно завантажено з '{self.model_path}'")
            print(f"✓ Кількість ознак: {len(self.feature_names)}")
            print(f"✓ Ознаки моделі: {self.feature_names}\n")

        except FileNotFoundError:
            raise FileNotFoundError(f"Файл моделі '{self.model_path}' не знайдено!")
        except Exception as e:
            raise Exception(f"Помилка при завантаженні моделі: {str(e)}")

    def predict_single(self, user_data):
        """
        Передбачення для одного користувача

        Parameters:
        -----------
        user_data : dict
            Словник з даними користувача. Ключі - назви ознак.
            Приклад: {'onboarding_skips': 0, 'quiz_answers': 3, ...}

        Returns:
        --------
        dict : Результат передбачення з ймовірностями
        """
        # Перевірка наявності всіх ознак
        missing_features = set(self.feature_names) - set(user_data.keys())
        if missing_features:
            raise ValueError(f"Відсутні ознаки: {missing_features}")

        # Створення DataFrame з правильним порядком колонок
        user_df = pd.DataFrame([user_data])[self.feature_names]

        # Нормалізація
        user_scaled = self.scaler.transform(user_df)

        # Передбачення
        prediction = self.model.predict(user_scaled)[0]
        probabilities = self.model.predict_proba(user_scaled)[0]

        result = {
            'prediction': int(prediction),
            'prediction_label': 'Churned' if prediction == 1 else 'Active',
            'churn_probability': float(probabilities[1]),
            'active_probability': float(probabilities[0]),
            'confidence': float(max(probabilities))
        }

        return result

    def predict_batch(self, users_data):
        """
        Передбачення для кількох користувачів

        Parameters:
        -----------
        users_data : list of dict або pandas.DataFrame
            Список словників або DataFrame з даними користувачів

        Returns:
        --------
        pandas.DataFrame : DataFrame з результатами передбачень
        """
        # Конвертація у DataFrame якщо потрібно
        if isinstance(users_data, list):
            users_df = pd.DataFrame(users_data)
        else:
            users_df = users_data.copy()

        # Перевірка наявності всіх ознак
        missing_features = set(self.feature_names) - set(users_df.columns)
        if missing_features:
            raise ValueError(f"Відсутні ознаки: {missing_features}")

        # Вибір потрібних колонок у правильному порядку
        users_df = users_df[self.feature_names]

        # Нормалізація
        users_scaled = self.scaler.transform(users_df)

        # Передбачення
        predictions = self.model.predict(users_scaled)
        probabilities = self.model.predict_proba(users_scaled)

        # Формування результатів
        results_df = pd.DataFrame({
            'prediction': predictions.astype(int),
            'prediction_label': ['Churned' if p == 1 else 'Active' for p in predictions],
            'churn_probability': probabilities[:, 1],
            'active_probability': probabilities[:, 0],
            'confidence': probabilities.max(axis=1)
        })

        return results_df

    def predict_from_csv(self, csv_path, output_path=None):
        """
        Передбачення для користувачів з CSV файлу

        Parameters:
        -----------
        csv_path : str
            Шлях до CSV файлу з даними користувачів
        output_path : str, optional
            Шлях для збереження результатів. Якщо None, не зберігає.

        Returns:
        --------
        pandas.DataFrame : DataFrame з оригінальними даними + передбаченнями
        """
        # Завантаження даних
        df = pd.read_csv(csv_path)
        print(f"✓ Завантажено {len(df)} користувачів з '{csv_path}'")

        # Передбачення
        results = self.predict_batch(df)

        # Об'єднання з оригінальними даними
        df_with_predictions = pd.concat([df, results], axis=1)

        # Збереження результатів
        if output_path:
            df_with_predictions.to_csv(output_path, index=False)
            print(f"✓ Результати збережено у '{output_path}'")

        # Виведення статистики
        churn_count = (results['prediction'] == 1).sum()
        churn_pct = (churn_count / len(results)) * 100
        print(f"\n📊 Статистика передбачень:")
        print(f"   Active користувачів:  {len(results) - churn_count} ({100 - churn_pct:.1f}%)")
        print(f"   Churned користувачів: {churn_count} ({churn_pct:.1f}%)")
        print(f"   Середня ймовірність churn: {results['churn_probability'].mean():.2%}")

        return df_with_predictions

    def get_feature_importance(self, top_n=10):
        """
        Отримання найважливіших ознак

        Parameters:
        -----------
        top_n : int
            Кількість топ-ознак для виведення

        Returns:
        --------
        pandas.DataFrame : DataFrame з важливістю ознак
        """
        if self.feature_importance is not None:
            return self.feature_importance.head(top_n)
        else:
            # Якщо feature_importance не збережено, витягуємо з моделі
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': self.model.coef_[0]
            }).sort_values('coefficient', key=abs, ascending=False)
            return importance_df.head(top_n)

    def explain_prediction(self, user_data, top_n=5):
        """
        Пояснення передбачення для користувача

        Parameters:
        -----------
        user_data : dict
            Словник з даними користувача
        top_n : int
            Кількість топ-факторів для пояснення

        Returns:
        --------
        dict : Результат з поясненням
        """
        # Отримання передбачення
        result = self.predict_single(user_data)

        # Створення DataFrame для користувача
        user_df = pd.DataFrame([user_data])[self.feature_names]
        user_scaled = self.scaler.transform(user_df)

        # Обчислення внеску кожної ознаки
        contributions = user_scaled[0] * self.model.coef_[0]

        # Сортування за абсолютним значенням
        contrib_df = pd.DataFrame({
            'feature': self.feature_names,
            'value': [user_data[f] for f in self.feature_names],
            'contribution': contributions
        }).sort_values('contribution', key=abs, ascending=False)

        # Топ-фактори що підвищують ризик churn
        top_churn_factors = contrib_df[contrib_df['contribution'] > 0].head(top_n)

        # Топ-фактори що знижують ризик churn
        top_active_factors = contrib_df[contrib_df['contribution'] < 0].head(top_n)

        result['top_churn_factors'] = top_churn_factors.to_dict('records')
        result['top_active_factors'] = top_active_factors.to_dict('records')

        return result

    def print_prediction(self, result):
        """Красиве виведення результату передбачення"""
        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТ ПЕРЕДБАЧЕННЯ")
        print("=" * 60)
        print(f"\n🎯 Передбачення: {result['prediction_label']}")
        print(f"   Ймовірність churn:      {result['churn_probability']:.2%}")
        print(f"   Ймовірність активності: {result['active_probability']:.2%}")
        print(f"   Впевненість моделі:     {result['confidence']:.2%}")

        if 'top_churn_factors' in result:
            print(f"\n🔴 Топ-фактори ризику churn:")
            for i, factor in enumerate(result['top_churn_factors'], 1):
                print(f"   {i}. {factor['feature']}: {factor['value']:.2f} "
                      f"(внесок: {factor['contribution']:.4f})")

            print(f"\n🟢 Топ-фактори утримання:")
            for i, factor in enumerate(result['top_active_factors'], 1):
                print(f"   {i}. {factor['feature']}: {factor['value']:.2f} "
                      f"(внесок: {factor['contribution']:.4f})")

        print("=" * 60 + "\n")


# ============================================================================
# ПРИКЛАДИ ВИКОРИСТАННЯ
# ============================================================================

def example_single_prediction():
    """Приклад 1: Передбачення для одного користувача"""
    print("\n" + "=" * 70)
    print("ПРИКЛАД 1: Передбачення для одного користувача")
    print("=" * 70 + "\n")

    # Завантаження моделі
    predictor = ChurnPredictor('churn_model.pkl')

    # Дані користувача (приклад)
    user = {
        'onboarding_skips': 0.0,
        'quiz_answers': 3.0,
        'total_events': 60.0,
        'avg_event_interval_sec': 211.58,
        'chat_opens': 4.0,
        'chat_views': 4.0,
        'messages_sent': 15.0,
        'messages_received': 8.0,
        'answer_errors': 7.0,
        'likes': 1.0,
        'dislikes': 3.0,
        'model_changes': 1.0,
        'successful_purchase': 1.0,
        'time_to_first_message_sec': 343.0,
        'error_rate': 0.467,
        'goals_creative': 0,
        'goals_professional': 0,
        'goals_personal_dev': 0,
        'goals_social_entertain': 1,
        'assist_detailed': 1,
        'assist_concise': 0,
        'interest_creative_arts': 1,
        'interest_practical': 1,
        'interest_outdoor': 0,
        'interest_intellectual': 1,
        'interest_business_tech': 0,
        'interest_lifestyle': 0
    }

    # Просте передбачення
    result = predictor.predict_single(user)
    predictor.print_prediction(result)

    # Передбачення з поясненням
    print("\n📋 Детальне пояснення:")
    result_with_explanation = predictor.explain_prediction(user, top_n=5)
    predictor.print_prediction(result_with_explanation)


def example_batch_prediction():
    """Приклад 2: Передбачення для кількох користувачів"""
    print("\n" + "=" * 70)
    print("ПРИКЛАД 2: Пакетне передбачення")
    print("=" * 70 + "\n")

    # Завантаження моделі
    predictor = ChurnPredictor('churn_model.pkl')

    # Список користувачів
    users = [
        {
            'onboarding_skips': 0.0, 'quiz_answers': 3.0, 'total_events': 60.0,
            'avg_event_interval_sec': 211.58, 'chat_opens': 4.0, 'chat_views': 4.0,
            'messages_sent': 15.0, 'messages_received': 8.0, 'answer_errors': 7.0,
            'likes': 1.0, 'dislikes': 3.0, 'successful_purchase': 1.0,
            'time_to_first_message_sec': 343.0, 'error_rate': 0.467,
            'goals_creative': 0, 'goals_professional': 0, 'goals_personal_dev': 0,
            'goals_social_entertain': 1, 'assist_detailed': 1, 'assist_concise': 0,
            'interest_creative_arts': 1, 'interest_practical': 1, 'interest_outdoor': 0,
            'interest_intellectual': 1, 'interest_business_tech': 0, 'interest_lifestyle': 0
        },
        {
            'onboarding_skips': 2.0, 'quiz_answers': 2.0, 'total_events': 73.0,
            'avg_event_interval_sec': 109.19, 'chat_opens': 3.0, 'chat_views': 3.0,
            'messages_sent': 20.0, 'messages_received': 18.0, 'answer_errors': 1.0,
            'likes': 6.0, 'dislikes': 3.0, 'successful_purchase': 1.0,
            'time_to_first_message_sec': 380.0, 'error_rate': 0.05,
            'goals_creative': 1, 'goals_professional': 1, 'goals_personal_dev': 1,
            'goals_social_entertain': 0, 'assist_detailed': 1, 'assist_concise': 0,
            'interest_creative_arts': 0, 'interest_practical': 0, 'interest_outdoor': 0,
            'interest_intellectual': 0, 'interest_business_tech': 0, 'interest_lifestyle': 0
        }
    ]

    # Пакетне передбачення
    results = predictor.predict_batch(users)
    print("Результати пакетного передбачення:")
    print(results)
    print()


def example_csv_prediction():
    """Приклад 3: Передбачення з CSV файлу"""
    print("\n" + "=" * 70)
    print("ПРИКЛАД 3: Передбачення з CSV файлу")
    print("=" * 70 + "\n")

    # Завантаження моделі
    predictor = ChurnPredictor('churn_model.pkl')

    # Передбачення для всіх користувачів з файлу
    results = predictor.predict_from_csv(
        csv_path='user_features.csv',
        output_path='predictions_output.csv'
    )

    # Виведення перших 10 результатів
    print("\n📋 Перші 10 результатів:")
    print(results[['prediction_label', 'churn_probability', 'confidence']].head(10))

    # Користувачі з високим ризиком churn
    high_risk = results[results['churn_probability'] > 0.7].sort_values(
        'churn_probability', ascending=False
    )

    print(f"\n⚠️  Користувачі з високим ризиком churn (>70%): {len(high_risk)}")
    if len(high_risk) > 0:
        print(high_risk[['prediction_label', 'churn_probability']].head(5))


def example_feature_importance():
    """Приклад 4: Перегляд важливості ознак"""
    print("\n" + "=" * 70)
    print("ПРИКЛАД 4: Найважливіші ознаки")
    print("=" * 70 + "\n")

    # Завантаження моделі
    predictor = ChurnPredictor('churn_model.pkl')

    # Топ-15 ознак
    importance = predictor.get_feature_importance(top_n=15)

    print("📊 Топ-15 найважливіших ознак:")
    for idx, row in importance.iterrows():
        direction = "📈 Підвищує churn" if row['coefficient'] > 0 else "📉 Знижує churn"
        print(f"   {row['feature']:.<45} {row['coefficient']:>8.4f}  {direction}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Запуск всіх прикладів"""

    print("\n" + "=" * 70)
    print("🚀 СИСТЕМА ПЕРЕДБАЧЕННЯ CHURN - INFERENCE MODE")
    print("=" * 70)

    # Виберіть потрібний приклад:

    # Приклад 1: Один користувач
    example_single_prediction()

    # Приклад 2: Кілька користувачів
    # example_batch_prediction()

    # Приклад 3: З CSV файлу
    # example_csv_prediction()

    # Приклад 4: Важливість ознак
    # example_feature_importance()


if __name__ == "__main__":
    main()