"""
Модель линейной регрессии для предсказания цен на жилье
Анализ данных и предсказание стоимости домов на основе различных характеристик
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("🏠 АНАЛИЗ И ПРОГНОЗИРОВАНИЕ ЦЕН НА НЕДВИЖИМОСТЬ")
print("=" * 80)

# ============================================
# 1. ЗАГРУЗКА И ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ
# ============================================

print("\n📂 1. ЗАГРУЗКА ДАННЫХ")
print("-" * 40)

# Загружаем training данные
df_train = pd.read_excel('predict_house_price_training_data.xlsx', engine='openpyxl')
df_test = pd.read_excel('predict_house_price_test_data.xlsx', engine='openpyxl')

# Переименовываем колонки для удобства
column_names = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                'floors', 'waterfront', 'view', 'condition', 'grade',
                'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
                'latitude', 'longitude']

df_train.columns = column_names
df_test.columns = column_names

print(f"✅ Обучающая выборка: {df_train.shape[0]} записей, {df_train.shape[1]} признаков")
print(f"✅ Тестовая выборка: {df_test.shape[0]} записей, {df_test.shape[1]} признаков")

# ============================================
# 2. ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ (EDA)
# ============================================

print("\n📊 2. ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ")
print("-" * 40)

# Основная статистика
print("\n📈 Статистика цен на жилье:")
print(f"   Средняя цена: ${df_train['price'].mean():,.2f}")
print(f"   Медианная цена: ${df_train['price'].median():,.2f}")
print(f"   Минимальная цена: ${df_train['price'].min():,.2f}")
print(f"   Максимальная цена: ${df_train['price'].max():,.2f}")
print(f"   Стандартное отклонение: ${df_train['price'].std():,.2f}")

# Проверка на пропуски
print("\n🔍 Проверка на пропущенные значения:")
missing = df_train.isnull().sum()
if missing.sum() == 0:
    print("   ✅ Нет пропущенных значений!")
else:
    print(missing[missing > 0])

# Корреляция с ценой
print("\n📊 Корреляция признаков с ценой (топ-10):")
correlations = df_train.corr()['price'].sort_values(ascending=False)
for feature, corr in correlations.head(10).items():
    print(f"   {feature:15s}: {corr:.3f}")

# ============================================
# 3. ВИЗУАЛИЗАЦИЯ ДАННЫХ
# ============================================

print("\n📈 3. СОЗДАНИЕ ГРАФИКОВ...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Анализ недвижимости - Визуализация данных', fontsize=16, fontweight='bold')

# График 1: Распределение цен
axes[0, 0].hist(df_train['price'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Цена ($)')
axes[0, 0].set_ylabel('Количество')
axes[0, 0].set_title('Распределение цен на жилье')
axes[0, 0].ticklabel_format(style='plain', axis='x')

# График 2: Цена vs Жилая площадь
axes[0, 1].scatter(df_train['sqft_living'], df_train['price'], alpha=0.5, s=10)
axes[0, 1].set_xlabel('Жилая площадь (кв. футы)')
axes[0, 1].set_ylabel('Цена ($)')
axes[0, 1].set_title('Зависимость цены от жилой площади')

# График 3: Цена vs Количество спален
df_train.boxplot(column='price', by='bedrooms', ax=axes[0, 2])
axes[0, 2].set_xlabel('Количество спален')
axes[0, 2].set_ylabel('Цена ($)')
axes[0, 2].set_title('Цена в зависимости от количества спален')

# График 4: Цена vs Год постройки
axes[1, 0].scatter(df_train['yr_built'], df_train['price'], alpha=0.5, s=10)
axes[1, 0].set_xlabel('Год постройки')
axes[1, 0].set_ylabel('Цена ($)')
axes[1, 0].set_title('Зависимость цены от года постройки')

# График 5: Цена vs Оценка качества (grade)
axes[1, 1].scatter(df_train['grade'], df_train['price'], alpha=0.5, s=10)
axes[1, 1].set_xlabel('Оценка качества (grade)')
axes[1, 1].set_ylabel('Цена ($)')
axes[1, 1].set_title('Зависимость цены от оценки качества')

# График 6: Тепловая карта корреляций
corr_matrix = df_train.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[1, 2])
axes[1, 2].set_title('Корреляционная матрица')

plt.tight_layout()
plt.savefig('house_price_analysis.png', dpi=150, bbox_inches='tight')
print("   ✅ Графики сохранены в 'house_price_analysis.png'")

# ============================================
# 4. ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ
# ============================================

print("\n🔧 4. ПОДГОТОВКА ДАННЫХ")
print("-" * 40)

# Выбираем признаки для модели
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
            'floors', 'waterfront', 'view', 'condition', 'grade',
            'sqft_above', 'sqft_basement', 'yr_built', 'latitude', 'longitude']

X = df_train[features]
y = df_train['price']

# Создаем дополнительные признаки (feature engineering)
print("   ➕ Создание дополнительных признаков...")

# Возраст дома
X['house_age'] = 2025 - X['yr_built']
# Площадь на комнату
X['sqft_per_room'] = X['sqft_living'] / (X['bedrooms'] + 1)
# Доля подвала
X['basement_ratio'] = X['sqft_basement'] / (X['sqft_living'] + 1)
# Есть ли реновация
X['is_renovated'] = (df_train['yr_renovated'] > 0).astype(int)

# Добавляем новые признаки и для тестовых данных
X_test_full = df_test[features].copy()
X_test_full['house_age'] = 2025 - X_test_full['yr_built']
X_test_full['sqft_per_room'] = X_test_full['sqft_living'] / (X_test_full['bedrooms'] + 1)
X_test_full['basement_ratio'] = X_test_full['sqft_basement'] / (X_test_full['sqft_living'] + 1)
X_test_full['is_renovated'] = (df_test['yr_renovated'] > 0).astype(int)

# Обновляем список признаков
features_extended = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                     'floors', 'waterfront', 'view', 'condition', 'grade',
                     'sqft_above', 'sqft_basement', 'latitude', 'longitude',
                     'house_age', 'sqft_per_room', 'basement_ratio', 'is_renovated']

X = X[features_extended]
X_test_full = X_test_full[features_extended]

print(f"   ✅ Количество признаков: {len(features_extended)}")
print(f"   ✅ Размер обучающей выборки: {X.shape}")
print(f"   ✅ Размер тестовой выборки: {X_test_full.shape}")

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_full)

# Разделение на обучение и валидацию
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"   ✅ Обучающая выборка: {X_train.shape[0]} записей")
print(f"   ✅ Валидационная выборка: {X_val.shape[0]} записей")

# ============================================
# 5. ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================

print("\n🤖 5. ОБУЧЕНИЕ МОДЕЛЕЙ")
print("-" * 40)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.001)
}

results = {}

for name, model in models.items():
    print(f"\n   📊 Обучаем {name}...")
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Метрики
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_mae = mean_absolute_error(y_val, y_pred_val)
    
    results[name] = {
        'model': model,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'val_rmse': val_rmse,
        'val_mae': val_mae
    }
    
    print(f"      R² (train): {train_r2:.4f}")
    print(f"      R² (val):   {val_r2:.4f}")
    print(f"      RMSE:       ${val_rmse:,.2f}")
    print(f"      MAE:        ${val_mae:,.2f}")

# Выбираем лучшую модель
best_model_name = max(results, key=lambda x: results[x]['val_r2'])
best_model = results[best_model_name]['model']
print(f"\n   ✅ Лучшая модель: {best_model_name}")

# ============================================
# 6. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ
# ============================================

print("\n📊 6. ВАЖНОСТЬ ПРИЗНАКОВ")
print("-" * 40)

# Для линейной регрессии используем коэффициенты
if best_model_name in ['Linear Regression', 'Ridge Regression']:
    coef_df = pd.DataFrame({
        'Признак': features_extended,
        'Коэффициент': best_model.coef_
    })
    coef_df['Абсолютное влияние'] = coef_df['Коэффициент'].abs()
    coef_df = coef_df.sort_values('Абсолютное влияние', ascending=False)
    
    print("\n   Топ-10 самых важных признаков:")
    for i, row in coef_df.head(10).iterrows():
        sign = "+" if row['Коэффициент'] > 0 else "-"
        print(f"      {row['Признак']:20s} → {sign}${abs(row['Коэффициент']):,.2f}")

# ============================================
# 7. ПРЕДСКАЗАНИЕ НА ТЕСТОВЫХ ДАННЫХ
# ============================================

print("\n🏠 7. ПРЕДСКАЗАНИЕ ЦЕН НА ТЕСТОВЫХ ДАННЫХ")
print("-" * 40)

# Предсказываем цены
test_predictions = best_model.predict(X_test_scaled)

# Добавляем предсказания в тестовый датафрейм
df_test_with_predictions = df_test.copy()
df_test_with_predictions['predicted_price'] = test_predictions

print(f"   ✅ Предсказания сделаны для {len(test_predictions)} объектов")

# Сохраняем результаты
df_test_with_predictions[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'predicted_price']].to_csv('predictions.csv', index=False)
print("   ✅ Результаты сохранены в 'predictions.csv'")

# Показываем примеры предсказаний
print("\n   📋 Примеры предсказаний (первые 10):")
sample_df = df_test_with_predictions[['price', 'bedrooms', 'sqft_living', 'predicted_price']].head(10)
for idx, row in sample_df.iterrows():
    print(f"      Дом {idx+1}: цена={row['price']:,.0f} → предсказано={row['predicted_price']:,.0f}")

# ============================================
# 8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ============================================

print("\n📈 8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Результаты предсказания цен на жилье', fontsize=14, fontweight='bold')

# Сравнение реальных и предсказанных цен на валидации
y_pred_val_best = best_model.predict(X_val)
axes[0].scatter(y_val, y_pred_val_best, alpha=0.5, s=10)
axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[0].set_xlabel('Реальная цена ($)')
axes[0].set_ylabel('Предсказанная цена ($)')
axes[0].set_title(f'Реальные vs Предсказанные цены\nR² = {results[best_model_name]["val_r2"]:.3f}')

# Распределение ошибок
errors = y_val - y_pred_val_best
axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Ошибка предсказания ($)')
axes[1].set_ylabel('Частота')
axes[1].set_title('Распределение ошибок предсказания')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)

# Топ-10 самых важных признаков (только для линейных моделей)
if best_model_name in ['Linear Regression', 'Ridge Regression']:
    top_features = coef_df.head(10)
    colors = ['green' if x > 0 else 'red' for x in top_features['Коэффициент']]
    axes[2].barh(range(len(top_features)), top_features['Коэффициент'], color=colors)
    axes[2].set_yticks(range(len(top_features)))
    axes[2].set_yticklabels(top_features['Признак'])
    axes[2].set_xlabel('Влияние на цену')
    axes[2].set_title('Топ-10 важных признаков')
    axes[2].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
print("   ✅ Графики сохранены в 'prediction_results.png'")

# ============================================
# 9. ИТОГОВЫЙ ОТЧЕТ
# ============================================

print("\n" + "=" * 80)
print("📊 ИТОГОВЫЙ ОТЧЕТ")
print("=" * 80)

print(f"""
🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model_name}

📈 МЕТРИКИ КАЧЕСТВА:
   • R² на обучении:     {results[best_model_name]['train_r2']:.4f}
   • R² на валидации:    {results[best_model_name]['val_r2']:.4f}
   • RMSE (ошибка):      ${results[best_model_name]['val_rmse']:,.2f}
   • MAE (средняя ошибка): ${results[best_model_name]['val_mae']:,.2f}

🔍 ИНТЕРПРЕТАЦИЯ:
   • Модель объясняет {results[best_model_name]['val_r2']*100:.1f}% вариации цен
   • Средняя ошибка предсказания: ${results[best_model_name]['val_mae']:,.2f}
   • Это составляет около {results[best_model_name]['val_mae']/df_train['price'].mean()*100:.1f}% от средней цены

💡 РЕКОМЕНДАЦИИ:
   • {'Модель показывает хорошее качество, можно использовать для оценки' if results[best_model_name]['val_r2'] > 0.7 else 'Качество модели среднее, рекомендуется добавить больше признаков'}
   • Самые важные факторы: {", ".join(coef_df.head(3)['Признак'].values)}
""")

print("=" * 80)
print("✅ Анализ завершен! Результаты сохранены в файлы:")
print("   📁 house_price_analysis.png - визуализация данных")
print("   📁 prediction_results.png - графики предсказаний")  
print("   📁 predictions.csv - предсказанные цены")
print("=" * 80)