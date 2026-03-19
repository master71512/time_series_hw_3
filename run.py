import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import json
import os
from datetime import datetime
import joblib


from config import *
from loader import load_and_sample_series
from features import create_feature_matrix
from clustering import cluster_by_features, cluster_by_dtw
from transform import *
from baselines import run_baselines
from models import train_catboost
from eval import calculate_metrics

warnings.filterwarnings("ignore")


info_df = pd.read_csv(os.path.join(DATA_PATH, 'M4-info.csv'))

# Создадим словарь: id - частота
frequency_map = dict(zip(info_df['M4id'], info_df['Frequency']))


def run_experiment_for_series(series, series_name, test_size, seasonal_period, max_lag):
    """Запускаем все эксперименты для одного ряда и собираем результаты в словарь"""
    results = {'series_name': series_name, 'seasonal_period': seasonal_period}

    train, test = series[:-test_size], series[-test_size:]

    final_predictions = {}

    # Бейзлайны на исходных данных
    baseline_forecasts = run_baselines(train, test_size, seasonal_period)
    for name, pred in baseline_forecasts.items():
        if pred is not None and not np.any(np.isnan(pred)):
            final_predictions[f'baseline_{name}'] = pred
        else:
            final_predictions[f'baseline_{name}'] = np.array(
                [np.nan] * test_size)

    # Исходный target
    cat_forecast_raw, _ = train_catboost(train, test_size, max_lag)
    if cat_forecast_raw is not None:
        final_predictions['catboost_raw'] = cat_forecast_raw
    else:
        final_predictions['catboost_raw'] = np.array([np.nan] * test_size)

    # log1p преобразование
    train_log = apply_log1p(train)
    cat_forecast_log, _ = train_catboost(train_log, test_size, max_lag)
    if cat_forecast_log is not None:
        # Обратное преобразование
        final_predictions['catboost_log'] = np.expm1(cat_forecast_log)
    else:
        final_predictions['catboost_log'] = np.array([np.nan] * test_size)

    # Box-Cox преобразование
    try:
        train_box, boxcox_pt = apply_boxcox(train)
        cat_forecast_box, _ = train_catboost(train_box, test_size, max_lag)
        if cat_forecast_box is not None:
            # Обратное преобразование
            forecast_2d = cat_forecast_box.reshape(-1, 1)
            # Если мы сдвигали ряд, нужно вернуть сдвиг обратно
            # будем использовать тот же объект pt
            inv_transformed = boxcox_pt.inverse_transform(
                forecast_2d).flatten()
            final_predictions['catboost_boxcox'] = inv_transformed
        else:
            final_predictions['catboost_boxcox'] = np.array(
                [np.nan] * test_size)
    except Exception as e:
        print(f"Box-Cox ошибка для {series_name}: {e}")
        final_predictions['catboost_boxcox'] = np.array([np.nan] * test_size)

    # Differencing
    try:
        train_diff, original_head = apply_differencing(train, order=1)
        # Для differencing max_lag нужно уменьшить, так как ряд стал короче
        cat_forecast_diff, _ = train_catboost(
            train_diff, test_size, min(max_lag, len(train_diff)-1))
        if cat_forecast_diff is not None:
            # Обратное преобразование
            reconstructed = inverse_differencing(
                np.concatenate([train_diff, cat_forecast_diff]),
                original_head
            )
            # Берем только прогнозную часть
            final_predictions['catboost_diff'] = reconstructed[-test_size:]
        else:
            final_predictions['catboost_diff'] = np.array([np.nan] * test_size)
    except Exception as e:
        print(f"Differencing ошибка для {series_name}: {e}")
        final_predictions['catboost_diff'] = np.array([np.nan] * test_size)

    # Считаем метрики для всех прогнозов
    for name, pred in final_predictions.items():
        if not np.any(np.isnan(pred)):
            metrics = calculate_metrics(test, pred, train)
            for metric_name, value in metrics.items():
                results[f'{name}_{metric_name}'] = value
        else:
            for metric_name in ['MAE', 'RMSE', 'SMAPE', 'MASE']:
                results[f'{name}_{metric_name}'] = np.nan

    return results


def main():

    print("Запуск эксперимента по проверке гипотезы о трансформациях")

    # Загрузка данных
    all_series = load_and_sample_series(DATA_PATH, N_SERIES)
    if not all_series:
        print("Нет данных для анализа. Проверьте путь DATA_PATH в config.py")
        return

    # Извлечение признаков и кластеризация
    feature_df = create_feature_matrix(all_series)

    # Кластеризация
    feature_clusters = cluster_by_features(feature_df, N_CLUSTERS)
    dtw_clusters = cluster_by_dtw(all_series, N_CLUSTERS)

    # Объединяем информацию о кластерах
    series_info = pd.DataFrame({
        'feature_cluster': feature_clusters,
        'dtw_cluster': dtw_clusters
    })

    # Основной цикл эксперимента
    print(f"\nЗапуск экспериментов для {len(all_series)} рядов...")
    all_results = []

    for series_name, series in tqdm(all_series.items()):
        if series_name not in series_info.index:
            continue

        # Получаем предполагаемый сезонный период из кластеризации признаков
        seasonal_period = 1  # значение по умолчанию
        if series_name in frequency_map:
            freq = frequency_map[series_name]
            # Преобразуем текстовую частоту в число
            freq_to_period = {'Yearly': 1, 'Quarterly': 4,
                              'Monthly': 12, 'Weekly': 52, 'Daily': 7, 'Hourly': 24}
            seasonal_period = freq_to_period.get(freq, 1)

        # Запускаем эксперимент для ряда
        series_results = run_experiment_for_series(
            series, series_name, TEST_SIZE, seasonal_period, MAX_LAG
        )

        # Добавляем информацию о кластерах
        series_results['feature_cluster'] = series_info.loc[series_name,
                                                            'feature_cluster']
        series_results['dtw_cluster'] = series_info.loc[series_name, 'dtw_cluster']

        all_results.append(series_results)

    # Сохранение результатов
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_PATH, f'full_results_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nРезультаты сохранены в {results_path}")

    # Сохраняем также информацию о кластерах
    series_info_path = os.path.join(
        RESULTS_PATH, f'series_clusters_{timestamp}.csv')
    series_info.to_csv(series_info_path)

    # Базовый анализ результатов
    print("ПРЕДВАРИТЕЛЬНЫЕ РЕЗУЛЬТАТЫ")

    # Средние метрики по всем рядам
    metric_cols = [
        col for col in results_df.columns if 'MAE' in col or 'SMAPE' in col]
    if metric_cols:
        print("\nСредние значения метрик по всем рядам:")
        print(results_df[metric_cols].mean().to_string())

    # Анализ по кластерам
    if 'feature_cluster' in results_df.columns:
        print("\nСредний SMAPE по кластерам:")
        smape_cols = [col for col in results_df.columns if 'SMAPE' in col]
        cluster_analysis = results_df.groupby('feature_cluster')[
            smape_cols].mean()
        print(cluster_analysis.to_string())

    print(f"\nПолный отчет и анализ в папке {RESULTS_PATH}/")


if __name__ == "__main__":
    main()
