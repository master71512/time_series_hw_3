from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
from config import CATBOOST_ITERATIONS, CATBOOST_DEPTH, CATBOOST_LEARNING_RATE, MAX_LAG


def create_lag_features(series, max_lag):
    """Создаем лаговые признаки для ряда"""
    df = pd.DataFrame({'y': series})
    for lag in range(1, max_lag + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    # Добавим временные признаки
    df['index'] = np.arange(len(df))
    df.dropna(inplace=True)
    return df


def train_catboost(train_series, test_size, max_lag):
    df = create_lag_features(train_series, max_lag)
    if df.empty or len(df) < 10:
        return np.array([np.nan] * test_size), None

    X = df.drop('y', axis=1)
    y = df['y']

    model = CatBoostRegressor(
        iterations=CATBOOST_ITERATIONS,
        depth=CATBOOST_DEPTH,
        learning_rate=CATBOOST_LEARNING_RATE,
        loss_function='MAE',
        verbose=False,
        random_seed=42
    )
    model.fit(X, y, verbose=False)

    forecasts = []
    current_series = list(train_series)

    for step in range(test_size):
        # Создаем признаки для последней точки
        last_values = current_series[-max_lag:]
        # Если не хватает лагов, дополняем нулями
        if len(last_values) < max_lag:
            last_values = [0] * (max_lag - len(last_values)) + last_values

        features = {'index': len(current_series)}
        for lag in range(1, max_lag + 1):
            features[f'lag_{lag}'] = last_values[-lag]

        X_pred = pd.DataFrame([features])
        # Убедимся, что колонки в том же порядке
        X_pred = X_pred[X.columns]

        pred = model.predict(X_pred)[0]
        forecasts.append(pred)
        current_series.append(pred)

    return np.array(forecasts), model
