import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Чтобы не было деления на ноль
    diff = np.abs(y_true - y_pred)
    mask = denominator > 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(diff[mask] / denominator[mask]) * 100


def mase(y_true, y_pred, y_train):
    naive_errors = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    if naive_errors == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / naive_errors


def calculate_metrics(y_true, y_pred, y_train):
    metrics = {}
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['SMAPE'] = smape(y_true, y_pred)
    metrics['MASE'] = mase(y_true, y_pred, y_train)
    return metrics
