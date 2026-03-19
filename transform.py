import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer


def apply_log1p(series):
    return np.log1p(series)


def apply_boxcox(series):
    # Сдвигаем ряд, если есть неположительные значения
    min_val = series.min()
    if min_val <= 0:
        series_shifted = series - min_val + 1e-6
    else:
        series_shifted = series

    # Обучаем Box-Cox на всем ряду
    pt = PowerTransformer(method='box-cox', standardize=False)
    # Нужен 2D массив (n_samples, 1)
    series_2d = series_shifted.reshape(-1, 1)
    transformed = pt.fit_transform(series_2d).flatten()
    return transformed, pt


def apply_differencing(series, order=1):
    """Взятие разности"""
    diff_series = np.diff(series, n=order)
    # Сохраняем первые order значений для обратного преобразования
    original_head = series[:order]
    return diff_series, original_head


def inverse_differencing(diff_series, original_head):
    """Восстановление ряда из разностей"""
    reconstructed = list(original_head)
    for val in diff_series:
        reconstructed.append(reconstructed[-1] + val)
    return np.array(reconstructed)
