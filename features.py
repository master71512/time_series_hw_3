import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf, adfuller
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import StandardScaler


def extract_tsfeatures(series):
    """Извлекаем статистические признаки из временного ряда"""
    series = series[~np.isnan(series)]
    if len(series) < 4:
        return None

    features = {}
    try:
        features['length'] = len(series)
        features['mean'] = np.mean(series)
        features['var'] = np.var(series)
        features['skew'] = skew(series)
        features['kurtosis'] = kurtosis(series)

        # Тренд и сезонность
        try:
            features['adf_stat'] = adfuller(series, autolag='AIC')[0]
        except:
            features['adf_stat'] = 0

        # Энтропия
        try:
            # Дискретизация для энтропии
            hist, _ = np.histogram(series, bins='auto')
            features['entropy'] = entropy(hist)
        except:
            features['entropy'] = 0

        # Автокорреляция
        if len(series) > 2:
            acf_vals = acf(series, nlags=min(5, len(series)//2-1), fft=True)
            for i, val in enumerate(acf_vals[1:6]):
                features[f'acf_lag_{i+1}'] = val

            # Сезонность
            if len(series) > 5:
                acf_full = acf(series, nlags=min(
                    20, len(series)//2-1), fft=True)
                # Игнорируем лаг 0
                if len(acf_full) > 1:
                    seasonal_candidate = np.argmax(np.abs(acf_full[1:])) + 1
                    features['seasonal_period'] = seasonal_candidate
                    features['seasonal_strength'] = acf_full[seasonal_candidate]
                else:
                    features['seasonal_period'] = 1
                    features['seasonal_strength'] = 0
    except Exception as e:
        print(f"Ошибка извлечения признаков: {e}")
        return None

    return features


def create_feature_matrix(all_series_dict):
    feature_list = []
    series_names = []
    for name, series in all_series_dict.items():
        feats = extract_tsfeatures(series)
        if feats:
            feature_list.append(feats)
            series_names.append(name)

    feature_df = pd.DataFrame(feature_list, index=series_names)
    # Заполняем NaN
    feature_df = feature_df.fillna(0)
    return feature_df
