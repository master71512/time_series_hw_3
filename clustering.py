from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import dtw
import numpy as np
import pandas as pd
from config import N_CLUSTERS, CLUSTER_RANDOM_STATE


def cluster_by_features(feature_df, n_clusters):
    """Кластеризация на основе извлеченных признаков."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_df)

    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=CLUSTER_RANDOM_STATE, n_init='auto')
    labels = kmeans.fit_predict(features_scaled)

    return pd.Series(labels, index=feature_df.index, name='feature_cluster')


def cluster_by_dtw(all_series_dict, n_clusters):
    """Кластеризация самих рядов с помощью DTW."""
    # все ряды должны быть одинаковой длины для tslearn
    # берем минимальную длину или обрезаем/интерполируем
    series_list = []
    series_names = []
    min_len = min(len(s) for s in all_series_dict.values())

    for name, series in all_series_dict.items():
        # Обрезаем до минимальной длины
        series_list.append(series[:min_len])
        series_names.append(name)

    X = np.array(series_list).reshape(len(series_list), min_len, 1)

    # Масштабируем
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)

    # Кластеризация DTW
    dtw_kmeans = TimeSeriesKMeans(n_clusters=n_clusters,
                                  metric="dtw",
                                  max_iter=10,
                                  random_state=CLUSTER_RANDOM_STATE,
                                  verbose=False)
    cluster_labels = dtw_kmeans.fit_predict(X_scaled)

    return pd.Series(cluster_labels, index=series_names, name='dtw_cluster')
