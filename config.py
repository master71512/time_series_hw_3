import os


DATA_PATH = 'Dataset'
RESULTS_PATH = 'results'
os.makedirs(RESULTS_PATH, exist_ok=True)


# Количество рядов для анализа (из диапазона, указанного в условии)
N_SERIES = 200
TEST_SIZE = 14  # Количество точек для тестирования (последние 14 точек ряда)
MAX_LAG = 28  # Максимальный лаг для извлечения признаков
# Количество кластеров для кластеризации (определите оптимальное значение с помощью метода локтя)
N_CLUSTERS = 4
CLUSTER_RANDOM_STATE = 42

# Параметры для модели CatBoost
CATBOOST_ITERATIONS = 500
CATBOOST_DEPTH = 6
CATBOOST_LEARNING_RATE = 0.05
