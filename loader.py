import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
from config import DATA_PATH, N_SERIES, TEST_SIZE


def load_series(data_path, n_series, subset='Train'):
    all_series = {}
    folder_path = os.path.join(data_path, subset)
    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))

    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if len(all_series) >= n_series:
                        break
                    parts = line.strip().split(',')
                    if len(parts) < 2:
                        continue
                    series_id = parts[0].strip('"')
                    try:
                        series = np.array([float(x.strip('"'))
                                          for x in parts[1:] if x.strip('"')])
                    except ValueError:
                        continue  # пропускаем строки с нечисловыми данными
                    if len(series) > 2 * TEST_SIZE:
                        all_series[series_id] = series
        except Exception as e:
            print(f"Ошибка загрузки {file_path}: {e}")
            continue
        if len(all_series) >= n_series:
            break

    print(f"Загружено {len(all_series)} рядов.")
    return all_series


def load_and_sample_series(data_path, n_series):
    """Обертка для загрузки данных"""
    return load_series(data_path, n_series, subset='Train')
