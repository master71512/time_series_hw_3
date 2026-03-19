import numpy as np
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.base import ForecastingHorizon
import warnings
warnings.filterwarnings("ignore")


def naive_forecast(train, test_size, sp=1):
    """Naive метод"""
    forecaster = NaiveForecaster(strategy="last")
    forecaster.fit(train, fh=np.arange(1, test_size+1))
    return forecaster.predict()


def seasonal_naive_forecast(train, test_size, sp):
    """Seasonal Naive"""
    if sp == 1 or sp > len(train):
        # Если сезонность не определена, используем обычный Naive
        return naive_forecast(train, test_size)
    forecaster = NaiveForecaster(strategy="last", sp=sp)
    forecaster.fit(train, fh=np.arange(1, test_size+1))
    return forecaster.predict()


def theta_forecast(train, test_size):
    """Theta метод"""
    try:
        forecaster = ThetaForecaster()
        forecaster.fit(train, fh=np.arange(1, test_size+1))
        return forecaster.predict()
    except:
        # NaN в случае ошибки
        return np.array([np.nan] * test_size)


def ets_forecast(train, test_size):
    """Auto ETS"""
    try:
        forecaster = AutoETS(auto=True, n_jobs=1)
        forecaster.fit(train, fh=np.arange(1, test_size+1))
        return forecaster.predict()
    except:
        return np.array([np.nan] * test_size)


def run_baselines(train, test_size, seasonal_period=1):
    forecasts = {}
    try:
        forecasts['naive'] = naive_forecast(train, test_size)
    except:
        forecasts['naive'] = np.array([np.nan] * test_size)

    try:
        forecasts['s_naive'] = seasonal_naive_forecast(
            train, test_size, seasonal_period)
    except:
        forecasts['s_naive'] = np.array([np.nan] * test_size)

    forecasts['theta'] = theta_forecast(train, test_size)
    forecasts['ets'] = ets_forecast(train, test_size)

    return forecasts
