import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error    

# Функция: тестирование стандартного алгоритма (сезонность к базовому периоду):

def St_m_test(train_data, test_data, av_salary_fact, base_period):
    '''Тестирование стандартной модели прогнозирования драйверов ФОТ'''
    
    train_base_median = []
    base_period_start = -base_period
    base_period_end = 0
    
    while -base_period_start <= len(train_data):
        if base_period_end == 0:
            train_base_median.append(np.median(train_data[base_period_start:]))
        else:
            train_base_median.append(np.median(train_data[base_period_start:base_period_end]))
        base_period_start = base_period_start - 12
        base_period_end = base_period_end - 12
    j = -12
    i = 1
    seasonal_kf = []
    
    while i < len(train_base_median):
        while j <= -1:
            seasonal_kf.append(train_data['av_salary'].iloc[j - (i - 1) * 12] / train_base_median[i])
            j += 1
        j = -12
        i += 1
    i = 0
    j = 0
    seasonal_kf_med = []
    seasonal_kf_med_ = []
    
    while i < 12:
        while j < len(seasonal_kf):
            seasonal_kf_med_.append(seasonal_kf[j])
            j = j + 12
        seasonal_kf_med.append(np.median(seasonal_kf_med_))
        seasonal_kf_med_ = []
        i = i + 1
        j = i

    St_m_test_prediction = []
    base = train_base_median[0]
    a = 0

    for i in range(0, len(test_data)):
        if i - a * 12 < 12:
            St_m_test_prediction.append(base * seasonal_kf_med[i - a * 12])
        else:
            a += 1
            base = np.median(St_m_test_prediction[-base_period:])
            St_m_test_prediction.append(base * seasonal_kf_med[i - a * 12])

    St_m_test_prediction = pd.DataFrame(data = St_m_test_prediction, index = test_data.index, columns = ['av_salary']).fillna(0)
    St_m_MAPE = mean_absolute_error(test_data['av_salary'], St_m_test_prediction['av_salary']) / np.mean(test_data)[0]

    return St_m_test_prediction, St_m_MAPE


# Функция: прогнозирование стандартным алгоритмом (сезонность к базовому периоду):

def St_m_forecast(train_data, test_data, av_salary_fact, base_period, forecast_period):
    '''Прогнозирование драйверов ФОТ стандартным образом'''
    
    forecast_base = pd.concat([train_data, test_data])
    train_base_median = []
    base_period_start = -base_period
    base_period_end = 0
    
    while -base_period_start <= len(forecast_base):
        if base_period_end == 0:
            train_base_median.append(np.median(forecast_base[base_period_start:]))
        else:
            train_base_median.append(np.median(forecast_base[base_period_start:base_period_end]))
        base_period_start = base_period_start - 12
        base_period_end = base_period_end - 12
    
    j = -12
    i = 1
    seasonal_kf = []
    
    while i < len(train_base_median):
        while j <= -1:
            seasonal_kf.append(forecast_base['av_salary'].iloc[j - (i - 1) * 12] / train_base_median[i])
            j += 1
        j = -12
        i += 1

    i = 0
    j = 0
    seasonal_kf_med = []
    seasonal_kf_med_ = []

    while i < 12:
        while j < len(seasonal_kf):
            seasonal_kf_med_.append(seasonal_kf[j])
            j = j + 12
        seasonal_kf_med.append(np.median(seasonal_kf_med_))
        seasonal_kf_med_ = []
        i = i + 1
        j = i

    St_m_forecast = []
    base = train_base_median[0]
    a = 0

    for i in range(0, len(forecast_period)):
        if i - a * 12 < 12:
            St_m_forecast.append(base * seasonal_kf_med[i - a * 12])
        else:
            a += 1
            base = np.median(St_m_forecast[-base_period:])
            St_m_forecast.append(base * seasonal_kf_med[i - a * 12])

    St_m_forecast = pd.DataFrame(data = St_m_forecast, index = forecast_period, columns = ['av_salary'])
    
    return St_m_forecast
