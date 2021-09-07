import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from Debug import err_log


# Функция: тестирование алгоритма тройного экспоненциального сглаживания Хольта-Винтерса
def HW_test(train_data, test_data, av_salary_fact, trend_dict, seasonal_dict):
    '''Функция: тестирование алгоритма тройного экспоненциального сглаживания Хольта-Винтерса'''

    best_trend_HW, best_seasonal_HW, HW_MAPE  = '', '', ''
    best_err_HW = 999999999
    
    for i in trend_dict:
        for j in seasonal_dict:
            #try:
            fitted_model_HW = ExponentialSmoothing(train_data[['av_salary']], trend = i, seasonal = j, seasonal_periods = 12).fit()
            test_prediction_HW = fitted_model_HW.forecast(len(test_data))

            MAE_HW = mean_absolute_error(test_data[['av_salary']], test_prediction_HW)
            if MAE_HW < best_err_HW:
                best_err_HW = MAE_HW
                best_trend_HW = i
                best_seasonal_HW = j
            #except Exception as e:
            #    print('Ошибка функции HW_test.')
            #   err_log(e)
            #    continue

    fitted_model_HW = ExponentialSmoothing(train_data[['av_salary']], trend = best_trend_HW, seasonal = best_seasonal_HW, seasonal_periods = 12).fit()
    test_prediction_HW = fitted_model_HW.forecast(len(test_data))
    HW_MAPE = mean_absolute_error(test_data[['av_salary']], test_prediction_HW) / np.mean(test_data)[0]

    # Преобразовываем Series в датафрейм и присваиваем название столбцу.
    test_prediction_HW = test_prediction_HW.to_frame()
    test_prediction_HW.columns = ['av_salary']

    return {'test_prediction_HW': test_prediction_HW,
            'best_trend_HW': best_trend_HW,
            'best_seasonal_HW': best_seasonal_HW,
            'HW_MAPE': HW_MAPE}


# Функция: прогноз алгоритмом тройного экспоненциального сглаживания Хольта-Винтерса

def HW_forecast(train_data, test_data, av_salary_fact, best_trend_HW, best_seasonal_HW, forecast_period):
    '''Функция: прогноз алгоритмом тройного экспоненциального сглаживания Хольта-Винтерса'''

    forecast_base = pd.concat([train_data, test_data])
    final_model_HW = ExponentialSmoothing(forecast_base['av_salary'], trend = best_trend_HW, seasonal = best_seasonal_HW, seasonal_periods = 12).fit()
    forecast_HW = final_model_HW.forecast(len(forecast_period))
    
    # Преобразовываем Series в датафрейм и присваиваем название столбцу.
    forecast_HW = forecast_HW.to_frame()
    forecast_HW.columns = ['av_salary']

    return forecast_HW