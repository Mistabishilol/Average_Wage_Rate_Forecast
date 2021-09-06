import warnings
warnings.filterwarnings('ignore')
from Debug import err_log
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# Функция: тестирование сезонной авторегрессионной интегрированной модели скользящего среднего Бокса-Дженкинса

def SARIMA_test(train_data, test_data, av_salary_fact, p_, d_, q_, P_, D_, Q_):
    '''Функция: тестирование сезонной авторегрессионной интегрированной модели скользящего среднего Бокса-Дженкинса'''
    
    best_err_SARIMA = 999999999
    for p in p_:
        for d in d_:
            for q in q_:
                for P in P_:
                    for D in D_:
                        for Q in Q_:
                            try:
                                # Есть данные по средней номинальной ЗП (Росстат).
                                if 'rs_avg_nom_sal' in train_data.columns:
                                    model = SARIMAX(endog=train_data['av_salary'],
                                                    exog=train_data['rs_avg_nom_sal'],
                                                    order = (p, d, q),
                                                    seasonal_order = (P, D, Q, 12)
                                                    )
                                    results = model.fit()

                                    exog_forecast = train_data.loc['2019-09-01':'2021-01-01', 'rs_avg_nom_sal']
                                    exog_forecast = exog_forecast.to_frame()
                                    exog_forecast.columns = ['rs_avg_nom_sal']
                                    test_prediction_SARIMA = results.predict(start = len(train_data), end = len(train_data) + len(test_data) - 1, exog=exog_forecast, dynamic = False, typ = 'levels')

                                # Нет данных по средней номинальной ЗП (Росстат).
                                else:
                                    model = SARIMAX(endog=train_data['av_salary'],
                                                    order = (p, d, q),
                                                    seasonal_order = (P, D, Q, 12)
                                                    )
                                    results = model.fit()
                                    test_prediction_SARIMA = results.predict(start = len(train_data), end = len(train_data) + len(test_data) - 1, dynamic = False, typ = 'levels')
                                

                                MAE_SARIMA = mean_absolute_error(test_data[['av_salary']], test_prediction_SARIMA)

                                if MAE_SARIMA < best_err_SARIMA:
                                    best_err_SARIMA = MAE_SARIMA
                                    best_p = p
                                    best_d = d
                                    best_q = q
                                    best_P = P
                                    best_D = D
                                    best_Q = Q
                            except Exception as e:
                                print('Ошибка функции SARIMA_test.')
                                err_log(e)
                                continue

    model = SARIMAX(train_data[['av_salary']], order=(best_p, best_d, best_q), seasonal_order = (best_P, best_D, best_Q, 12), enforce_stationarity=False, initialization='approximate_diffuse')
    results = model.fit()
    test_prediction_SARIMA = results.predict(start = len(train_data), end = len(train_data) + len(test_data) - 1, dynamic = False, typ = 'levels')
    SARIMA_MAPE = mean_absolute_error(test_data[['av_salary']], test_prediction_SARIMA) / np.mean(test_data)[0]
    
    # Преобразовываем Series в датафрейм и присваиваем название столбцу.
    test_prediction_SARIMA = test_prediction_SARIMA.to_frame()
    test_prediction_SARIMA.columns = ['av_salary']

    return {'test_prediction_SARIMA': test_prediction_SARIMA,
            'SARIMA_MAPE':SARIMA_MAPE,
            'best_p': best_p,
            'best_d': best_d,
            'best_q': best_d,
            'best_P': best_P,
            'best_D': best_D,
            'best_Q': best_Q
            }


# Функция: сезонная авторегрессионная интегрированная модель скользящего среднего Бокса-Дженкинса

def SARIMA_forecast(train_data, test_data, av_salary_fact, best_p, best_d, best_q, best_P, best_D, best_Q, forecast_period):
    '''Функция: сезонная авторегрессионная интегрированная модель скользящего среднего Бокса-Дженкинса'''
    
    forecast_base = pd.concat([train_data, test_data])
    
    # Проверяем, есть ли данные по средней номинальной ЗП (Росстат).
    if 'rs_avg_nom_sal' in forecast_base.columns:
        final_model_SARIMA = SARIMAX(endog=forecast_base[['av_salary']],
                                    exog=forecast_base[['rs_avg_nom_sal']],
                                    order = (best_p, best_d, best_q),
                                    seasonal_order = (best_P, best_D, best_Q, 12)
                                    )

        results_SARIMA = final_model_SARIMA.fit()

        exog_forecast = forecast_base.loc['2020-03-01':'2021-01-01', 'rs_avg_nom_sal']
        exog_forecast = exog_forecast.to_frame()
        exog_forecast.columns = ['rs_avg_nom_sal']
        forecast_SARIMA = results_SARIMA.predict(start = len(forecast_base), end = len(forecast_base) + len(forecast_period) - 1, exog=exog_forecast, dynamic = False, typ = 'levels')

    else:
        final_model_SARIMA = SARIMAX(endog=forecast_base[['av_salary']],
                                    order = (best_p, best_d, best_q),
                                    seasonal_order = (best_P, best_D, best_Q, 12)
                                    )
        results_SARIMA = final_model_SARIMA.fit()
        forecast_SARIMA = results_SARIMA.predict(start = len(forecast_base), end = len(forecast_base) + len(forecast_period) - 1, dynamic = False, typ = 'levels')
    
    # Преобразовываем Series в датафрейм и присваиваем название столбцу.
    forecast_SARIMA = forecast_SARIMA.to_frame()
    forecast_SARIMA.columns = ['av_salary']

    return forecast_SARIMA
