# Функция: тестирование наивного сезонного алгоритма

def SNaive_test(train_data, test_data, av_salary_fact):
    '''Тестирование наивного сезонного алгоритма'''
    
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_absolute_error
    
    SNaive_test_prediction_list = []
    for i in range(-12, 0):
        SNaive_test_prediction_list.append(train_data['av_salary'].iloc[i]) 
    SNaive_test_prediction_list_2 = []
    a = 0
    for i in range(0, len(test_data)):
        if i - a * 12 < 12:
            SNaive_test_prediction_list_2.append(SNaive_test_prediction_list[i - a * 12])
        else:
            a += 1
            SNaive_test_prediction_list_2.append(SNaive_test_prediction_list[i - a * 12])
    SNaive_test_prediction = pd.DataFrame(index = test_data.index, data = SNaive_test_prediction_list_2[:len(test_data)], columns = ['av_salary'])
    SNaive_MAPE = mean_absolute_error(test_data['av_salary'], SNaive_test_prediction) / np.mean(test_data)[0]
    return {'SNaive_test_prediction': SNaive_test_prediction,
            'SNaive_MAPE': SNaive_MAPE}


# Функция: прогнозирование наивным сезонным алгоритмом

def SNaive_forecast(train_data, test_data, av_salary_fact, forecast_period):
    '''Прогнозирование наивным сезонным алгоритмом'''
    
    import pandas as pd
    
    forecast_base = pd.concat([train_data, test_data])
    SNaive_forecast_list = []
    for i in range(-12, 0):
        SNaive_forecast_list.append(forecast_base['av_salary'].iloc[i]) 
    SNaive_forecast_list_2 = []
    a = 0
    for i in range(0, len(forecast_period)):
        if i - a * 12 < 12:
            SNaive_forecast_list_2.append(SNaive_forecast_list[i - a * 12])
        else:
            a += 1
            SNaive_forecast_list_2.append(SNaive_forecast_list[i - a * 12])
    SNaive_forecast = pd.DataFrame(index = forecast_period, data = SNaive_forecast_list_2[:len(forecast_period)], columns = ['av_salary'])
    return SNaive_forecast