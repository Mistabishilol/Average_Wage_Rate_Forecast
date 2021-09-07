# Функция: тестирование комплексной модели прогнозирования

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def Complex_test(
                test_data,
                auto_choice_models,
                SNaive_choice,
                St_m_choice,
                HW_choice,
                SARIMA_choice,
                LSTM_choice,
                SNaive_test_prediction,
                St_m_test_prediction,
                HW_test_prediction,
                SARIMA_test_prediction,
                LSTM_test_prediction
                ):
    '''Функция: тестирование комплексной модели прогнозирования'''

    Complex_list = []
    for i in range(0, len(test_data)):
        if (auto_choice_models in ['yes', 'да']) or (SNaive_choice in ['yes', 'да']):
            a = SNaive_test_prediction['av_salary'][i]
        else:
            a = np.nan
        if (auto_choice_models in ['yes', 'да']) or (St_m_choice in ['yes', 'да']):
            b = St_m_test_prediction['av_salary'][i]
        else:
            b = np.nan
        if (auto_choice_models in ['yes', 'да']) or (HW_choice in ['yes', 'да']):
            c = HW_test_prediction['av_salary'][i]
        else:
            c = np.nan
        if (auto_choice_models in ['yes', 'да']) or (SARIMA_choice in ['yes', 'да']):
            d = SARIMA_test_prediction['av_salary'][i]
        else:
            d = np.nan
        if (auto_choice_models in ['yes', 'да']) or (LSTM_choice in ['yes', 'да']):
            e = LSTM_test_prediction['av_salary'][i]
        else:
            e = np.nan
        
        Complex_list.append(np.nanmedian([a, b, c, d, e]))
    
    Complex_test_prediction = pd.DataFrame(data = Complex_list, index = test_data.index, columns = ['av_salary'])
    Complex_MAPE = mean_absolute_error(test_data['av_salary'], Complex_test_prediction['av_salary']) / np.mean(test_data)[0]
    
    return {'Complex_test_prediction': Complex_test_prediction,
            'Complex_MAPE': Complex_MAPE}


# Функция: комплексная модель прогнозирования

def Complex_forecast(
                    forecast_period,
                    auto_choice_models,
                    SNaive_choice,
                    St_m_choice,
                    HW_choice,
                    SARIMA_choice,
                    LSTM_choice,
                    SNaive_final_prediction,
                    St_m_final_prediction,
                    HW_final_prediction,
                    SARIMA_final_prediction,
                    LSTM_final_prediction
                    ):
    '''Функция: комплексная модель прогнозирования.'''
    
    import numpy as np
    import pandas as pd
    
    Complex_list = []
    for i in range(0, len(forecast_period)):
        if (auto_choice_models in ['yes', 'да']) or (SNaive_choice in ['yes', 'да']):
            a = SNaive_final_prediction['av_salary'][i]
        else:
            a = np.nan
        if (auto_choice_models in ['yes', 'да']) or (St_m_choice in ['yes', 'да']):
            b = St_m_final_prediction['av_salary'][i]
        else:
            b = np.nan
        if (auto_choice_models in ['yes', 'да']) or (HW_choice in ['yes', 'да']):
            c = HW_final_prediction['av_salary'][i]
        else:
            c = np.nan
        if (auto_choice_models in ['yes', 'да']) or (SARIMA_choice in ['yes', 'да']):
            d = SARIMA_final_prediction['av_salary'][i]
        else:
            d = np.nan
        if (auto_choice_models in ['yes', 'да']) or (LSTM_choice in ['yes', 'да']):
            e = LSTM_final_prediction['av_salary'][i]
        else:
            e = np.nan

        Complex_list.append(np.nanmedian([a, b, c, d, e]))
    
    Complex_final = pd.DataFrame(data = Complex_list, index = forecast_period, columns = ['av_salary'])
    
    return Complex_final