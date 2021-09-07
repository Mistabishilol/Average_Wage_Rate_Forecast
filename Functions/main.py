# Импорт библиотек из PyPI
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import datetime

# Импорт пользовательских функций
from data_io import data_load, periods_choice, forecasts_to_dataframe, save_file, create_folders
from SNaive import SNaive_test, SNaive_forecast 
from St_m import St_m_test, St_m_forecast
from St_m_add import St_m_test_add, St_m_forecast_add
from HW import HW_test, HW_forecast
from SARIMA import SARIMA_test, SARIMA_forecast
from lstm import LSTM_test, LSTM_forecast
from Complex import Complex_test, Complex_forecast
from user_input import branch_choice, choice_date, choice_models
from plot_graph import plot_graph


def forecast_main():
    '''Основная функция для построения прогнозов.'''

    ### 1. Загружаем данные из базы планирования. ###
    data_loaded = data_load()   
    forecasts_table = data_loaded['forecasts_table']
    fot_database_dict = data_loaded['fot_database_dict']
    all_sections = data_loaded['all_sections']
    fot_database = data_loaded['fot_database']


    ### 2. Взаимодействие с пользователем. ###
    # Просим пользователя выбрать отделение:
    tb_gosb_segm_type_choice = branch_choice(fot_database_dict, all_sections)
    # Просим пользователя выбрать даты:
    user_choice_date = choice_date()
    # Просим пользователя выбрать алгоритмы и их параметры:    
    user_choice_models = choice_models()    
    # Создаем основные папки по умолчанию, куда в дальнейшем поместим результаты прогнозов, графики расчетов и т.д.
    folders = create_folders()


    ### 3. Основной цикл программы, в котором перебираются атрибуты (выбранные ТБ/ГОСБ/Сегменты и типы), для каждого из них строятся прогнозы по выбранным моделям.
    for i in range(0, len(tb_gosb_segm_type_choice)):
        
        ### 3.0 Определение границ обучающего, тестового и прогнозного периода. ###
        periods = periods_choice(i,
                            fot_database,
                            tb_gosb_segm_type_choice,
                            user_choice_date['auto_choice_date'],
                            user_choice_date['train_date_start'],
                            user_choice_date['train_date_end'],
                            user_choice_date['test_date_end'],
                            user_choice_date['forecast_date_end']
                           )
        
        av_salary_fact = periods['av_salary_fact']
        train_border = periods['train_border']
        train_data = periods['train_data']
        test_data = periods['test_data']
        chosen_tb_gosb_segm_type = periods['chosen_tb_gosb_segm_type']
        forecast_period = periods['forecast_period']
        
        # Оповещаем пользователя о текущем сегменте.
        now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        print(f'{now} {chosen_tb_gosb_segm_type} работаю')

        # Если выбраны все модели, то проставляем соответствующие флаги.
        if user_choice_models['auto_choice_models'] in ['yes', 'да']:
            user_choice_models['SNaive_choice'] = 'yes'
            user_choice_models['St_m_choice'] = 'yes'
            user_choice_models['HW_choice'] = 'yes'
            user_choice_models['SARIMA_choice'] = 'yes'
            user_choice_models['LSTM_choice'] = 'yes'


        ### 3.1 Наивный сезонный алгоритм: ###
        if user_choice_models['SNaive_choice'] in ['yes', 'да']:
            SNaive_results = SNaive_test(train_data, test_data, av_salary_fact)
            SNaive_test_prediction = SNaive_results['SNaive_test_prediction']
            SNaive_MAPE = SNaive_results['SNaive_MAPE']
            SNaive_final_prediction = SNaive_forecast(train_data, test_data, av_salary_fact, forecast_period)

            # Наивный сезонный алгоритм - график
            plot_graph(
                        model_name='SNaive',
                        path=folders['graphs_folder_path'],
                        train_data=train_data,
                        test_data=test_data,
                        chosen_tb_gosb_segm_type = chosen_tb_gosb_segm_type,
                        test_results=SNaive_test_prediction,
                        prediction_results=SNaive_final_prediction,
                        MAPE=SNaive_MAPE
                        )
        else:
            SNaive_test_prediction = pd.DataFrame(data = [], index = test_data.index, columns = ['av_salary'])
            SNaive_MAPE = np.nan
            SNaive_final_prediction = pd.DataFrame(data = [], index = forecast_period, columns = ['av_salary'])



        ### 3.2 Стандартный алгоритм база - сезонность. ###
        if user_choice_models['St_m_choice'] in ['yes', 'да']:
            if (user_choice_models['auto_choice_models'] in ['no', 'нет']) and (user_choice_models['St_m_param_auto'] in ['no', 'нет']):
                St_m_results = St_m_test(train_data, test_data, av_salary_fact, user_choice_models['base_period'])
                St_m_best_base_period = user_choice_models['base_period']
            else:
                best_err_St_m = 999999999
                
                for base_period in range(3, 13):
                    St_m_results = St_m_test(train_data, test_data, av_salary_fact, base_period)
                    St_m_test_prediction = St_m_results[0]
                    St_m_MAPE = St_m_results[1]
                    
                    if St_m_MAPE < best_err_St_m:
                        best_err_St_m = St_m_MAPE
                        St_m_best_base_period = base_period
                
                St_m_results = St_m_test(train_data, test_data, av_salary_fact, St_m_best_base_period)
            
            St_m_test_prediction = St_m_results[0]
            St_m_MAPE = St_m_results[1]
            St_m_final_prediction = St_m_forecast(train_data, test_data, av_salary_fact, St_m_best_base_period, forecast_period)

            # Стандартный алгоритм база - сезонность - график.
            plot_graph(
                        model_name='St_m',
                        path=folders['graphs_folder_path'],
                        train_data=train_data,
                        test_data=test_data,
                        chosen_tb_gosb_segm_type = chosen_tb_gosb_segm_type,
                        test_results=St_m_test_prediction,
                        prediction_results=St_m_final_prediction,
                        MAPE=St_m_MAPE
                        )
        else:
            St_m_test_prediction = pd.DataFrame(data = [], index = test_data.index, columns = ['av_salary'])
            St_m_MAPE = np.nan
            St_m_final_prediction = pd.DataFrame(data = [], index = forecast_period, columns = ['av_salary'])
            St_m_best_base_period = ''


        ### 3.3 Тройное экспоненциальное сглаживание. ###
        if user_choice_models['HW_choice'] in ['yes', 'да']:
            if (user_choice_models['auto_choice_models'] in ['no', 'нет']) and (user_choice_models['HW_param_auto'] in ['no', 'нет']):
                HW_results = HW_test(train_data, test_data, av_salary_fact, user_choice_models['trend_dict'], user_choice_models['seasonal_dict'])
            else:
                trend_dict = ['mul', 'add'] 
                seasonal_dict = ['mul', 'add']
                HW_results = HW_test(train_data, test_data, av_salary_fact, trend_dict, seasonal_dict)
            
            HW_MAPE = HW_results['HW_MAPE']
            HW_test_prediction = HW_results['test_prediction_HW']
            HW_test_prediction.index = test_data.index
            HW_final_prediction = HW_forecast(train_data, test_data, av_salary_fact, HW_results['best_trend_HW'], HW_results['best_seasonal_HW'], forecast_period)
            HW_best_trend = HW_results['best_trend_HW']
            HW_best_seasonal = HW_results['best_seasonal_HW']

            # Тройное экспоненциальное сглаживание - график.       
            plot_graph(
                        model_name='HW',
                        path=folders['graphs_folder_path'],
                        train_data=train_data,
                        test_data=test_data,
                        chosen_tb_gosb_segm_type = chosen_tb_gosb_segm_type,
                        test_results=HW_test_prediction,
                        prediction_results=HW_final_prediction,
                        MAPE=HW_MAPE
                        )
        else:
            HW_test_prediction = pd.DataFrame(data = [], index = test_data.index, columns = ['av_salary'])
            HW_MAPE = np.nan
            HW_final_prediction = pd.DataFrame(data = [], index = forecast_period, columns = ['av_salary'])
            HW_results = ''
            HW_best_trend = ''
            HW_best_seasonal = ''


        ### 3.4 SARIMA: Сезонный авторегрессионный интегрированный алгоритм скользящего среднего. ###
        if user_choice_models['SARIMA_choice'] in ['yes', 'да']:
            if (user_choice_models['SARIMA_param_auto'] in ['no', 'нет']):
                
                SARIMA_results = SARIMA_test(
                                            train_data,
                                            test_data,
                                            av_salary_fact,
                                            user_choice_models['SARIMA_p_'],
                                            user_choice_models['SARIMA_d_'],
                                            user_choice_models['SARIMA_q_'],
                                            user_choice_models['SARIMA_P_'],
                                            user_choice_models['SARIMA_D_'],
                                            user_choice_models['SARIMA_Q_']
                                            )
            else:
                SARIMA_p_ = SARIMA_d_ = SARIMA_q_ = [0, 1]
                SARIMA_P_ = SARIMA_D_ = SARIMA_Q_ = [0, 1]
                SARIMA_results = SARIMA_test(train_data,
                                            test_data,
                                            av_salary_fact,
                                            SARIMA_p_,
                                            SARIMA_d_,
                                            SARIMA_q_,
                                            SARIMA_P_,
                                            SARIMA_D_,
                                            SARIMA_Q_
                                            )
            
            SARIMA_test_prediction = SARIMA_results['test_prediction_SARIMA']
            SARIMA_MAPE = SARIMA_results['SARIMA_MAPE']
            SARIMA_best_p = SARIMA_results['best_p']
            SARIMA_best_d = SARIMA_results['best_d']
            SARIMA_best_q = SARIMA_results['best_q']
            SARIMA_best_P = SARIMA_results['best_P']
            SARIMA_best_D = SARIMA_results['best_D']
            SARIMA_best_Q = SARIMA_results['best_Q']
            SARIMA_final_prediction = SARIMA_forecast(train_data,
                                                    test_data,
                                                    av_salary_fact,
                                                    SARIMA_best_p,
                                                    SARIMA_best_d,
                                                    SARIMA_best_q,
                                                    SARIMA_best_P,
                                                    SARIMA_best_D,
                                                    SARIMA_best_Q,
                                                    forecast_period
                                                    )

            # SARIMA: график.
            plot_graph(
                        model_name='SARIMA',
                        path=folders['graphs_folder_path'],
                        train_data=train_data,
                        test_data=test_data,
                        chosen_tb_gosb_segm_type = chosen_tb_gosb_segm_type,
                        test_results=SARIMA_test_prediction,
                        prediction_results=SARIMA_final_prediction,
                        MAPE=SARIMA_MAPE
                        )
        else:
            SARIMA_test_prediction = pd.DataFrame(data = [], index = test_data.index, columns = ['av_salary'])
            SARIMA_MAPE = np.nan
            SARIMA_final_prediction = pd.DataFrame(data = [], index = forecast_period, columns = ['av_salary'])
            SARIMA_best_p = ''
            SARIMA_best_d = ''
            SARIMA_best_q = ''
            SARIMA_best_P = ''
            SARIMA_best_D = ''
            SARIMA_best_Q = ''


        ### 3.5 LSTM: Рассчитаем прогноз при помощи нейросети. ###
        if user_choice_models['LSTM_choice'] in ['yes', 'да']:
            LSTM_test_results = LSTM_test(train_data, test_data, av_salary_fact)
            LSTM_test_prediction = LSTM_test_results['test_prediction_LSTM']
            LSTM_MAPE = LSTM_test_results['LSTM_MAPE']
            LSTM_final_prediction = LSTM_forecast(train_data, test_data, av_salary_fact, forecast_period)

         # LSTM: график.
            plot_graph(
                        model_name='LSTM',
                        path=folders['graphs_folder_path'],
                        train_data=train_data,
                        test_data=test_data,
                        chosen_tb_gosb_segm_type = chosen_tb_gosb_segm_type,
                        test_results=LSTM_test_prediction,
                        prediction_results=LSTM_final_prediction,
                        MAPE=LSTM_MAPE
                        )
        else:
            LSTM_test_prediction = pd.DataFrame(data = [], index = test_data.index, columns = ['av_salary'])
            LSTM_MAPE = np.nan
            LSTM_final_prediction = pd.DataFrame(data = [], index = forecast_period, columns = ['av_salary'])


        ### 3.6 Complex: рассчитаем комплексный прогноз. ###
        Complex_test_results = Complex_test(
                                            test_data,
                                            user_choice_models['auto_choice_models'],
                                            user_choice_models['SNaive_choice'],
                                            user_choice_models['St_m_choice'],
                                            user_choice_models['HW_choice'],
                                            user_choice_models['SARIMA_choice'],
                                            user_choice_models['LSTM_choice'],
                                            SNaive_test_prediction,
                                            St_m_test_prediction,
                                            HW_test_prediction,
                                            SARIMA_test_prediction,
                                            LSTM_test_prediction
                                            )

        Complex_test_prediction = Complex_test_results['Complex_test_prediction']
        Complex_MAPE = Complex_test_results['Complex_MAPE']
        Complex_final_prediction = Complex_forecast(
                                        forecast_period,
                                        user_choice_models['auto_choice_models'],
                                        user_choice_models['SNaive_choice'],
                                        user_choice_models['St_m_choice'],
                                        user_choice_models['HW_choice'],
                                        user_choice_models['SARIMA_choice'],
                                        user_choice_models['LSTM_choice'],
                                        SNaive_final_prediction,
                                        St_m_final_prediction,
                                        HW_final_prediction,
                                        SARIMA_final_prediction,
                                        LSTM_final_prediction
                                        )
        # Complex: комплексный прогноз - график.
        plot_graph(
                    model_name='Complex',
                    path=folders['graphs_folder_path'],
                    train_data=train_data,
                    test_data=test_data,
                    chosen_tb_gosb_segm_type = chosen_tb_gosb_segm_type,
                    test_results=Complex_test_prediction,
                    prediction_results=Complex_final_prediction,
                    MAPE=Complex_MAPE
                    )

        ### 3.7 Сохраняем прогноз в датафрейм. ###
        forecasts_table = forecasts_to_dataframe(
                                                train_data=train_data,
                                                test_data=test_data,
                                                forecasts_table=forecasts_table,
                                                chosen_tb_gosb_segm_type=chosen_tb_gosb_segm_type,
                                                auto_choice_date=user_choice_date['auto_choice_date'],
                                                auto_choice_models=user_choice_models['auto_choice_models'],
                                                av_salary_fact=av_salary_fact,
                                                SNaive_final_prediction=SNaive_final_prediction,
                                                St_m_final_prediction=St_m_final_prediction,
                                                HW_final_prediction=HW_final_prediction,
                                                SARIMA_final_prediction=SARIMA_final_prediction,
                                                LSTM_final_prediction=LSTM_final_prediction,
                                                Complex_final_prediction=Complex_final_prediction,
                                                SNaive_MAPE=SNaive_MAPE,
                                                St_m_MAPE=St_m_MAPE,
                                                HW_MAPE=HW_MAPE,
                                                SARIMA_MAPE=SARIMA_MAPE,
                                                LSTM_MAPE=LSTM_MAPE,
                                                Complex_MAPE=Complex_MAPE,
                                                HW_best_trend=HW_best_trend,
                                                HW_best_seasonal=HW_best_seasonal,
                                                SARIMA_best_p=SARIMA_best_p,
                                                SARIMA_best_d=SARIMA_best_d,
                                                SARIMA_best_q=SARIMA_best_q,
                                                SARIMA_best_P=SARIMA_best_P,
                                                SARIMA_best_D=SARIMA_best_D,
                                                SARIMA_best_Q=SARIMA_best_Q,
                                                St_m_best_base_period=St_m_best_base_period
                                                )

        # Уведомляем пользователя об успешном завершении работы с сегментом.
        now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        print(f'{now} {chosen_tb_gosb_segm_type} ok')

    ### 4.0 Сохраняем результаты в файл ###
    save_file(fot_database, forecasts_table, tb_gosb_segm_type_choice, folders['main_folder_path'])