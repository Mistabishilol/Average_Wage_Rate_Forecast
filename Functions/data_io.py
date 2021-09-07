import numpy as np
import pandas as pd
import datetime
from pathlib import Path

########## Функция: загрузка данных из базы планирования ##########

def data_load():
    '''Функция для загрузки данных из базы планирования.'''

    # Загрузка базы прогнозирования.
    try:
        fot_database = pd.read_csv('./Base/fot_database.csv', encoding = 'utf8', sep = '\t', index_col = 'report_date', parse_dates = ['report_date'])
        print('Данные из базы планирования успешно загружены.')
    except:
        print('Ошибка загрузки данных из базы планирования.')

    # Загрузка базы с сегментами.   
    try:    
        fot_database_dict = pd.read_csv('./Base/fot_database_dict.csv', encoding = 'utf8', sep = '\t')
        print('Данные из базы сегментов успешно загружены.')
    except:
        print('Ошибка загрузки егментов из базы планирования.')

    # Определение уникальных сегментов.
    try:
        # Все уникальные сочетания ТБ, ГОСБ, сегментов и типов.
        all_sections = fot_database[['tb_korr', 'gosb_korr', 'segm_korr', 'type']].drop_duplicates()
    except:
        print('Ошибка определения сегментов.')

    # Загрузка номильной средней ЗП из файла.
    try:
        df_avg_nominal_sal = pd.read_excel('./External variables/Nominal average salary.xlsx', parse_dates=['report_date'], index_col='report_date')
        
    
        fot_database = fot_database.reset_index()
        df_avg_nominal_sal = df_avg_nominal_sal.reset_index()
        cols_merge = ['report_date', 'tb_korr', 'gosb_korr']
        fot_database = pd.merge(left=fot_database,
                                right=df_avg_nominal_sal,
                                how='left',
                                left_on=cols_merge,
                                right_on=cols_merge
                                )

        fot_database = fot_database.set_index('report_date')
        print('Данные по номинальной средней ЗП успешно загружены.')
    except FileNotFoundError:
        print('Данные по номинальной средней ЗП не были загружены, так как файл отсутствует.')
        df_avg_nominal_sal = ''

    
    # Создаем пустой датафрейм.
    cols = [
            'SNaive',
            'St_m',
            'HW',
            'SARIMA',
            'LSTM',
            'Complex',
            'tb_gosb_segm_type',
            'SNaive_MAPE',
            'St_m_MAPE',
            'HW_MAPE',
            'SARIMA_MAPE',
            'LSTM_MAPE',
            'Complex_MAPE',
            'St_m_param',
            'HW_param',
            'SARIMA_param',
            'Train_period_start',
            'Train_period_end',
            'Test_period_start',
            'Test_period_end',
            'Train_period_len',
            'Test_period_len',
            'Train_len / (Train+Test)_len',
            'Test_len / Forecast_len',
            'Forecast_type',
            'Date_time',
            'Comment'
            ]
    forecasts_table = pd.DataFrame(columns = cols)

    return {'fot_database': fot_database,
            'fot_database_dict': fot_database_dict,
            'forecasts_table': forecasts_table,
            'all_sections': all_sections}


########## Функция: определение границ периодов ##########

def periods_choice(i, fot_database, tb_gosb_segm_type_choice, auto_choice_date, train_date_start, train_date_end, test_date_end, forecast_date_end):
    '''Определение границ обучающего, тестового и прогнозного периода'''

    if 'rs_avg_nom_sal' in fot_database.columns:
        av_salary_fact = fot_database[['av_salary', 'rs_avg_nom_sal']][fot_database['tb_gosb_segm_type'] == tb_gosb_segm_type_choice[i]]
    else:
        av_salary_fact = fot_database[['av_salary']][fot_database['tb_gosb_segm_type'] == tb_gosb_segm_type_choice[i]]

    av_salary_fact = av_salary_fact.fillna(0)
    av_salary_fact = av_salary_fact.replace(np.inf, 0)
    
    if auto_choice_date in ['yes', 'да']:
        train_border = round(len(av_salary_fact) * 0.8)
        train_data = av_salary_fact[:train_border].fillna(0)
        test_data = av_salary_fact[train_border:].fillna(0)
        forecast_period = pd.date_range(start = pd.date_range(start = test_data.index[-1], periods = 2, freq = 'M')[1], periods = len(test_data), freq = 'M')
        chosen_tb_gosb_segm_type = tb_gosb_segm_type_choice[i]
    else:
        train_data = av_salary_fact[train_date_start:train_date_end].fillna(0)
        train_border = len(train_data)
        test_date_start = pd.date_range(start = train_date_end, periods = 2, freq = 'M')[1]
        test_data = av_salary_fact[test_date_start:test_date_end].fillna(0)
        forecast_period_start = pd.date_range(start = test_date_end, end = forecast_date_end, freq = 'M')[1]
        forecast_period = pd.date_range(start = forecast_period_start, end = forecast_date_end, freq = 'M')
        chosen_tb_gosb_segm_type = tb_gosb_segm_type_choice[i]

    return {'av_salary_fact': av_salary_fact,
            'train_border': train_border,
            'train_data': train_data,
            'test_data': test_data,
            'chosen_tb_gosb_segm_type': chosen_tb_gosb_segm_type,
            'forecast_period': forecast_period
            }


########## Функция для создания папки ##########

def create_folders():
    '''Функция для создания папки, куда будут складываться прогнозы и графики.'''

    # Основная папка.
    main_folder_path = './Predictions/' + datetime.datetime.now().strftime("%d.%m.%Y %H-%M-%S")
    Path(main_folder_path).mkdir(parents=True, exist_ok=True)

    # Папка для ошибок.
    error_folder_path = main_folder_path +'/Errors/'
    Path(main_folder_path).mkdir(parents=True, exist_ok=True)

    # Папка для графиков.
    graphs_folder_path = main_folder_path + '/Graphs/'
    Path(graphs_folder_path).mkdir(parents=True, exist_ok=True)   

    return {
            'main_folder_path': main_folder_path,
            'error_folder_path': error_folder_path,
            'graphs_folder_path': graphs_folder_path
            }


########## Функция: запись прогнозов в единый датафрейм. ##########

def forecasts_to_dataframe(
                            train_data,
                            test_data,
                            forecasts_table,
                            chosen_tb_gosb_segm_type,
                            auto_choice_date,
                            auto_choice_models,
                            av_salary_fact,
                            SNaive_final_prediction,
                            St_m_final_prediction,
                            HW_final_prediction,
                            SARIMA_final_prediction,
                            LSTM_final_prediction,
                            Complex_final_prediction,
                            SNaive_MAPE,
                            St_m_MAPE,
                            HW_MAPE,
                            SARIMA_MAPE,
                            LSTM_MAPE,
                            Complex_MAPE,
                            HW_best_trend,
                            HW_best_seasonal,
                            SARIMA_best_p,
                            SARIMA_best_d,
                            SARIMA_best_q,
                            SARIMA_best_P,
                            SARIMA_best_D,
                            SARIMA_best_Q,
                            St_m_best_base_period
                            ):
    '''Сохранение прогнозов в датафрейм'''
    
    forecasts = pd.DataFrame(data = (
                                    SNaive_final_prediction['av_salary'],
                                    St_m_final_prediction['av_salary'],
                                    HW_final_prediction['av_salary'],
                                    SARIMA_final_prediction['av_salary'],
                                    LSTM_final_prediction['av_salary'],
                                    Complex_final_prediction['av_salary'])
                                    ).transpose()

    forecasts.columns = ['SNaive', 'St_m', 'HW', 'SARIMA', 'LSTM', 'Complex']
    forecasts['tb_gosb_segm_type'] = chosen_tb_gosb_segm_type
    forecasts['SNaive_MAPE'] = SNaive_MAPE
    forecasts['St_m_MAPE'] = St_m_MAPE
    forecasts['HW_MAPE'] = HW_MAPE
    forecasts['SARIMA_MAPE'] = SARIMA_MAPE
    forecasts['LSTM_MAPE'] = LSTM_MAPE
    forecasts['Complex_MAPE'] = Complex_MAPE
    forecasts['St_m_param'] = f'{St_m_best_base_period}'
    forecasts['HW_param'] = f'tr = {HW_best_trend}, se = {HW_best_seasonal}'
    forecasts['SARIMA_param'] = f'({SARIMA_best_p}, {SARIMA_best_d}, {SARIMA_best_q}) x ({SARIMA_best_P}, {SARIMA_best_D}, {SARIMA_best_Q}, 12)'
    forecasts['Train_period_start'] = train_data.index[0]
    forecasts['Train_period_end'] = train_data.index[-1]
    forecasts['Test_period_start'] = test_data.index[0]
    forecasts['Test_period_end'] = test_data.index[-1]
    forecasts['Train_period_len'] = len(train_data)
    forecasts['Test_period_len'] = len(test_data)
    forecasts['Train_len / (Train+Test)_len'] = len(train_data) / len(av_salary_fact)
    forecasts['Test_len / Forecast_len'] = len(test_data) / len(Complex_final_prediction)
    
    if auto_choice_date in ['yes', 'да']:
        forecasts['Period_type'] = 'Auto'
    else:
        forecasts['Period_type'] = 'Manual'
    if auto_choice_models in ['yes', 'да']:
        forecasts['Forecast_type'] = 'Auto'
    else:
        forecasts['Forecast_type'] = 'Manual'
    forecasts['Date_time'] = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    forecasts_table = pd.concat([forecasts_table, forecasts])
    
    return forecasts_table


########## Функция для сохранения результатов построения прогнозов. ##########

def save_file(fot_database, forecasts_table, tb_gosb_segm_type_choice, folder_path):
    '''Сохраняем прогноз в файл. Если нет подпапки с текущей датой/временем - создаем.'''

    now = datetime.datetime.now().strftime("%d.%m.%Y %H-%M-%S")

    file_name = f'Results_{now}.xlsx'
    need_comment = input('Дополнить расчёт комментарием или пояснением? (yes, да) / (no, нет): ')
    
    if need_comment in ['yes', 'да']:
        user_comment = input('Введите комментарий или пояснение к Вашему расчёту: ')
        forecasts_table['Comment'] = user_comment
    
    writer = pd.ExcelWriter(folder_path + '//' + file_name, engine='xlsxwriter')
    with writer:
        forecasts_table.to_excel(writer, sheet_name='Прогнозы')
    
    print(f'Файл {file_name} сохранён в папке {folder_path}.\nПерезапустите ячейку для нового расчёта.')