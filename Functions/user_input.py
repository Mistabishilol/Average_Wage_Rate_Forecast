import numpy as np

# Функция: выбор сегментов и отделений для прогноза

def branch_choice(fot_database_dict, all_sections):
    '''Выбор сегментов и отделений для прогноза'''
    
    tb_gosb_segm_type_choice = []
    tb_choice = []
    gosb_choice = []
    segm_choice = []
    type_choice = []

    ###### Выбор ТБ
    
    tb_avail = all_sections['tb_korr'].unique()
    print('Укажите отделения и сегменты для построения прогноза. Можно указать несколько отделений или сегментов через пробел.',
    '\nДля выбора всех отделений или сегментов используйте слово "all" или "все" без кавычек.',
    '\nДоступны следующие ТБ:',
    *tb_avail,
    sep=' '
    )

    tb_choice = input('ТБ: ').lower().split(' ')

    if tb_choice in [['all'], ['все']]:
        tb_choice = all_sections['tb_korr'].unique()
    
    tb_choice = [int(x) for x in tb_choice]
    
    ###### Выбор ГОСБ
    
    gosb_avail = np.unique(all_sections[all_sections['tb_korr'].isin(tb_choice)]['gosb_korr'].values)
    print('Для выбранных ТБ доступны следующие ГОСБ:', *gosb_avail, sep=' ')
    gosb_choice = input('ГОСБ: ').lower().split(' ')
    if gosb_choice in [['all'], ['все']]:
        gosb_choice = all_sections['gosb_korr'].unique()

    gosb_choice = [int(x) for x in gosb_choice]
    
    ###### Выбор сегмента
    
    segm_avail = np.unique(all_sections[(all_sections['tb_korr'].isin(tb_choice)) &
                                        (all_sections['gosb_korr'].isin(gosb_choice))
                                        ]['segm_korr'].values)
    print('Для выбранных ТБ и ГОСБ доступны следующие сегменты:', *segm_avail, sep=' ')
    segm_choice = input('Сегмент: ').split(' ')
    if segm_choice in [['all'], ['все']]:
        segm_choice = all_sections['segm_korr'].unique()
    
    
    ###### Выбор типа
    
    types_avail = np.unique(all_sections[(all_sections['tb_korr'].isin(tb_choice)) &
                                        (all_sections['gosb_korr'].isin(gosb_choice)) &
                                        (all_sections['segm_korr'].isin(segm_choice))
                                        ]['type'].values)
    print('Для выбранных ТБ и ГОСБ доступны следующие типы:', *types_avail, sep=' ')
    type_choice = input('Тип: ').split(' ')
    if type_choice in [['all'], ['все']]:
        type_choice = all_sections['type'].unique()
    

    index_avail = np.unique(all_sections[(all_sections['tb_korr'].isin(tb_choice)) &
                                        (all_sections['gosb_korr'].isin(gosb_choice)) &
                                        (all_sections['segm_korr'].isin(segm_choice) &
                                        (all_sections['type'].isin(type_choice)))
                                        ]['type'].values)
    ###### Собираем выбранные ТБ/ГОСБ/Сегменты/Типы в единый атрибут.
    for tb in tb_choice:
        for gosb in gosb_choice:
            for segm in segm_choice:
                for tp in type_choice:
                    a = f'{tb}_{gosb}_{segm}_{tp}'
                    if a in fot_database_dict['fot_database_dict'].unique():
                        tb_gosb_segm_type_choice.append(a)

    return tb_gosb_segm_type_choice


def choice_date():
    '''Просим пользователя выбрать даты.'''
    
    auto_choice_date = input('Определить границы периодов автоматически? (yes, да) / (no, нет): ')
    
    if auto_choice_date not in ['yes', 'да']:
        print('Введите даты в формате YYYY-MM-DD:')
        train_date_start = input('Начало обучающего периода: ')
        train_date_end = input('Окончание обучающего периода (должен быть не менее 12 месяцев, рекомендуется около 80% от доступных данных): ')
        test_date_end = input('Окончание тестового периода (должен быть не более обучающего, рекомендуется около 20% от доступных данных): ')
        forecast_date_end = input('Окончание прогнозного периода (рекомендуется не более тестового периода): ')
    else:
        train_date_start = train_date_end = test_date_end = forecast_date_end = '2017-01-01'
        
    return {'auto_choice_date': auto_choice_date,
            'train_date_start': train_date_start,
            'train_date_end': train_date_end,
            'test_date_end': test_date_end,
            'forecast_date_end': forecast_date_end}


def choice_models():
    '''Просим пользователя выбрать алгоритмы и их параметры'''

    base_period = trend_dict = seasonal_dict = ''
    SARIMA_p_ = SARIMA_d_ = SARIMA_q_ = SARIMA_P_ = SARIMA_D_ = SARIMA_Q_ = [0, 1]    

    auto_choice_models = input('Использовать все алгоритмы прогнозирования с автоматическими настройками? (yes, да) / (no, нет): ')
    if auto_choice_models not in ['yes', 'да']:
        SNaive_choice = input('Использовать наивный сезонный алгоритм SNaive? (yes, да) / (no, нет): ')
        St_m_choice = input('Использовать стандартный алгоритм (сезонность к базе) St_m? (yes, да) / (no, нет): ')
        
        if St_m_choice in ['yes', 'да']:
            St_m_param_auto = input('Использовать автоматический поиск оптимального базового периода для St_m? (yes, да) / (no, нет): ')
            
            if St_m_param_auto not in ['yes', 'да']:
                base_period = int(input('Укажите количество месяцев базового периода для усреднения: '))
        else:
            St_m_param_auto = 'no'
            base_period = ''
        
        HW_choice = input('Использовать алгоритм тройного экспоненциального сглаживания Хольта-Винтерса HW? (yes, да) / (no, нет): ')
        if HW_choice in ['yes', 'да']:
            HW_param_auto = input('Использовать автоматический выбор тренда и сезонности для HW? (yes, да) / (no, нет): ')
            
            if HW_param_auto not in ['yes', 'да']:
                trend_dict = [input('Выберите тип тренда: мультипликативный (mul) или аддитивный (add): ')]
                seasonal_dict = [input('Выберите тип сезонности: мультпликативная (mul) или аддитивная (add): ')]
        else:
            HW_param_auto = 'no'
            trend_dict = ''
            seasonal_dict = ''
        
        SARIMA_choice = input('Использовать сезонную авторегрессионную интегрированную модель скользящего среднего Бокса-Дженкинса SARIMA? (yes, да) / (no, нет): ')
        if SARIMA_choice in ['yes', 'да']:
            SARIMA_param_auto = input('Использовать автоматический выбор параметров для SARIMA (0 / 1 для всех параметров)? (yes, да) / (no, нет): ')
            if SARIMA_param_auto not in ['yes', 'да']:
                SARIMA_grid_search = input('Введите yes/да для указания сетки, no/нет для указания конкретных порядков:')
                if SARIMA_grid_search in ['yes', 'да']:
                    SARIMA_pdq = input('Введите через пробел правую границу сетки (включительно) для порядков регрессии (p), дифференцирования (d), скользящего среднего (q): ').split(' ')
                    SARIMA_PDQ = input('Введите через пробел правую границу сетки (включительно) для порядков регрессии (P), дифференцирования (D), скользящего среднего (Q) для сезонности: ').split(' ')
                    SARIMA_p_ = [x for x in range(0, int(SARIMA_pdq[0])+1) if x < int(SARIMA_pdq[0])+1]
                    SARIMA_d_ = [x for x in range(0, int(SARIMA_pdq[1])+1) if x < int(SARIMA_pdq[1])+1]
                    SARIMA_q_ = [x for x in range(0, int(SARIMA_pdq[2])+1) if x < int(SARIMA_pdq[2])+1]
                    SARIMA_P_ = [x for x in range(0, int(SARIMA_PDQ[0])+1) if x < int(SARIMA_PDQ[0])+1]
                    SARIMA_D_ = [x for x in range(0, int(SARIMA_PDQ[1])+1) if x < int(SARIMA_PDQ[1])+1]
                    SARIMA_Q_ = [x for x in range(0, int(SARIMA_PDQ[2])+1) if x < int(SARIMA_PDQ[2])+1]               
                else:
                    SARIMA_pdq = input('Введите через пробел порядок регрессии (p), дифференцирования (d), скользящего среднего (q) для временного ряда: ').split(' ')
                    SARIMA_PDQ = input('Введите через пробел порядок регрессии (P), дифференцирования (D), скользящего среднего (Q) для сезонности: ').split(' ')
                    SARIMA_p_ = [int(SARIMA_pdq[0])]
                    SARIMA_d_ = [int(SARIMA_pdq[1])]
                    SARIMA_q_ = [int(SARIMA_pdq[2])]
                    SARIMA_P_ = [int(SARIMA_PDQ[0])]
                    SARIMA_D_ = [int(SARIMA_PDQ[1])]
                    SARIMA_Q_ = [int(SARIMA_PDQ[2])]
        else:
            SARIMA_param_auto = 'no'
            SARIMA_p_ = SARIMA_d_ = SARIMA_q_ = SARIMA_P_ = SARIMA_D_ = SARIMA_Q_ = [0, 1]

        LSTM_choice = input('Использовать нейросеть LSTM? (yes, да) / (no, нет): ')

    else:
        SNaive_choice = St_m_choice = HW_choice = SARIMA_choice = LSTM_choice = 'yes'
        St_m_param_auto = HW_param_auto = SARIMA_param_auto = 'yes'

        
    return {'auto_choice_models': auto_choice_models,
            'SNaive_choice': SNaive_choice,
            'St_m_choice': St_m_choice,
            'HW_choice': HW_choice,
            'SARIMA_choice': SARIMA_choice,
            'LSTM_choice': LSTM_choice,
            'St_m_param_auto': St_m_param_auto,
            'HW_param_auto': HW_param_auto,
            'SARIMA_param_auto': SARIMA_param_auto,
            'base_period': base_period,
            'trend_dict': trend_dict,
            'seasonal_dict': seasonal_dict,
            'SARIMA_p_': SARIMA_p_,
            'SARIMA_d_': SARIMA_d_,
            'SARIMA_q_': SARIMA_q_,
            'SARIMA_P_': SARIMA_P_,
            'SARIMA_D_': SARIMA_D_,
            'SARIMA_Q_': SARIMA_Q_
            }