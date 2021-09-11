import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('fivethirtyeight')
from pathlib import Path

def plot_graph(model_name, path, train_data, test_data, chosen_tb_gosb_segm_type, test_results, prediction_results, MAPE):
    '''Функция для построения графика по отдельно выранным сегменту и модели.'''

    fig, ax = plt.subplots(figsize=(16, 12))

    ax.plot(train_data[['av_salary']], color = 'teal', label = 'Обучающие данные')
    ax.plot(test_data[['av_salary']], color = 'yellow', label = 'Тестовые данные')
    ax.plot(prediction_results[['av_salary']], color = 'limegreen', linestyle = '--', label = model_name + ' прогноз')
    ax.plot(test_results[['av_salary']], color = 'crimson', linestyle = '--', label = f'{model_name} test ({round(MAPE * 100, 1)}%)')
    plt.title(chosen_tb_gosb_segm_type + '_' + model_name,  loc='center')
    plt.legend(loc='upper left')

    # В основной папке создаем подпапку с названием сегмента.
    path = path + chosen_tb_gosb_segm_type + '/'
    Path(path).mkdir(parents=True, exist_ok=True)

    # Сохраняем и закрываем график.
    plt.savefig(path + chosen_tb_gosb_segm_type + '_' + model_name + '.png')
    plt.close()


def plot_common_graph(path,
                      train_data,
                      test_data,
                      chosen_tb_gosb_segm_type,
                      common_graph_dict):
    '''Функция для построения общего графика.'''
    graphs_num = len(common_graph_dict)
    fig, axs = plt.subplots(graphs_num, 1, figsize=(12, graphs_num*4))

    # Список ключей, счетчик для графика.
    key_list, i = list(common_graph_dict), 0
    
    for model_name in key_list:
        test_prediction = common_graph_dict[model_name][0]
        MAPE = common_graph_dict[model_name][1]
        final_prediction = common_graph_dict[model_name][2]

        # Каждый график отдельно.
        axs[i].plot(train_data[['av_salary']], color = 'teal', label = 'Обучающие данные')
        axs[i].plot(test_data[['av_salary']], color = 'yellow', label = 'Тестовые данные')
        axs[i].plot(final_prediction[['av_salary']], color = 'limegreen', linestyle = '--', label = model_name + ' прогноз')
        axs[i].plot(test_prediction[['av_salary']], color = 'crimson', linestyle = '--', label = f'{model_name} test ({round(MAPE * 100, 1)}%)')
        axs[i].set_title(chosen_tb_gosb_segm_type + '_' + model_name,  loc='center')
        axs[i].set_ylabel('Average wage rate', fontsize=16)
        axs[i].tick_params(axis='both', which='major', labelsize=16)
        axs[i].set_xlabel('Period', fontsize=16)
        axs[i].legend()
        
        i +=1

    fig.autofmt_xdate()

    # Сохраняем в основной папке.
    Path(path).mkdir(parents=True, exist_ok=True)

    # Сохраняем и закрываем график.
    plt.savefig(path + chosen_tb_gosb_segm_type + '.png')
    plt.show() 