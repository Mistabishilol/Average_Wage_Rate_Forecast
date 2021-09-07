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


def plot_common_graph(model_name, path, train_data, test_data, chosen_tb_gosb_segm_type, SNaive_test_prediction, SNaive_final, MAPE):
    '''Функция для построения общего графика.'''