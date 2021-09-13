import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


def LSTM_test(train_data, test_data, av_salary_fact):
    '''Функция тестирования LSTM. Мощная и популярная рекуррентная нейронная сеть - это модель долгосрочной сети или LSTM.'''

    # Отфильтровываем номинальную з/п, пока функция не будет проработана.
    train_data, test_data, av_salary_fact = train_data[['av_salary']], test_data[['av_salary']], av_salary_fact[['av_salary']]

    scaler = MinMaxScaler()
    scaler.fit(train_data)
    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)
    n_input = 12
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit_generator(generator, epochs=100, verbose=0)
    test_predictions = []
    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))

    for i in range(len(test_data)):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    true_predictions = scaler.inverse_transform(test_predictions)
    LSTM_MAPE = mean_absolute_error(test_data, true_predictions) / np.mean(test_data)[0]
    test_prediction_LSTM = pd.DataFrame(index=test_data.index, data=true_predictions, columns=['LSTM'])

    # Присваиваем название столбцу.
    test_prediction_LSTM.columns = ['av_salary']

    return {
            'test_prediction_LSTM': test_prediction_LSTM,
            'LSTM_MAPE': LSTM_MAPE
            }


def LSTM_forecast(train_data, test_data, av_salary_fact, forecast_period):
    '''Функция прогнозирования LSTM: мощная и популярная рекуррентная нейронная сеть - это модель долгосрочной сети или LSTM.'''
 
    # Отфильтровываем номинальную з/п, пока функция не будет проработана.
    train_data, test_data, av_salary_fact = train_data[['av_salary']], test_data[['av_salary']], av_salary_fact[['av_salary']]
 
    forecast_base = pd.concat([train_data, test_data])

    scaler = MinMaxScaler()
    scaler.fit(forecast_base)
    scaled_train = scaler.transform(forecast_base)

    n_input = 12
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit_generator(generator, epochs=100, verbose=0)

    test_predictions = []

    first_eval_batch = scaled_train[-n_input:]

    current_batch = first_eval_batch.reshape((1, n_input, n_features))

    for i in range(len(forecast_period)):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred) 
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    true_predictions = scaler.inverse_transform(test_predictions)
    LSTM_predictions = pd.DataFrame(index=forecast_period, data=true_predictions, columns=['LSTM'])

    # Присваиваем название столбцу.
    LSTM_predictions.columns = ['av_salary']

    return LSTM_predictions
