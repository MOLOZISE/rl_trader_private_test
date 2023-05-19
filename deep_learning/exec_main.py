import settings
import data_manager
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input
from keras.layers import BatchNormalization
from keras.layers import Dropout
import keras.backend as K
from keras.callbacks import EarlyStopping, callbacks

if __name__ == '__main__':

    # Keras Backend 설정
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    from networks import Network, DNN, LSTMNetwork, CNN
    network_name = 'test_01'
    network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(network_name))
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    # 분봉의 경우
    # start_time = args.date + "090000"  # 9시 10분부터
    # end_time = args.date + "150000"  # 15시 까지
    start_time = "20190101"
    end_time = "20210104"
    lr = 0.001
    epochs = 100
    list_stock_code.append("005930")
    for stock_code in list_stock_code:
        chart_data, training_data = data_manager.load_data(
            os.path.join(settings.BASE_DIR, 'files/OSSP_KOSPI/{}_day_data.txt'.format(stock_code)), ver='ossp_lstm',
            start_time=start_time, end_time=end_time)
        K.clear_session()
        ####################################LSTM
        num_steps = 5
        input_dims = training_data.shape[2]
        output_dims = 1
        n_hidden = 32
        dropout_rate = 0.1
        model_path = None
        model_name = "lstm01"
        save_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(model_name))
        #LSTM model
        lstmmodel = Sequential()  # Sequeatial Model
        lstmmodel.add(LSTM(128, return_sequences=True, activation='tanh', kernel_initializer='he_normal', input_shape=(num_steps, input_dims)))  # (timestep, feature)
        lstmmodel.add(BatchNormalization())
        lstmmodel.add(Dropout(dropout_rate))
        for i in range(n_hidden-2):
            if (i / 8 > 1):
                lstmmodel.add(LSTM(64, return_sequences=True, activation='tanh', kernel_initializer='he_normal'))
                lstmmodel.add(BatchNormalization())
                lstmmodel.add(Dropout(dropout_rate))
                continue
            if (i / 8 > 2):
                lstmmodel.add(LSTM(32, return_sequences=True, activation='tanh', kernel_initializer='he_normal'))
                lstmmodel.add(BatchNormalization())
                lstmmodel.add(Dropout(dropout_rate))
                continue
            if (i / 8 > 3):
                lstmmodel.add(LSTM(16, return_sequences=True, activation='tanh', kernel_initializer='he_normal'))
                lstmmodel.add(BatchNormalization())
                lstmmodel.add(Dropout(dropout_rate))
                continue
            lstmmodel.add(LSTM(128, return_sequences=True, activation='tanh', kernel_initializer='he_normal'))
            lstmmodel.add(BatchNormalization())
            lstmmodel.add(Dropout(dropout_rate))
        lstmmodel.add(LSTM(16, activation='tanh', kernel_initializer='he_normal'))
        lstmmodel.add(Dense(1))  # output = 1
        lstmmodel.compile(loss='mean_squared_error', optimizer='adam')
        checkpoint_cb = callbacks.ModelCheckpoint(save_path, save_best_only=True)
        early_stopping_cb = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        # converting
        #lstm_chart_data = np.array(chart_data).reshape((-1, num_steps, output_dims))
        history = lstmmodel.fit(training_data, chart_data, epochs=epochs, validation_split=0.1,
                                     # validation_data=(X_valid, Y_valid),
                                     callbacks=[checkpoint_cb, early_stopping_cb], batch_size=8, verbose=2)

        test_start_time = "20200105"
        test_end_time = "20210101"
        test_chart_data, test_training_data = data_manager.load_data(
            os.path.join(settings.BASE_DIR, 'files/OSSP_KOSPI/{}_day_data.txt'.format(stock_code)), ver='ossp_lstm',
            start_time=test_start_time, end_time=test_end_time)
        score = lstmmodel.evaluate(test_training_data, test_chart_data, verbose=0)
        print('Test loss:', score)
        print('Test accuracy:', score)

        # num_steps = 5
        # input_dims = training_data.shape[2]
        # output_dims = 1
        # n_hidden = 32
        # dropout_rate = 0.1
        # model_path = None
        # model_name = "cnn01"
        # save_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(model_name))
        # # cnnmodel
        # # cnn_training_data = np.array(training_data).reshape((-1, input_dims*num_steps))
        # # print(cnn_training_data.shape)
        # print(training_data.shape)
        # cnnmodel = Sequential()  # Sequeatial Model
        # cnnmodel.add(Conv2D(128, kernel_size=(1, 5),
        #                 padding='same', activation='tanh',
        #                 kernel_initializer='he_normal', input_shape=(num_steps, input_dims, 1)))
        # cnnmodel.add(BatchNormalization())
        # cnnmodel.add(MaxPooling2D(pool_size=(1, 2)))
        # cnnmodel.add(Dropout(0.1))
        # for i in range(n_hidden - 2):
        #     if (i / 8 > 1):
        #         cnnmodel.add(Conv2D(64, kernel_size=(1, 5),
        #                             padding='same', activation='tanh',
        #                             kernel_initializer='he_normal'))
        #         cnnmodel.add(BatchNormalization())
        #         cnnmodel.add(MaxPooling2D(pool_size=(1, 2)))
        #         cnnmodel.add(Dropout(0.1))
        #         continue
        #     if (i / 8 > 2):
        #         cnnmodel.add(Conv2D(32, kernel_size=(1, 5),
        #                             padding='same', activation='tanh',
        #                             kernel_initializer='he_normal'))
        #         cnnmodel.add(BatchNormalization())
        #         cnnmodel.add(MaxPooling2D(pool_size=(1, 2)))
        #         cnnmodel.add(Dropout(0.1))
        #         continue
        #     if (i / 8 > 3):
        #         cnnmodel.add(Conv2D(16, kernel_size=(1, 5),
        #                             padding='same', activation='tanh',
        #                             kernel_initializer='he_normal'))
        #         cnnmodel.add(BatchNormalization())
        #         cnnmodel.add(MaxPooling2D(pool_size=(1, 2)))
        #         cnnmodel.add(Dropout(0.1))
        #         continue
        #     cnnmodel.add(Conv2D(16, kernel_size=(1, 5),
        #                         padding='same', activation='tanh',
        #                         kernel_initializer='he_normal'))
        #     cnnmodel.add(BatchNormalization())
        #     cnnmodel.add(MaxPooling2D(pool_size=(1, 2)))
        #     cnnmodel.add(Dropout(0.1))
        # cnnmodel.add(Flatten())
        # cnnmodel.add(Dense(output_dims))
        # cnnmodel.compile(loss='mean_squared_error', optimizer='adam')
        # checkpoint_cb = callbacks.ModelCheckpoint(save_path, save_best_only=True)
        # early_stopping_cb = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        # # converting
        # # lstm_chart_data = np.array(chart_data).reshape((-1, num_steps, output_dims))
        # history = cnnmodel.fit(cnn_training_data, chart_data, epochs=epochs, validation_split=0.1,
        #                         # validation_data=(X_valid, Y_valid),
        #                         callbacks=[checkpoint_cb, early_stopping_cb], batch_size=8, verbose=2)
        #
        # test_start_time = "20200105"
        # test_end_time = "20210101"
        # test_chart_data, test_training_data = data_manager.load_data(
        #     os.path.join(settings.BASE_DIR, 'files/OSSP_KOSPI/{}_day_data.txt'.format(stock_code)), ver='ossp_lstm',
        #     start_time=test_start_time, end_time=test_end_time)
        # cnn_training_data = np.array(test_training_data).reshape((-1, input_dims * num_steps))
        # print(cnn_training_data.shape)
        # score = cnnmodel.evaluate(cnn_training_data, test_chart_data, verbose=0)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])

        # DNN
        # dnn = DNN(input_dim=training_data.shape[1], output_dim=1, lr=lr, activation='tanh', loss='mse')
        # history1 = dnn.model.fit(training_data, chart_data, epochs=epochs, batch_size=8, verbose=2)
        # CNN
        #cnn = CNN(input_dim=training_data.shape[1], output_dim=1, lr=lr, activation='tanh', loss='mse', num_steps=5)
        # cnn_training_data = np.array(training_data).reshape((-1, cnn.num_steps, cnn.input_dim, 1))
        # print(cnn_training_data.shape)
        # cnn_chart_data = np.array(chart_data).reshape((-1, cnn.num_steps, cnn.output_dim, 1))
        # print(cnn_chart_data.shape)
        # history2 = cnn.model.fit(cnn_training_data, cnn_chart_data, epochs=epochs, batch_size=8, verbose=2)
        # LSTM
        # lstm = LSTMNetwork(input_dim=training_data.shape[1], output_dim=1, lr=lr, activation='tanh', loss='mse', num_steps=5)
        # lstm_training_data = np.array(training_data).reshape((-1, lstm.num_steps, lstm.input_dim))
        # print(lstm_training_data.shape)
        # lstm_chart_data = np.array(chart_data).reshape((-1, lstm.num_steps, lstm.output_dim))
        # print(lstm_chart_data.shape)
        # history3 = lstm.model.fit(lstm_training_data, lstm_chart_data, epochs=epochs, batch_size=8, verbose=2)

