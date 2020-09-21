import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import datetime
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

scaler = MinMaxScaler()
train = r"C:\Users\Vijay\Desktop\Data Science Project\Training\\"
test = r"C:\Users\Vijay\Desktop\Data Science Project\Testing\\"
log_dir = r"model\logs\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)
csv_logger = tf.keras.callbacks.CSVLogger("logs.csv", separator = ",", append = False)

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (60, 5)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

for file in os.listdir(test):
    print(file)
    data = pd.read_csv(os.path.join(train, file), date_parser = True)
    data = data.drop(['Date'], axis = 1)
    data = scaler.fit_transform(data)
    x_train = []
    y_train = []
    for i in range(60, data.shape[0]):
        x_train.append(data[i-60:i])
        y_train.append(data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    model.fit(x_train, y_train, epochs = 10, batch_size = 32, callbacks = [csv_logger])

model.save("model")