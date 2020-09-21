import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.keras.models

scaler = MinMaxScaler()

test = r"C:\\Users\\Vijay\\Desktop\\Data Science Project\\Testing\\"
model = tf.keras.models.load_model(r"C:\\Users\\Vijay\\Desktop\\Data Science Project\\model\\", custom_objects = None, compile = True)

for file in os.listdir(test):
    print(file)
    data = pd.read_csv(os.path.join(test, file), date_parser = True)
    data = data.drop(['Date'], axis = 1)
    data = scaler.fit_transform(data)
    x_test = []
    y_test = []
    for i in range(60, data.shape[0]):
        x_test.append(data[i-60:i])
        y_test.append(data[i,0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    results = model.evaluate(x_test, y_test, batch_size = 32)
    print("test loss, test acc:", results)