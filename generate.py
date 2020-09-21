import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.keras.models

scaler = MinMaxScaler()

model = tf.keras.models.load_model(r"C:\\Users\\Vijay\\Desktop\\Data Science Project\\model\\", custom_objects = None, compile = True)

data = pd.read_csv("NVDA.csv")
data = data.drop(['Date'], axis = 1)
data = scaler.fit_transform(data)
x_test = []
y_test = []
for i in range(60, data.shape[0]):
    x_test.append(data[i-60:i])
    y_test.append(data[i,0])
x_test = np.array(x_test)
results = model.predict(x_test)
y_test = y_test/scaler.scale_[0]
results = results/scaler.scale_[0]

plt.figure(figsize=(10,5))
plt.plot(y_test, color = 'red', label = "Real NVidia Price")
plt.plot(results, color = 'blue', label = "Predicted NVidia Price")
plt.title("NVidia Price Simulation")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()