import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy

rTrain = r"C:\Users\Vijay\Desktop\Data Science Project\Raw Training"
rTest = r"C:\Users\Vijay\Desktop\Data Science Project\Raw Testing"

train = r"C:\Users\Vijay\Desktop\Data Science Project\Training"
test = r"C:\Users\Vijay\Desktop\Data Science Project\Testing"

rActual = r"C:\Users\Vijay\Desktop\Data Science Project\Raw Actual Prices"
actual = r"C:\Users\Vijay\Desktop\Data Science Project\Actual Prices"

scaler = MinMaxScaler()

for file in os.listdir(rActual):
    print(file)
    data = pd.read_csv(os.path.join(rActual, file))
    data["Average"] = data.apply(lambda row: (row.High + row.Low)/2, axis = 1)
    data = data.drop(["Date", "Open", "High", "Low", "Close", "Volume"], axis = 1)
    scaler.fit(data)
    data = scaler.transform(data)
    numpy.savetxt(os.path.join(actual, file), data, delimiter = ",")