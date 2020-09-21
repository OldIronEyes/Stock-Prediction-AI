import yfinance as yf
import pandas as pd

symbols = open("symbols.txt","r").readlines()

for symbol in symbols:
    stock = yf.download(symbol, start = "2020-01-01", end = "2020-05-01")
    location = r'c:\Users\Vijay\Desktop\Data Science Project\Actual Prices\\'
    fileName = location + symbol.replace('\n','') + ".csv"
    stock.to_csv(fileName, sep=",", columns =["Open", "High", "Low", "Close", "Volume"], index = True)