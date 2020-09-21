# Stock-Prediction-AI
Neural network written for CS 439 (Intro to Data Science) at Rutgers University.

Our goal with this project is to simulate what the stock market would have looked like in the past few months if the COVID-19 pandemic hadn’t happened. The best way to accomplish this would be to train an AI on the past performance of a strong, stable subset of the market and then feed it data from the last few months of 2019 in order to generate data for the first few months of 2020.

Our data was gathered from the Yahoo Finance database. Since it would be impossible for us to analyze every single stock on the market over their entire lifetime, due to time and computational constraints, we narrowed the scope of our training set to data on stocks in the S&P 100 index, starting from January 1st, 2015 and ending on December 31st, 2018. Following an approximation of the 70/30 split rule for splitting data, our test data set is all of the information on those companies from January 1st, 2019 to December 31st, 2019. We chose the S&P 100 since it’s widely regarded as a well curated sample of strong companies on the US exchanges with low volatility, high volume of trade, and widely representative of all business sectors.
