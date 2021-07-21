# Week 4 assignment: Deployment on Flask
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Using Life Expectancy data from WHO (2000-2015)
# URL: https://www.kaggle.com/augustus0498/life-expectancy-who

lifeexp_data = pd.read_csv('~/Desktop/Data Glacier Virtual Internship May - Aug 2021/Week 4/led.csv')
# print(lifeexp_data.info())

# Data preprocessing
lifeexp_data.dropna(inplace = True)
# print(lifeexp_data.isnull().any())

# Multiple Linear Regression model building 
X = lifeexp_data[['AdultMortality', 'infantdeaths', 'Alcohol', 'BMI']] 
y = lifeexp_data['Lifeexpectancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

lm = LinearRegression()
lm.fit(X_train, y_train)

# Dumping the model in a pickle file
with open('model.pickle', 'wb') as file:
    pickle.dump(lm, file)
    file.close() 