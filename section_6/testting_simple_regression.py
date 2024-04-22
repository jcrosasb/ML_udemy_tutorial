import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # Reads all rows, from first column up to, but not inlcuding, the last
y = dataset.iloc[:, -1].values  # Reads all rows from last column

# print(f'Type of X = {type(X)}')
# print(f'Type of y = {type(y)}')

# Splits the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(f'Type of X_train = {type(X_train)}')
# print(f'Type of X_test = {type(X_test)}')
# print(f'Type of y_train = {type(y_train)}')
# print(f'Type of y_test = {type(y_test)}')


regressor = LinearRegression()
reg = regressor.fit(X_train, y_train)
print(type(reg))


#regressor.fit(X_train, y_train)