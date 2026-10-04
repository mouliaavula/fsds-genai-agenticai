# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 21:08:36 2026
Simple Linear Regression:y=mx+c
@author: Mouli
"""
# random_state=0 makes that random selection repeatable.
# Run 1 → different rows
# Run 2 → different rows
# you get the same split each time.
# It is basically a fixed random seed.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pickle

data_set = pd.read_csv(r"E:\vs-code-prakash\data\Salary_Data.csv")
# check shape of data set
print(data_set.shape)
# Feature selection independent variable X and dependen/target is y
X = data_set.iloc[:,:-1]
y = data_set.iloc[:,-1]

# Split the data set into train and test sets (train set 80% and test set 20%)
X_train, X_test,y_train,y_test = train_test_split(X, y,test_size=0.2, random_state=0)

#Fit the linear regression model to training set
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# predicting the result for test set

y_predict = regression_model.predict(X_test)

# Visualizing training set results
# In regression, targets are continuous real numbers (floats).
# Because of floating-point precision and continuous variance, 
# a model’s prediction (y_predict) will almost never be exactly 
# equal to the actual continuous label (y_test).
# Instead of exact equality, check if the error is within an acceptable margin like below
threshold = 1000
error = np.abs(y_test - y_predict)
plt.scatter(X_train, y_train,color='green', label='Training Sample')
plt.scatter(X_test, y_test,color='black', marker='s',label='Testing Sample')
plt.scatter(X_test[error <= threshold], y_test[error <= threshold],color='blue', marker='*',label='Test Sample Predicted Correct')
plt.scatter(X_test[error > threshold], y_test[error > threshold],color='red', marker='*',label='Test Sample Wrongly Predicted')
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.plot(X_train, regression_model.predict(X_train), color = 'blue')  
# Regression line from training set
plt.legend()
plt.show()


# Predict salary for 12 and 20 years of experience using the trained model
y_12 = regression_model.predict([[12]])
y_20 = regression_model.predict([[20]])
print(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f"Predicted salary for 20 years of experience: ${y_20[0]:,.2f}")



# Coefficients of the linear model
print(f"Intercept:{regression_model.intercept_}")
print(f"Coefficient or slope:{regression_model.coef_}")

# Regressor score:Evaluates model performance on the training data.
# we can also call this as R2 score on training data

bias = regression_model.score(X_train, y_train)
print("\n Regression score/bias: ",bias)

# Regressor score:Evaluates model performance on unseen test data.
# we can also call this as R2 score on test data
variance = regression_model.score(X_test, y_test)
print("\n Regression score/variance: ",variance)


# Check model performance
bias = regression_model.score(X_train, y_train)
variance = regression_model.score(X_test, y_test)
train_mse = mean_squared_error(y_train, regression_model.predict(X_train))
test_mse = mean_squared_error(y_test, y_predict)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Save the trained model to disk
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regression_model, file)
print("Model has been pickled and saved as linear_regression_model.pkl")


# stats for ml 
# Compare predicted and actual salaries from the test set
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
print(comparison)

#STATISTICS FOR MACHINE LEARNING

data_set.mean()

data_set['Salary'].mean() 

data_set.median() 

data_set['Salary'].mode()

data_set.describe()

data_set.var() 

data_set.std()

data_set.corr()


print("==================Test set  Results====================================")
# Actual test values
y_actual = y_test

# Predicted test values
y_predicted = y_predict

# Mean of actual test values 
y_mean = np.mean(y_actual)

# SST - Total Sum of Squares
SST = np.sum((y_actual-y_mean)**2)

# SSE - Error Sum of Squares
SSE = np.sum((y_actual-y_predicted)**2)

#SSR - Regression Sum of Suqares
SSR = np.sum((y_predicted-y_mean)**2)

# R2
r_square_one_minus = 1 - (SSE / SST)
r_square = SSR / SST

print("==================Test set  Results====================================")
print("SST:", SST)
print("SSE:", SSE)
print("SSR:", SSR)
print("R2 from 1-SSE/SST:", r_square_one_minus)
print("R2 from SSR/SST:", r_square)

print("R2 from SKlearn r2_score:", r2_score(y_test, y_predict))


print("=====================================Train set results======================")


# Actual test values
y_actual = y_train

# Predicted test values
y_predicted = regression_model.predict(X_train)

# Mean of actual test values 
y_mean = np.mean(y_actual)

# SST - Total Sum of Squares
SST = np.sum((y_actual-y_mean)**2)

# SSE - Error Sum of Squares
SSE = np.sum((y_actual-y_predicted)**2)

#SSR - Regression Sum of Suqares
SSR = np.sum((y_predicted-y_mean)**2)

# R2
r_square_one_minus = 1 - (SSE / SST)
r_square = SSR / SST

print("==================Train set Results====================================")
print("SST:", SST)
print("SSE:", SSE)
print("SSR:", SSR)
print("R2 from 1-SSE/SST:", r_square_one_minus)
print("R2 from SSR/SST:", r_square)

print("R2 from SKlearn r2_score:", r2_score(y_train, y_predicted))

"""
SST  = SSE + SSR on train dataset and all below 3 are equal on train data set 
R² = 1 - SSE/SST 
R² = SSR/SST 
R² = r2_score from sklearn
all 3 values are equal


but on test data set SST  !=  SSE + SSR  due to some cross term value and that values is zero on train dataset as linear line fitted for trains set 
but it is not zero on test dataset so that 

R² = 1 - SSE/SST  = r2_score from sklearn
but R² = SSR/SST != (1 - SSE/SST  = r2_score from sklearn)

So for test dataset R² we have to see : 1 - SSE/SST or  r2_score from sklearn not SSR/SST
"""
