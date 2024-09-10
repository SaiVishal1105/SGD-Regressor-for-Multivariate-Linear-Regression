# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step 1: 
Load California housing data, select features and targets, and split into training and testing sets.
### Step 2:
Scale both X (features) and Y (targets) using StandardScaler.
### Step 3:
Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data.
### Step 4:
Predict on test data, inverse transform the results, and calculate the mean squared error.


## Program:
```
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SAI VISHAL D
RegisterNumber: 212223230180

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
```
![image](https://github.com/user-attachments/assets/527a974c-4a0f-4c21-9a23-40d2d927fd51)
```
df.info()
```
![image](https://github.com/user-attachments/assets/acad642a-6879-4dd5-acf6-3a623c06bcee)
```
X=df.drop(columns=['AveOccup','target'])
X.info()
```
![image](https://github.com/user-attachments/assets/75fb51aa-d69a-4c6c-abbe-520cb0f90649)
```
Y=df[['AveOccup','target']]
Y.info()
```
![image](https://github.com/user-attachments/assets/6d9bd824-655a-4ff2-bdc8-7e139066f3e8)
```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
X.head()
```
![image](https://github.com/user-attachments/assets/8efe2823-7328-459d-9667-ad5be304c9bb)

```
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
print(X_train)
```
![image](https://github.com/user-attachments/assets/85f84686-2d97-48da-a96d-aba969c91e4b)

```
# Initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

# Use MultiOutputRegressor to handle multiple output variables
multi_output_sgd = MultiOutputRegressor(sgd)

# Train the model
multi_output_sgd.fit(X_train, Y_train)

# Predict on the test data
Y_pred = multi_output_sgd.predict(X_test)

# Initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

# Use MultiOutputRegressor to handle multiple output variables
multi_output_sgd = MultiOutputRegressor(sgd)

# Train the model
multi_output_sgd.fit(X_train, Y_train)

# Predict on the test data
Y_pred = multi_output_sgd.predict(X_test)

# Inverse transform the predictions to get them back to the original scale
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

# Optionally, print some predictions
print("\nPredictions:\n", Y_pred[:5])
```
![image](https://github.com/user-attachments/assets/5aadc1d5-2539-4c5d-85d2-91d0caf2bb24)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
