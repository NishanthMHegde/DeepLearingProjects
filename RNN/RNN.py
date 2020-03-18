import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM 
from sklearn.metrics import mean_squared_error
import math 

"""
Recurrent Neural Networks are predictors and not classifiers.
Structure: They consists of one input layer, one Recurrent Neural layer and one output layer.
The recurrent layer references/contacts itself during different time intervals like t-3, t-2, t-1,
t, t+1, t+2, etc.
During backpropogation, there is a chance that the weights adjusted in one time step might not be suitable
for adjusting weights in another timestep when the recurrent layer references itself backward in time.
Lets assume Wrec be the wright to propogate backwards through time. There are 2 problems involved here.
1. Small Wrec, i.e, Wrec < 1 leads to Vanishing Gradient problem.
2. Big Wrec, i.e Wrec >1  leads to Exploding Gradient problem.

To overcome that we use LSTM (Long short-term memory) RNN with Wrec=1.
"""
"""
The aim of this project is to train our RNN model to predict the next day's Open value based
on previous day's Open value as read by our dataset.
To achieve this, we train our model using a dataset of Google Stock prices which we obtained from 2012
and use a timestep of 1 so as to train the model to get to know the difference between the Open value of
consecutive days.
We can use this model to predict next day's Open value for different year's Stock price dataset.
We then estimate the efficiency of our model by calculating the root mean squared error.

"""
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
test_set = pd.read_csv('Google_Stock_Price_Test.csv')

print(training_set.head())
print(test_set.head())

#Let us select the Open column but make it a matrix instead of a vector
training_set = training_set.iloc[:,1:2].values
real_test_set = test_set.iloc[:, 1:2].values 

#Let us scale the values using MinMaxScaler to get values in range [0,1].
#MinMaxScaler gives marginally better results than StandardScaler for this particular dataset,so lets use it.

sc = MinMaxScaler()
stock_training_set = sc.fit_transform(training_set)
stock_test_set = sc.transform(real_test_set)
print(stock_training_set.shape)

#X_train will contain open values of our training days
#y_train will contain open values of next days for training our model to predict the next day's OPEN value.
X_train = stock_training_set[0:1257] #Stock price at time t
y_train = stock_training_set[1:1258]#Stock price at time t+1

print(X_train)
print(y_train)

#Let us reshape the X_train data to include the timestep of 1 which we want to introduce
X_train = np.reshape(X_train, (1257, 1, 1)) #shape is of format (number of columns, timestep, number of features)

#Create the sequential model
#It is a regressor model as we are predicting values instead of classifying values as we did in ANN or CNN

regressor = Sequential()
#Let us add an LSTM layer which has 4 recurrent memory unit cells. 
#input_shape is (number of timesteps, number of features to predict)
#if number of timesteps = none, it can accept any number of timesteps.
regressor.add(LSTM(units=4, activation="sigmoid", input_shape = (None, 1))) 

#Let us now add a Dense layer which has one unit, which is stock price at time t.
regressor.add(Dense(units=1))

#Let us now compile our model.
regressor.compile(optimizer="adam", loss = "mean_squared_error")
#Let us now fit our model
regressor.fit(X_train, y_train, batch_size=32, epochs=200)

#Let us now test our regressor by using the real time data available in our test set
print("Let us now test our regressor by using the real time data available in our test set")
print(stock_test_set.shape)
stock_test_inputs = np.reshape(stock_test_set, (20, 1, 1))
predicted_stock_values = regressor.predict(stock_test_inputs)
print("The predicte stock prices are")
print(predicted_stock_values)

#But these values are MinMaxScaled. So we need to inverse transform it.
predicted_stock_values = sc.inverse_transform(predicted_stock_values)
print("The actual stock prices are")
print(sc.inverse_transform(stock_test_set))
print("The predicte stock prices are")
print(predicted_stock_values)

#Let us plot a graph of the real stock prices vs our predicted stock prices. Please keep in mind that there is a timestep of 1.
print("Let us plot a graph of the real stock prices vs our predicted stock prices. Please keep in mind that there is a timestep of 1.")
plt.plot(real_test_set, color="red", label="Real Stock Price")
plt.plot(predicted_stock_values, color="blue", label="Predicted Stock Price for t+1")
plt.title("Stock price prediction for timestep 1")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend()
plt.show()
#Evaluating the RNN model using RMSE.
print("Evaluating the RNN model using RMSE.")
rmse = math.sqrt(mean_squared_error(real_test_set, predicted_stock_values))
print("The RMSE is %s" % (rmse))

#Divide the rmse by total number of observations in the test set to obtain the real accuracy/loss
print("Divide the rmse by total number of observations in the test set to obtain the real accuracy/loss")
print(float(rmse/20))