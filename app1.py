import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


start_date = '2000-01-01'
end_date = '2024-12-30'

st.title('stock trend prediction')

user_input = st.text_input('enter stock ticker','AAPL')

# Fetch data for Apple (AAPL)
df = yf.download(user_input, start=start_date, end=end_date)
df=df.reset_index()
df.columns=['date','ad close','close','high','low','open','volume']
df=df[['date','close','high','low','open','volume']]


# Display the first few rows
# df.head(10)

#describing data
st.subheader('data from 2000 to 2024')
st.write(df.head())

#visualizations
st.subheader('closing price vs time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.close)
st.pyplot(fig)

st.subheader('price with 100 days moving average')
ma100=df['close'].rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.close)
plt.plot(ma100,'r')
st.pyplot(fig)

st.subheader('price with 100 & 200 days moving average')
ma100=df['close'].rolling(100).mean()
ma200=df['close'].rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.close)
plt.plot(ma100,'g')
plt.plot(ma200,'r')
st.pyplot(fig)

#pre-processing
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

##splitting dataset into train and test split
training_size=int(len(scaled_data)*0.80)
test_size=len(scaled_data)-training_size
train_data,test_data=scaled_data[0:training_size,:],scaled_data[training_size:len(scaled_data),:1]

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

#load model
# model_lstm= load_model('price_forecast_lstm_model.h5')
from tensorflow.keras.metrics import MeanSquaredError

model_lstm = load_model('price_forecast_lstm_model.h5', custom_objects={'mse': MeanSquaredError()})

Train_pred = model_lstm.predict(X_train, verbose=0)
Val_pred = model_lstm.predict(X_test, verbose=0)

st.subheader('predictions on training data')
# Plot Training Data
fig=plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(len(y_train)), y_train, label='True Training Values', color='blue')
plt.plot(range(len(Train_pred)), Train_pred, label='Predicted Training Values', color='red', linestyle='--')
plt.title('Training Data vs. Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.legend()
st.pyplot(fig)


st.subheader('predictions on test data')
# Plot Validation Data
fig=plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 2)
plt.plot(range(len(y_test)), y_test, label='True Validation Values', color='blue')
plt.plot(range(len(Val_pred)), Val_pred, label='Predicted Validation Values', color='red', linestyle='--')
plt.title('Validation Data vs. Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.legend()
st.pyplot(fig)


# ploting forecast for next 30 days

x_input=test_data[int(len(test_data))-100:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output = []
n_steps = 100
i = 0
while (i < 30):

	if (len(temp_input) > 100):
		# print(temp_input)
		x_input = np.array(temp_input[1:])
		print("{} day input {}".format(i, x_input))
		x_input = x_input.reshape(1, -1)
		x_input = x_input.reshape((1, n_steps, 1))
		# print(x_input)
		yhat = model_lstm.predict(x_input, verbose=0)
		print("{} day output {}".format(i, yhat))
		temp_input.extend(yhat[0].tolist())
		temp_input = temp_input[1:]
		# print(temp_input)
		lst_output.extend(yhat.tolist())
		i = i + 1
	else:
		x_input = x_input.reshape((1, n_steps, 1))
		yhat = model_lstm.predict(x_input, verbose=0)
		print(yhat[0])
		temp_input.extend(yhat[0].tolist())
		print(len(temp_input))
		lst_output.extend(yhat.tolist())
		i = i + 1


day_new=np.arange(1,101)
day_pred=np.arange(101,131)

st.subheader('forecasting for next 30 days')
fig= plt.figure(figsize=(12,4))
plt.plot(day_new,scaler.inverse_transform(scaled_data[int(len(df))-100:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
st.pyplot(fig)

