'''
 Main Author: Dirk Duclos

 CSV Formatting: Sean Nickell

 Last Modified: 5-28-19

 Purpose: This program is for power consumption prediction
 based off of 22 days worth of data, at 5 second
 intervals,each interval holds temperature, humidity,
 solar, and wind data. This approach uses a multivariate
 LSTM network for prediction.
'''

#imports for tensorflow,numpy,and pandas using keras deep learning library
import numpy as np
from numpy import hstack

import pandas as pd
import datetime
import time
import h5py

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import load_model

from matplotlib import pyplot
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# split a multivariate sequence into samples
def split_all_data(allData, t_steps):
	X,y = list(),list()
	for i in range(len(allData)):
		end_ix = i + t_steps
		# see if at end of dataset
		if end_ix > 345560:
			break
		seq_x, seq_y = allData[i:end_ix, :-1], allData[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)

	return np.array(X), np.array(y)

# --------------------------- DATA PREP FOR LSTM NETWORK ---------------------------------- #
# This code below is commented out because data prep has already been done
'''
# import CSV file data, leaving out last day (22) and prev day (21) for finishing model accuracy based on real data
humidity_data = pd.read_csv("HumidityS.csv", skiprows = 0,usecols = ['Day 1','Day 2','Day 3','Day 4','Day 5','Day 6','Day 7','Day 8','Day 9','Day 10','Day 11','Day 12','Day 13','Day 14','Day 15','Day 16','Day 17','Day 18','Day 19','Day 20'])
solar_data = pd.read_csv("SolarS.csv",skiprows = 0,usecols = ['Day1','Day2','Day3','Day4','Day5','Day6','Day7','Day8','Day9','Day10','Day11','Day12','Day13','Day14','Day15','Day16','Day17','Day18','Day19','Day20'])
temp_data = pd.read_csv("TemperatureS.csv",skiprows = 0, usecols = ['Day1','Day2','Day3','Day4','Day5','Day6','Day7','Day8','Day9','Day10','Day11','Day12','Day13','Day14','Day15','Day16','Day17','Day18','Day19','Day20'])
wind_data = pd.read_csv("WindS.csv",skiprows = 0, usecols = ['Day 1','Day 2','Day 3','Day 4','Day 5','Day 6','Day 7','Day 8','Day 9','Day 10','Day 11','Day 12','Day 13','Day 14','Day 15','Day 16','Day 17','Day 18','Day 19','Day 20'])
power_data = pd.read_csv("powerS.csv",skiprows = 0, usecols = ['Day1','Day2','Day3','Day4','Day5','Day6','Day7','Day8','Day9','Day10','Day11','Day12','Day13','Day14','Day15','Day16','Day17','Day18','Day19','Day20'])

# for formatting new CSV
dummy_dates = ['2019-04-21','2019-04-22','2019-04-23','2019-04-24','2019-04-25','2019-04-26',
'2019-04-27','2019-04-28','2019-04-29','2019-04-30','2019-05-01',
'2019-05-02','2019-05-03','2019-05-04','2019-05-05','2019-05-06',
'2019-05-07','2019-05-08','2019-05-09','2019-05-10']

# flatten data into sequences for inputs into LSTM
humidity_seq = np.array(humidity_data)
humidity_seq = humidity_seq.flatten('F')

# set time of day for variable in learning
time_of_day_seconds = []

t = datetime.datetime(2019,4,21,0,0)
for i in range(20):
	for j in range(len(humidity_data)):
		tn = t + datetime.timedelta(0,5)
		t = t + datetime.timedelta(0,5)
		time_of_day_seconds.append(tn)
	j = 0

solar_seq = np.array(solar_data)
solar_seq = solar_seq.flatten('F')

temp_seq = np.array(temp_data)
temp_seq = temp_seq.flatten('F')

wind_seq = np.array(wind_data)
wind_seq = wind_seq.flatten('F')

power_seq = np.array(power_data)
power_seq = power_seq.flatten('F')

time_of_day_seconds_seq = np.array(time_of_day_seconds)

# shape sequences to vectors
humidity_seq = humidity_seq.reshape(len(humidity_seq),1)
temp_seq = temp_seq.reshape(len(temp_seq),1)
wind_seq = wind_seq.reshape(len(wind_seq),1)
solar_seq = solar_seq.reshape(len(solar_seq),1)
power_seq = power_seq.reshape(len(power_seq),1)
time_of_day_seconds_seq = time_of_day_seconds_seq.reshape(len(time_of_day_seconds_seq),1)
# combine all unit vectors in horizontal fashion
# format of data is in scientific notation,
# due to float values of Solar data

all_data = np.hstack((time_of_day_seconds_seq,humidity_seq,temp_seq,wind_seq,solar_seq,power_seq))

# format all_data for LSTM network (3 dimensions)
# Data is in final format using EXCEL formula
'''

# read in newly formatted dataset for prediction
print("Reading in Data...")
data = pd.read_csv('all_data.csv',skiprows=0, header=0, index_col=0, low_memory=False)
print("Data read complete.")
data = np.array(data)

time_steps = 60
print("Splitting Data for LSTM network input/output sequences...")
X,y = split_all_data(data,time_steps)
print("Splitting Done")

# commented out code below is for testing data formatting and visualizing data
'''
#for plotting the data purposes
values = data.values
groups = [0,1,2,3,4]
i = 1
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups),1,i)
	pyplot.plot(values[:,group])
	pyplot.title(data.columns[group],y=0.5,loc='right')
	i += 1
pyplot.show()

# testing data prep results below...
print(X.shape,y.shape)
#for i in range(len(X)):
print(X[len(X)-1],y[len(X)-1])
'''
# --------------------------------- END OF DATA PREP -------------------------------------- #

# --------------------------------- LSTM NETWORK ------------------------------------------ #
# LSTM has already been trained, but for more training, the code below does so
'''
n_features = 4

print("Beginning to train LSTM network...")
print("Hidden Layers = 50, Epochs = 6, Activation = ReLU, Features = 4, Time Steps = 60(5 minutes of 5 second intervals)")
start = time.time()
model = Sequential()
model.add(LSTM(50,activation = 'relu',input_shape=(time_steps,n_features)))
#model.add(Dense(1))
model.compile(optimizer='adam',loss='mse',metrics=['mean_squared_error'])
history = model.fit(X,y,validation_split=0.33,epochs=6)
model.save('power_prediction_model.h5')
print("Model Metrics")

print("Training Done")
end = time.time()
print("Total training time: ")
print(end-start)

'''
# number of features of LSTM network input
n_features = 4

# prediction
day_19_5_minutes = []
print("Gathering Day 19 data for Day 20 prediciton...")
# append manual input (humid,temp,wind,solar) at end of real data
X[len(X)-1][59] = [35,68,1.3,1.2]

for i in range(len(X)):
	if i == (5000):
		day_19_5_minutes.append(X[i])

# load already trained model
model = load_model("power_prediction_model.h5")
print("Day 19 last 5 minutes info gathered")
x_input = np.array(day_19_5_minutes)
print(x_input)
x_input = x_input.reshape((1,time_steps,n_features))
print("Doing prediction...")
yhat = model.predict(x_input,verbose = 0)
print("Predicted Power is: ",yhat)
