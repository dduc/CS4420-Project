# This program is for power prediction based off
# of 22 days worth of data, at 5 second intervals,
# at each 5 seconds looking at temperature,
# humidity, etc. This approach uses a
# multivariate LSTM network


#imports for tensorflow,numpy,pandas using keras deep learning library
import numpy as np
from numpy import hstack

import pandas as pd
import datetime

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

from matplotlib import pyplot
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# split a multivariate sequence into samples
def split_all_data(allData, t_steps):
	X,y = list(),list()
	for i in range(len(allData)-150000):
		# find the end of this pattern
		end_ix = i + t_steps
		# check if we are beyond the dataset
		if end_ix > len(allData):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = allData[i:end_ix, :-1], allData[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# --------------------------- DATA PREP FOR LSTM NETWORK ---------------------------------- #

'''
# import CSV file data, leaving out last day (22) and prev day (21) until model is finished and accurate
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


# shape sequences to  vectors
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
# Data was formatted using EXCEL formula
'''

# read in newly formatted dataset
data = pd.read_csv('all_data.csv',header = 0,index_col = 0,low_memory=False)
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

time_steps = 1
X,y = split_all_data(all_data,time_steps)

# testing data prep results below...
print(X.shape,y.shape)
for i in range(3):
   print(X[i],y[i])

'''
# --------------------------------- END OF DATA PREP -------------------------------------- #

# --------------------------------- LSTM NETWORK ------------------------------------------ #
n_features = 5

print("Beginning to train LSTM network...")
model = Sequential()
model.add(LSTM(50,activation = 'relu',input_shape=(time_steps,n_features)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

model.fit(X,y,epochs=1,verbose=0)

print("Doing prediction...")
# prediction
x_input = np.array(X[X.length()-1])
x_input = x_input.reshape((1,time_steps,n_features))
yhat = model.predict(x_input,verbose = 0)
print(yhat)
'''
