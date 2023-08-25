#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 1 2021

# RF_v1_demo.py
# Program calls a trained RAP-gridded RF model, then opens the RF model to run on
# specified inputs, and finally produces a dataframe of model-generated outputs.

@author: christina.e.kumler
"""
import numpy as np
from numpy import savetxt
from numpy import save
from numpy import asarray
import pandas as pd
import netCDF4 as nc
import pygrib
from pyproj import Proj

import time

import csv

import matplotlib.pyplot as plt
import math
import dateutil.parser
import glob
from datetime import datetime
from random import randint, random
import time
import seaborn as sns
import os
import glob
import statistics
from datetime import datetime, timedelta

import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

from datetime import datetime, timedelta

import pydot
import pickle

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.tree import export_graphviz
import pydot


## TThe following are the necessary RAP variables
#grbs = pygrib.open('./RAP_FI')
#for grb in grbs:
#    print(grb)
#    print(grb.name)
#lats, lons = np.array(grb.latlons())
#Wind speed (gust):m s**-1 (instant):lambert:surface:level 0:fcst time 0 hrs:from 201807211700
#2 metre temperature:K (instant):lambert:heightAboveGround:level 2 m:fcst time 0 hrs:from 201807211700
#2 metre relative humidity:% (instant):lambert:heightAboveGround:level 2 m:fcst time 0 hrs:from 201807211700

## ---- prepare the data
# NOTE: So long as the FRP values are compariable those of the desired satellite in magnitude
# and are plotted to the RAP grid, then any input format can be used. Combine
# the FRP value(s) with corresponding RAP variables for the specified model and
# corresponding mFRP values to run the model.

## The following example was generated from all FRP satellite points that were non-zero GOES values
# This is the 3x3 RAP pixel box mean of a fire location
# This was created from plot_data_from_npys.py
# The paticular fire for this demo is the Williams Flat Fire
frp_all = pd.read_csv('./williams_2019_9pixels_mean.csv') 

## This section might not be necessary for your data input but is where you specify
# the date and time. The hour will be an input into the model, so if your data
# doesn't have an hour column, then you might consider constructing one.
frp_all['date'] = pd.to_datetime(frp_all['date'], format='%Y-%m-%d %H:%M:%S')
# Add the hour, lat, and lon columns
frp_all['hour'] = frp_all.date.dt.hour


## Conditional to drop rows where FRP_yesterday == 0 and
## any instances that goes start hour FRP is zero
frp_all = frp_all.loc[~((frp_all['yester_frp_goes'] <= 0))]

## Section that specifies the location and time of the fire in the case study
# This step can be skipped if you're not seeking to plot a case study because it is
# not a required input into the model.
#Williams Flat Fire
start_time = datetime(2019, 8, 1, 0)
end_time = datetime(2019, 8, 9, 23)
frp_lat = 47.98
frp_lon = -118.624

# drop rows containing NaNs
frp_all.dropna(axis=0, subset=['lat', 'lon', 'hour', 'temp', 'wind_gust', 'rh', 'yester_frp_goes'], inplace=True)

## This step caluclates the equivalent HRRR predicted 
# hourly FRP value based on location
# This csv stores the amplitudes for the HRRR Guassian Curve in mountain and
# pacific time. It also has the next_hour amplitutde. 
frp_climo = pd.read_csv("FRP_climo_HRRR_amps.csv")
# create the climo column in frp_all
frp_all['HRRR_FRP'] = ""
# read in the timestamp where i=index or column #
for index in frp_all.index:
#    print('index: ', index)
    if frp_all.loc[index,'lon'] <= -115:
        frp_all.loc[index, 'HRRR_FRP'] = frp_all.loc[index, 'yester_frp_goes']*frp_climo['summer_PCT'][(frp_all.loc[index, 'hour'])]
    else:
        frp_all.loc[index, 'HRRR_FRP'] = frp_all.loc[index, 'yester_frp_goes']*frp_climo['summer_MST'][(frp_all.loc[index, 'hour'])]

hrrr_frps = frp_all['HRRR_FRP']


## ------- Using the machine learning model in the inference stage
# Here is where you drop any rows that might contain NaNs
frp_all.dropna(axis=0, subset=['lat', 'lon', 'hour', 'temp', 'wind_gust', 'rh', 'yester_frp_goes'], inplace=True)

# These are the input FEATURES for the machine learning algorithm
features = frp_all[['lat', 'lon', 'hour', 'temp', 'wind_gust', 'rh', 'yester_frp_goes']]
validate_features = features

# This is the name of the machine learning model that is being used in the inference
filename = './RF_v1_models/RF_conus_goes_2018_v1.sav'

# This is how to load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Run the model (inference stage) and save the outputs into "predictors"
# Then save predictors into a corresponding RF column to your inputs
predictors = loaded_model.predict(validate_features)
frp_all['RF'] = predictors

## ---- Analysis of the machine learning model
# export table if so desired
# frp_all.to_csv('.williams_2019_actual_and_RF_predicted.csv')

# calculate differences between GOES and HRRR and ML
ML_goes_diff = predictors - frp_all['FRP_goes']
HRRR_goes_diff = hrrr_frps - frp_all['FRP_goes']
mfrp_diff = frp_all['yester_frp_goes'] - frp_all['FRP_goes'] 

#calculate the MAE
errors = abs(predictors - frp_all['FRP_goes'])
# Print out the mean absolute error (mae)
print('Mean Absolute Error for FRP:', round(np.mean(errors), 2))
errors_climo = abs(hrrr_frps - frp_all['FRP_goes'])
# Print out the mean absolute error (mae)
print('Mean Absolute Error for FRP HRRR Gaussian Curve:', round(np.mean(errors_climo), 2))

#create some figures
fig3 = plt.figure(figsize=(12, 7))
plt.title('ML and HRRR FRP Differences Williams Flat Fire Aug 1-9 2019')
plt.plot(frp_all['date'], ML_goes_diff, linewidth=0.75, color='b', linestyle=':', marker='*', label='ML_goes_diff')
plt.plot(frp_all['date'], HRRR_goes_diff, linewidth=0.75, color='g', linestyle=':', marker='x', label='HRRR_goes_diff')
plt.xlim([frp_all['date'].min()-timedelta(days=1), frp_all['date'].max()])
plt.xticks(rotation = 45)
plt.legend(loc="upper left")
plt.grid(b=True, which='major', color='black', linestyle='-')

#create a figure
fig1 = plt.figure(figsize=(12, 7))
plt.title('ML, HRRR, and GOES FRP Williams Flat Fire Aug 1-9 2019')
plt.plot(frp_all['date'], predictors, linewidth=0.75, color='b', linestyle=':', marker='*', label='ML')
plt.plot(frp_all['date'], hrrr_frps, linewidth=0.75, color='g', linestyle=':', marker='x', label='HRRR')
plt.plot(frp_all['date'], frp_all['FRP_goes'], linewidth=0.75, color='r', linestyle='-', marker='.', alpha=0.4, label='GOES FRP')
plt.plot(frp_all['date'], frp_all['yester_frp_goes'], linewidth=2, color='m', linestyle='--', alpha=0.5, label='Yesterday mFRP')
plt.xlim([frp_all['date'].min()-timedelta(days=1), frp_all['date'].max()])
plt.xticks(rotation = 45)
plt.legend(loc="upper left")
plt.grid(b=True, which='major', color='black', linestyle='-')