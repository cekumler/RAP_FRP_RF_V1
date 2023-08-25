#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 10 2021

This is RF model that will take the HRRR and GOES inputs to build RF model
Compares this model to HRRR FRP Guassian Curve methods

If looking to compare just wildfires, implement the western Lon thresholds

@author: christina.bonfanti
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

import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt

###### Uncomment if you need to process the dataframe and save into csv - RAP_polar_goes_FRP.csv #####

li = []
## Read *npy's from 2018:
year = 2018
for m in range(1, 13):
    for d in range(1, 32):
        for h in range(24):
            try:
                date = datetime(year, m, d, h)
                next_24_hr_date = date + timedelta(hours=24)
                # Read the current date file
                if os.path.exists("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (date.strftime("%y%m%d%H"))):
                    frp_hr_file = np.load("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (date.strftime("%y%m%d%H")))
                    print('Processing File')
                    # If the GOES FRP is not zero, then store it and all other enteries into dataframe (or a large matrix depending on what makes sense?)
                    frp_0= frp_hr_file[0,:,:]
                    #same as above but polar picked
#                    frp_0= frp_hr_file[13,:,:]
                    #same as above but the averaged frp was  picked
#                    frp_0= frp_hr_file[14,:,:]
                    ff = frp_0 > 0
                    ff.astype(np.int)
                    # Non-zero FRP index values from files
                    nzi = np.nonzero(ff)
                    # Store into Dataframe
                    # GOES_avgd_FRP(0), lat(1), lon(2), temp2m(3), RH(4), Uwind(5), Vwind(6), wind_gust(7), precip(8), vis(9), 24_hour_past_FRP(10), 
                    # yesterday_frp(11), goes_rap_grid_count(12), polar_frp_avgd(13), FRP_avgd_all(14)
                    
                    # 9 columns need names: FRP(0), lat(1), lon(2), temp(3), RH(4), uwind(5), vwind(6), vegType(7), FRP_24hr(8), count(9)
                    frp_points=pd.DataFrame({'goes_FRP':frp_hr_file[0,(nzi[0]),(nzi[1])], 'lat':frp_hr_file[1,(nzi[0]),(nzi[1])], 
                                              'lon':frp_hr_file[2,(nzi[0]),(nzi[1])],'temp':frp_hr_file[3,(nzi[0]),(nzi[1])], 
                                              'RH':frp_hr_file[4,(nzi[0]),(nzi[1])], 'uwind':frp_hr_file[5,(nzi[0]),(nzi[1])],
                                              'time': date, 'hour': date.strftime("%H"),
                                              'vwind':frp_hr_file[6,(nzi[0]),(nzi[1])], 'wind_gust':frp_hr_file[7,(nzi[0]),(nzi[1])], 
                                              'precip':frp_hr_file[8,(nzi[0]),(nzi[1])], 'vis':frp_hr_file[9,(nzi[0]),(nzi[1])],
                                              'FRP_24hr':frp_hr_file[10,(nzi[0]),(nzi[1])], 'FRP_yesterday':frp_hr_file[11,(nzi[0]),(nzi[1])],
                                              'polar_FRP':frp_hr_file[13,(nzi[0]),(nzi[1])], 'FRP':frp_hr_file[14,(nzi[0]),(nzi[1])],
                                              'polar_FRP_24hr':frp_hr_file[15,(nzi[0]),(nzi[1])], 'polar_FRP_yesterday':frp_hr_file[16,(nzi[0]),(nzi[1])],
                                              'goes_FRP_24hr':frp_hr_file[17,(nzi[0]),(nzi[1])], 'goes_FRP_yesterday':frp_hr_file[18,(nzi[0]),(nzi[1])]})
                    # while count is less than 24, open future 24 hours of files and extract the
                    # same gridded information to store to data frame (10+(23*5)=125)
                    cc = 0
                    time_traveler = date
                    while cc < 24:
                        cc = cc + 1
                        time_traveler = time_traveler + timedelta(hours=1)
                        if os.path.exists("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (time_traveler.strftime("%y%m%d%H"))):
                            next_hr = np.load("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (time_traveler.strftime("%y%m%d%H")))
                            frp_points['hour_%s' %cc] = time_traveler.strftime("%H")
                            goes_frp_nxt = next_hr[0,(nzi[0]),(nzi[1])]
                            frp_points['goes_FRP_%s' % cc] = goes_frp_nxt
                            temp_nxt = next_hr[3,(nzi[0]),(nzi[1])]
                            frp_points['temp_%s' % cc] = temp_nxt
                            rh_nxt = next_hr[4,(nzi[0]),(nzi[1])]
                            frp_points['RH_%s' % cc] = rh_nxt
                            uwind_nxt = next_hr[5,(nzi[0]),(nzi[1])]
                            frp_points['uwind_%s' % cc] = uwind_nxt
                            vwind_nxt = next_hr[6,(nzi[0]),(nzi[1])]
                            frp_points['vwind_%s' % cc] = vwind_nxt
                            gust_nxt = next_hr[7, (nzi[0]),(nzi[1])]
                            frp_points['wind_gust_%s' % cc] = gust_nxt
                            precip_nxt = next_hr[8,(nzi[0]),(nzi[1])]
                            frp_points['precip_%s' % cc] = precip_nxt
                            vis_nxt = next_hr[9,(nzi[0]),(nzi[1])]
                            frp_points['vis_%s' % cc] = vis_nxt
#                            frp_24avg_nxt = next_hr[10,(nzi[0]),(nzi[1])]
#                            frp_points['FRP_24hr_%s' % cc] = frp_24avg_nxt
#                            frp_yester_nxt = next_hr[5,(nzi[0]),(nzi[1])]
#                            frp_points['FRP_yesterday_%s' % cc] = frp_yester_nxt
                            polar_nxt = next_hr[13,(nzi[0]),(nzi[1])]
                            frp_points['polar_FRP_%s' % cc] = polar_nxt
                            frp_nxt = next_hr[14, (nzi[0]),(nzi[1])] 
                            frp_points['FRP_%s' % cc] = frp_nxt
                    li.append(frp_points)
            except ValueError as e:
                    "date doesn't exist"                   
frp_all = pd.concat(li, axis=0, ignore_index=True)
size_frp_all = frp_all.shape
print('size of FRP database: ', size_frp_all)

frp_all.to_csv('./RAP_goes_FRP.csv', index = False)
