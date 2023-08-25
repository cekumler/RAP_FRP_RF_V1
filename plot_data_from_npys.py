#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:26:49 2021

PLot time series and data
idx[0] is the up down direction
idx[1] is the left right direction

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

import pydot
import matplotlib.pyplot as plt
import pickle

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

import pydot
import matplotlib.pyplot as plt
import seaborn as sns


rap_files = 'RAP_proc_files'

# specify time range
# specify input lat and lon
# loop - read files from time range
# select point(s) nearest to lat and lon
# append time and desired variables into dataframe

#specify time range

# specify location
# Camp Fire
# start_time = datetime(2018, 11, 7, 0)
# end_time = datetime(2018, 11, 15, 23)
# frp_lat = 39.810
# frp_lon = -121.437
# #MC
# frp_lat = 39.24
# frp_lon = -123.1
# start_time = datetime(2018, 7, 26, 0)
# end_time = datetime(2018, 8, 29, 23)
##Watson Creek
#frp_lat = 42.5
#frp_lon = -120.7
#start_time = datetime(2018, 8, 14, 0)
#end_time = datetime(2018, 8, 31, 23)
# Swan Lake Fire
start_time = datetime(2019, 6, 5, 0)
end_time = datetime(2019, 8, 15, 23)
frp_lat = 60.631
frp_lon = -150.438

# open RAP file to extract lats and lons
grbs = pygrib.open('./RAP/1820217000000')
for grb in grbs:
    print(grb)
    print(grb.name)
lats, lons = np.array(grb.latlons())
grid = np.shape(lats)

# calc distance between lats and lons to locate the rap grid w/ pt
abslon = np.abs(lons - frp_lon)
abslat = np.abs(lats - frp_lat)
c = np.maximum(abslon, abslat)
# idx has two numbers referencing the lat and lon point in np array for values to be stored
idx = np.argwhere(c==np.min(c))[0]

li = []
li_tot = []
li_e = []
li_s = []
li_se = []
li_pt = []

# desired region by index count where single-point can be 0  --> rrX2:
rr = 2
tt = start_time
while tt < end_time:
            if os.path.exists("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (tt.strftime("%y%m%d%H"))):
                file = np.load("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (tt.strftime("%y%m%d%H")))
#            if os.path.exists("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (tt.strftime("%y%m%d%H"))):
#                file = np.load("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (tt.strftime("%y%m%d%H")))
#                frp_points=pd.DataFrame({'FRP':[file[0,(idx[0]),(idx[1])]], 'date': [tt]})
                if rr > 0:
                    li = []
                    for i in range(rr):
                        for j in range (rr):
                            if i == 0 and j == 0:
                                frp_tt=pd.DataFrame({'FRP_goes':[file[0,(idx[0]+i),(idx[1]+j)]],'wind_gust':[file[7,(idx[0]+i),(idx[1]+j)]], 
                                                  'rh':[file[4,(idx[0]+i),(idx[1]+j)]], 'precip':[file[8,(idx[0]+i),(idx[1]+j)]], 
                                                  'FRP_both':[file[14,(idx[0]+i),(idx[1]+j)]], 'FRP_polar':[file[13,(idx[0]+i),(idx[1]+j)]],
                                                  'lat':[file[1,(idx[0]+i),(idx[1]+j)]], 'lon':[file[2,(idx[0]+i),(idx[1]+j)]],
                                                  'dist_from_pt_i': i,'dist_from_pt_j': j, 'temp':file[3,(idx[0]+i),(idx[1]+j)], 
                                                  '24hr_avg_FRP_polar':[file[15,(idx[0]+i),(idx[1]+j)]], 'yester_frp_polar':[file[16,(idx[0]+i),(idx[1]+j)]], 'date': [tt]})
                                li.append(frp_tt)
                                li_tot.append(frp_tt)
                            elif i==rr-1 and j==rr-1:
                                frp_pp=pd.DataFrame({'FRP_goes':[file[0,(idx[0]-i),(idx[1]-j)]],'wind_gust':[file[7,(idx[0]-i),(idx[1]-j)]], 
                                                      'rh':[file[4,(idx[0]-i),(idx[1]-j)]], 'precip':[file[8,(idx[0]-i),(idx[1]-j)]], 
                                                      'FRP_both':[file[14,(idx[0]-i),(idx[1]-j)]], 'FRP_polar':[file[13,(idx[0]-i),(idx[1]-j)]],
                                                      'lat':[file[1,(idx[0]-i),(idx[1]-j)]], 'lon':[file[2,(idx[0]-i),(idx[1]-j)]],
                                                      'dist_from_pt_i': -i,'dist_from_pt_j': -j, 'temp':file[3,(idx[0]-i),(idx[1]-j)], 
                                                      '24hr_avg_FRP_polar':[file[15,(idx[0]-i),(idx[1]-j)]], 'yester_frp_polar':[file[16,(idx[0]-i),(idx[1]-j)]], 'date': [tt]})
                                li.append(frp_pp)
                                li_tot.append(frp_pp)
                                frp_ttt=pd.DataFrame({'FRP_goes':[file[0,(idx[0]+i),(idx[1]+j)]],'wind_gust':[file[7,(idx[0]+i),(idx[1]+j)]], 
                                                      'rh':[file[4,(idx[0]+i),(idx[1]+j)]], 'precip':[file[8,(idx[0]+i),(idx[1]+j)]], 
                                                      'FRP_both':[file[14,(idx[0]+i),(idx[1]+j)]], 'FRP_polar':[file[13,(idx[0]+i),(idx[1]+j)]],
                                                      'lat':[file[1,(idx[0]+i),(idx[1]+j)]], 'lon':[file[2,(idx[0]+i),(idx[1]+j)]],
                                                      'dist_from_pt_i': i,'dist_from_pt_j': j, 'temp':file[3,(idx[0]+i),(idx[1]+j)], 
                                                      '24hr_avg_FRP_polar':[file[15,(idx[0]+i),(idx[1]+j)]], 'yester_frp_polar':[file[16,(idx[0]+i),(idx[1]+j)]], 'date': [tt]})                            
                                li.append(frp_ttt)
                                li_tot.append(frp_ttt)     
                                frp_tttt=pd.DataFrame({'FRP_goes':[file[0,(idx[0]-i),(idx[1]+j)]],'wind_gust':[file[7,(idx[0]-i),(idx[1]+j)]], 
                                                      'rh':[file[4,(idx[0]-i),(idx[1]+j)]], 'precip':[file[8,(idx[0]-i),(idx[1]+j)]], 
                                                      'FRP_both':[file[14,(idx[0]-i),(idx[1]+j)]], 'FRP_polar':[file[13,(idx[0]-i),(idx[1]+j)]],
                                                      'lat':[file[1,(idx[0]-i),(idx[1]+j)]], 'lon':[file[2,(idx[0]-i),(idx[1]+j)]],
                                                      'dist_from_pt_i': -i,'dist_from_pt_j': j, 'temp':file[3,(idx[0]-i),(idx[1]+j)], 
                                                      '24hr_avg_FRP_polar':[file[15,(idx[0]-i),(idx[1]+j)]], 'yester_frp_polar':[file[16,(idx[0]-i),(idx[1]+j)]], 'date': [tt]})
                                li.append(frp_tttt)
                                li_tot.append(frp_tttt)
                                frp_p=pd.DataFrame({'FRP_goes':[file[0,(idx[0]+i),(idx[1]-j)]],'wind_gust':[file[7,(idx[0]+i),(idx[1]-j)]], 
                                                      'rh':[file[4,(idx[0]+i),(idx[1]-j)]], 'precip':[file[8,(idx[0]+i),(idx[1]-j)]], 
                                                      'FRP_both':[file[14,(idx[0]+i),(idx[1]-j)]], 'FRP_polar':[file[13,(idx[0]+i),(idx[1]-j)]],
                                                      'lat':[file[1,(idx[0]+i),(idx[1]-j)]], 'lon':[file[2,(idx[0]+i),(idx[1]-j)]],
                                                      'dist_from_pt_i': i,'dist_from_pt_j': -j, 'temp':file[3,(idx[0]+i),(idx[1]-j)], 
                                                      '24hr_avg_FRP_polar':[file[15,(idx[0]+i),(idx[1]-j)]], 'yester_frp_polar':[file[16,(idx[0]+i),(idx[1]-j)]], 'date': [tt]})                            
                                li.append(frp_p)
                                li_tot.append(frp_p)  
                            else:
                                frp_q=pd.DataFrame({'FRP_goes':[file[0,(idx[0]-i),(idx[1]-j)]],'wind_gust':[file[7,(idx[0]-i),(idx[1]-j)]], 
                                                      'rh':[file[4,(idx[0]-i),(idx[1]-j)]], 'precip':[file[8,(idx[0]-i),(idx[1]-j)]], 
                                                      'FRP_both':[file[14,(idx[0]-i),(idx[1]-j)]], 'FRP_polar':[file[13,(idx[0]-i),(idx[1]-j)]],
                                                      'lat':[file[1,(idx[0]-i),(idx[1]-j)]], 'lon':[file[2,(idx[0]-i),(idx[1]-j)]],
                                                      'dist_from_pt_i': -i,'dist_from_pt_j': -j, 'temp':file[3,(idx[0]-i),(idx[1]-j)], 
                                                      '24hr_avg_FRP_polar':[file[15,(idx[0]-i),(idx[1]-j)]], 'yester_frp_polar':[file[16,(idx[0]-i),(idx[1]-j)]], 'date': [tt]})
                                li.append(frp_q)
                                li_tot.append(frp_q)
                                frp_qq=pd.DataFrame({'FRP_goes':[file[0,(idx[0]+i),(idx[1]+j)]],'wind_gust':[file[7,(idx[0]+i),(idx[1]+j)]], 
                                                      'rh':[file[4,(idx[0]+i),(idx[1]+j)]], 'precip':[file[8,(idx[0]+i),(idx[1]+j)]], 
                                                      'FRP_both':[file[14,(idx[0]+i),(idx[1]+j)]], 'FRP_polar':[file[13,(idx[0]+i),(idx[1]+j)]],
                                                      'lat':[file[1,(idx[0]+i),(idx[1]+j)]], 'lon':[file[2,(idx[0]+i),(idx[1]+j)]],
                                                      'dist_from_pt_i': i,'dist_from_pt_j': j, 'temp':file[3,(idx[0]+i),(idx[1]+j)], 
                                                      '24hr_avg_FRP_polar':[file[15,(idx[0]+i),(idx[1]+j)]], 'yester_frp_polar':[file[16,(idx[0]+i),(idx[1]+j)]], 'date': [tt]})                            
                                li.append(frp_qq)
                                li_tot.append(frp_qq)
                    # Sum the frp_points into a new dataframe entry
                    frp_t = pd.concat(li, axis=0, ignore_index=True)
                    frp_t = frp_t.replace(0,np.nan)
                    # change all the zeros in non-FRP columns to NaNs for a mean
                    frp_points = pd.DataFrame({'FRP_goes':[frp_t['FRP_goes'].sum()], 'wind_gust':[frp_t['wind_gust'].mean()], 
                                               'rh': [frp_t['rh'].mean()], 'precip':[frp_t['precip'].mean()],
                                               'lat':[file[1,(idx[0]),(idx[1])]], 'lon':[file[1,(idx[0]),(idx[1])]], 'temp':[frp_t['temp'].mean()], 
                                               'FRP_both':[frp_t['FRP_both'].sum()], 'FRP_polar':[frp_t['FRP_polar'].sum()],
                                               '24hr_avg_FRP_polar':[frp_t['24hr_avg_FRP_polar'].sum()], 'yester_frp_polar':[frp_t['yester_frp_polar'].sum()], 'date':[tt] })
                    #change all the NaNs back to zero
                    # append that dataframe and let frp_points reset (or rename frp_points)
                    li_pt.append(frp_points)
                else:
                    frp_points=pd.DataFrame({'FRP_goes':[file[0,(idx[0]),(idx[1])]],'wind_gust':[file[7,(idx[0]),(idx[1])]], 
                                          'rh':[file[4,(idx[0]),(idx[1])]], 'precip':[file[8,(idx[0]),(idx[1])]], 
                                          'temp':file[3,(idx[0]),(idx[1])], 
                                          'lat':[file[1,(idx[0]),(idx[1])]], 'lon':[file[2,(idx[0]),(idx[1])]],
                                          'FRP_both':[file[14,(idx[0]),(idx[1])]], 'FRP_polar':[file[13,(idx[0]),(idx[1])]], 
                                          '24hr_avg_FRP_polar':[file[15,(idx[0]),(idx[1])]], 'yester_frp_polar':[file[16,(idx[0]),(idx[1])]], 'date': [tt]})
                    li_pt.append(frp_points)
                    frp_points_e=pd.DataFrame({'FRP_goes':[file[0,(idx[0]+1),(idx[1])]],'wind_gust':[file[7,(idx[0]+1),(idx[1])]], 
                                          'rh':[file[4,(idx[0]+1),(idx[1])]], 'precip':[file[8,(idx[0]+1),(idx[1])]], 
                                          'FRP_both':[file[14,(idx[0]+1),(idx[1])]], 'FRP_polar':[file[13,(idx[0]+1),(idx[1])]], 
                                          '24hr_avg_FRP_polar':[file[15,(idx[0]+1),(idx[1])]], 'yester_frp_polar':[file[16,(idx[0]+1),(idx[1])]], 'date': [tt]})
                    li_e.append(frp_points_e)
                    frp_points_s=pd.DataFrame({'FRP_goes':[file[0,(idx[0]),(idx[1]+1)]],'wind_gust':[file[7,(idx[0]),(idx[1]+1)]], 
                                          'rh':[file[4,(idx[0]),(idx[1]+1)]], 'precip':[file[8,(idx[0]),(idx[1]+1)]], 
                                          'FRP_both':[file[14,(idx[0]),(idx[1]+1)]], 'FRP_polar':[file[13,(idx[0]),(idx[1]+1)]], 
                                          '24hr_avg_FRP_polar':[file[15,(idx[0]),(idx[1]+1)]], 'yester_frp_polar':[file[16,(idx[0]),(idx[1]+1)]], 'date': [tt]})
                    li_s.append(frp_points_s)
                    frp_points_se=pd.DataFrame({'FRP_goes':[file[0,(idx[0]+1),(idx[1]+1)]],'wind_gust':[file[7,(idx[0]+1),(idx[1]+1)]], 
                                          'rh':[file[4,(idx[0]+1),(idx[1]+1)]], 'precip':[file[8,(idx[0]+1),(idx[1]+1)]], 
                                          'FRP_both':[file[14,(idx[0]+1),(idx[1]+1)]], 'FRP_polar':[file[13,(idx[0]+1),(idx[1]+1)]], 
                                          '24hr_avg_FRP_polar':[file[15,(idx[0]+1),(idx[1]+1)]], 'yester_frp_polar':[file[16,(idx[0]+1),(idx[1]+1)]], 'date': [tt]})
                    li_se.append(frp_points_se)
                    

            tt = tt + timedelta(hours=1)
            print('time: ', tt)

            
frp_avgd = pd.concat(li_pt, axis=0, ignore_index=True)
frp_avgd = frp_avgd.replace(0,np.nan) # make any zero a NaN
frp_all = pd.concat(li_tot, axis=0, ignore_index=True)
frp_all = frp_all.replace(0,np.nan) # make any zero a NaN
frp_all = frp_avgd

def fahr_to_celsius(temp_fahr):
    temp_celsius = (temp_fahr - 273.15)
    return temp_celsius

frp_avgd["temp_C"] = fahr_to_celsius(frp_avgd["temp"])
frp_all["temp_C"] = fahr_to_celsius(frp_all["temp"])


# export case study dataframes
#frp_avgd.to_csv('./campfire_results/frp_camp_9pixels_mean.csv')
#frp_all.to_csv('./campfire_results/frp_camp_9pixels_all.csv')

# Fil the averaged with NaN values at missing hours for a complete dataset
# export case study dataframes

#frp_all.to_csv('./campfire_results/frp_camp_9pixels_complete.csv')



#####

#Reverse the NaNs so plotting doesn't freak'
frp_avgd = frp_avgd.replace(np.nan,0) # make any zero a NaN
frp_all = frp_all.replace(np.nan,0) # make any zero a NaN

# extract indicies from a certain date(s):
# plt_start = datetime(2018, 11, 7, 0)
# plt_end = datetime(2018, 11, 15, 23) 
# plt_start = datetime(2018, 7, 26, 0)
# plt_end = datetime(2018, 8, 29, 23)
plt_start = datetime(2018, 8, 14, 0)
plt_end = datetime(2018, 8, 31, 23) 
srt = 0
stp = 0
fd = 0
for i in range(np.shape(frp_all)[0]):
    if (frp_all.date[i].day >= plt_start.day) and (frp_all.date[i].day < plt_end.day):
        if fd == 0:
            fd = 1
            srt = i
        else:
            stp = i
    else:
        # out of range
        print ('out of date range')
                
    
    # find first time date is hit
    # store idx until last time end date is hit
    # output the two index values from the database
 
# Make a dataframe with just FRP points
plt_case = frp_avgd[srt:stp]
plt_case_both = plt_case.loc[~((plt_case['FRP_both'] == 0))]
plt_case_goes = plt_case.loc[~((plt_case['FRP_goes'] == 0))]
plt_case_polar = plt_case.loc[~((plt_case['FRP_polar'] == 0))]
# plt_case_e = frp_all_e[srt:stp]
# plt_case_both_e = plt_case.loc[~((plt_case_e['FRP_both'] == 0))]
# plt_case_goes_e = plt_case.loc[~((plt_case_e['FRP_goes'] == 0))]
# plt_case_polar_e = plt_case.loc[~((plt_case_e['FRP_polar'] == 0))]
# plt_case_se = frp_all_e[srt:stp]
# plt_case_both_se = plt_case.loc[~((plt_case_se['FRP_both'] == 0))]
# plt_case_goes_se = plt_case.loc[~((plt_case_se['FRP_goes'] == 0))]
# plt_case_polar_se = plt_case.loc[~((plt_case_se['FRP_polar'] == 0))]
# plt_case_s = frp_all_e[srt:stp]
# plt_case_both_s = plt_case.loc[~((plt_case_s['FRP_both'] == 0))]
# plt_case_goes_s = plt_case.loc[~((plt_case_s['FRP_goes'] == 0))]
# plt_case_polar_s = plt_case.loc[~((plt_case_s['FRP_polar'] == 0))]


# fig1 = plt.gcf()
# x = frp_all.lon[srt:stp]
# y = frp_all.lat[srt:stp]
# colors = frp_all.FRP_both[srt:stp]
# sc = plt.scatter(x, y, c=colors, alpha=0.75, vmin=frp_all.FRP_both[srt:stp].min(), vmax=max(frp_all.FRP_both[srt:stp]))
# plt.xlim([min(frp_all.lon[srt:stp])-0.75, min(frp_all.lon[srt:stp])+0.75])
# plt.ylim([max(frp_all.lat[srt:stp])-0.75, max(frp_all.lat[srt:stp])+0.75])
# plt.colorbar(sc)
# plt.title('FRP Values over the CA Aug 7 2018')
# plt.tight_layout()
# plt.show()
# #fig1.savefig('FRP_RAP_avgd_CONUS_2018.png', dpi=200)
# plt.clf()
# plt.cla()
# plt.close()

# # # make a subset
#plt_case = frp_all[srt:stp]
plt.title('FRP both June 6 - Aug 15 2019 Swan Lake Fire')
#plt.scatter(plt_case_both['date'], plt_case_both['FRP_both'], s=10)
plt.plot(plt_case_both['date'], plt_case_both['FRP_both'], linewidth=1)
plt.xlim([frp_all.date[srt], frp_all.date[stp]])
plt.locator_params(nbins=10)
plt.xticks(rotation = 45)
plt.ylabel('FRP (MW)')
plt.grid(b=True, which='major', color='black', linestyle='-')
#plt.savefig('./swan_lake_fire_results/swan_lake_2019_frpBOTH.png' , dpi=200, bbox_inches='tight')
plt.clf()
plt.cla()

#plt_case = frp_all[srt:stp]
plt.title('FRP polar June 6 - Aug 15 2019 Swan Lake Fire')
#plt.scatter(plt_case_polar['date'], plt_case_polar['FRP_polar'], s=10)
plt.plot(plt_case_both['date'], plt_case_both['FRP_polar'], linewidth=1)
plt.xlim([frp_all.date[srt], frp_all.date[stp]])
plt.xticks(rotation = 45)
plt.ylabel('FRP (MW)')
plt.grid(b=True, which='major', color='black', linestyle='-')
#plt.savefig('./swan_lake_fire_results/swan_lake_2019_frppolar.png' , dpi=200, bbox_inches='tight')
plt.clf()
plt.cla()

#plt_case = frp_all[srt:stp]
plt.title('FRP GOES June 6 - Aug 15 2019 Swan Lake Fire')
#plt.scatter(plt_case_goes['date'], plt_case_goes['FRP_goes'], s=10)
plt.plot(plt_case_both['date'], plt_case_both['FRP_goes'], linewidth=1)
plt.xlim([frp_all.date[srt], frp_all.date[stp]])
plt.xticks(rotation = 45)
plt.ylabel('FRP (MW)')
plt.grid(b=True, which='major', color='black', linestyle='-')
#plt.savefig('./swan_lake_fire_results/swan_lake_2019_frpgoes.png' , dpi=200, bbox_inches='tight')
plt.clf()
plt.cla()

# # # # make a subset
#plt_case = frp_all[srt:stp]
plt.title('Wind Gust (m/s) June 6 - Aug 15 2019 Swan Lake Fire')
#plt.scatter(plt_case_goes['date'], plt_case_goes['wind_gust'], s=10)
plt.plot(plt_case_both['date'], plt_case_both['wind_gust'], linewidth=1)
plt.xlim([frp_all.date[srt], frp_all.date[stp]])
plt.xticks(rotation = 45)
plt.ylabel('Wind Gust (m/s)')
plt.grid(b=True, which='major', color='black', linestyle='-')
#plt.savefig('./swan_lake_fire_results/swan_lake_2019_windgust.png' , dpi=200, bbox_inches='tight')
plt.clf()
plt.cla()

# # # # make a subset
#plt_case = frp_all[srt:stp]
plt.title('RH (%) June 6 - Aug 15 2019 Swan Lake Fire')
#plt.scatter(plt_case_polar['date'], plt_case_polar['rh'], s=10)
plt.plot(plt_case_both['date'], plt_case_both['rh'], linewidth=1)
plt.xlim([frp_all.date[srt], frp_all.date[stp]])
plt.xticks(rotation = 45)
plt.ylabel('Relative Humidity (%)')
plt.grid(b=True, which='major', color='black', linestyle='-')
#plt.savefig('./swan_lake_fire_results/swan_lake_2019_rh.png' , dpi=200, bbox_inches='tight')
plt.clf()
plt.cla()

# # # make a subset
#plt_case = frp_all[srt:stp]
plt.title('Pecip June 6 - Aug 15 2019 Swan Lake Fire')
#plt.scatter(plt_case_polar['date'], plt_case_polar['precip'], s=10)
plt.plot(plt_case_both['date'], plt_case_both['precip'], linewidth=1)
plt.xlim([frp_all.date[srt], frp_all.date[stp]])
plt.xticks(rotation = 45)
plt.ylabel('Instant Precipitation (kg/(s*m^2))')
plt.grid(b=True, which='major', color='black', linestyle='-')
#plt.savefig('./swan_lake_fire_results/swan_lake_2019_precip.png' , dpi=200, bbox_inches='tight')
plt.clf()
plt.cla()

# # # make a subset
#plt_case = frp_all[srt:stp]
plt.title('Temp June 6 - Aug 15 2019 Swan Lake Fire')
#plt.scatter(plt_case_polar['date'], plt_case_polar['temp_C'], s=10)
plt.plot(plt_case_both['date'], plt_case_both['temp_C'], linewidth=1)
plt.xlim([frp_all.date[srt], frp_all.date[stp]])
plt.xticks(rotation = 45)
plt.ylabel('Temperature (C)')
plt.grid(b=True, which='major', color='black', linestyle='-')
#plt.savefig('./swan_lake_fire_results/swan_lake_2019_tempC.png', dpi=200, bbox_inches='tight')
plt.clf()
plt.cla()

# plot two on same plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(plt_case_polar['date'], plt_case_polar['FRP_polar'], linewidth=1.5)
ax1.set_ylabel('FRP (MW)')

ax2 = ax1.twinx()
ax2.plot(plt_case_polar['date'], plt_case_polar['wind_gust'], 'r-', linewidth=1.5)
ax2.set_ylabel('Wind Gust (m/s)', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.title('FPR Polar and Wind Gust June 6 - Aug 15 2019 Swan Lake Fire')
fig.autofmt_xdate()
#plt.savefig('./swan_lake_fire_results/swan_lake_2019_frppolar_windgust.png', dpi=200, bbox_inches='tight')
plt.clf()
plt.cla()

# plot two on same plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(plt_case_polar['date'], plt_case_polar['FRP_polar'], linewidth=1.5)
ax1.set_ylabel('FRP (MW)')

ax2 = ax1.twinx()
ax2.plot(plt_case_polar['date'], plt_case_polar['yester_frp_polar'], 'r-', linewidth=1.5)
ax2.set_ylabel('FRP (MW)', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.title('FPR Polar and Yesterday Mean FRP June 6 - Aug 15 2019 Swan Lake Fire')
fig.autofmt_xdate()
plt.savefig('./swan_lake_fire_results/swan_lake_2019_frppolar_yesterdayFRP.png', dpi=200, bbox_inches='tight')

# fig1 = plt.gcf()
# x = plt_case_polar['date']
# y = plt_case_polar['FRP']
# colors = plt_case_polar['FRP']
# sc = plt.scatter(x, y, c=colors, alpha=0.75, vmin=frp_all['FRP'].min(), vmax=frp_all['FRP'].max())
# plt.colorbar(sc)
# plt.title('FRP Values in CA 39.81N 121.44W')
# plt.tight_layout()
# plt.show()
# #fig1.savefig('FRP_RAP_avgd_CONUS_2018.png', dpi=300)
# # plt.clf()
# # plt.cla()

