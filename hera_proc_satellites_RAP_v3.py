#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feb 3 2021 --> working version

Read all satellite processed FRP files, GOES and polars, to create hourly RAP_gridded progrocessed files
Creates a previous 24 hour FRP avg variable
Creats a yesterday 24 hour FRP average --> this is done by utc day

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

# will need to add this in loop when using 2019 and 2020, but for now this can be here
year = 2018
# read the RAP file to extract lat and lon fields
grbs = pygrib.open('/scratch1/RDARCH/rda-aidata/rap/grib2/20180509/1812910000000')
# hera RAP files
# rap_directory = '/scratch1/RDARCH/rda-aidata/rap/grib2/'

# read the processed file
#data_dir = ('/Users/christina.bonfanti/Documents/machine_learning/FRP/2018_new_processed*.csv')
# read the satellite file
#ds = nc.Dataset('./goesABI_2018/fdcc/001/OR_ABI-L2-FDCC-M3_G16_s20180010002196_e20180010004569_c20180010005147.nc')

# Sets up files for RAP empty grid, lat, and lon arrays
for grb in grbs:
    print(grb)
    print(grb.name)

lats, lons = np.array(grb.latlons())
grid = np.shape(lats)

# create empty grid array size variables, lat, lon --> variables = count, lat, lon, FRP, 
# temp, uwind, vwind, humimdity, vegType == 9
mergedf = np.zeros((15, grid[0], grid[1]))
#export grid to be reference in future
#np.savetxt("rap_empty_grid.csv", mergedf, delimiter=",")
#np.savetxt("rap_lats_grid.csv", lats, delimiter=",")
#np.savetxt("rap_lons_grid.csv", lons, delimiter=",")


def create_proc(date, frp_regrid, prev_24_hr_date, yesterday_date):
    mergedf = np.zeros((15, grid[0], grid[1]))
 
    # open the RAP lats and lons -- make into mesh grid
    # open the empty grid and fill with FRP and weather information
    frp_list = []
    
    #Following loop runs through all enteries composited from the hour.
    print("Entering Storage loop. FRP size is: ", range(len(frp_regrid)))
    for i in range(len(frp_regrid)):
        curr_lon = frp_regrid.lon[i]
        curr_lat = frp_regrid.lat[i]
#        print ("current lat: ", curr_lat, " and current lon: ", curr_lon)

        abslon = np.abs(lons - curr_lon)
        abslat = np.abs(lats - curr_lat)
        c = np.maximum(abslon, abslat)
        
        # idx has two numbers referencing the lat and lon point in np array for values to be stored
        idx = np.argwhere(c==np.min(c))[0]
#        dist = ((np.radians(lons[idx[0], idx[1]])-np.radians(curr_lon))*np.cos((np.radians(lats[idx[0], idx[1]]) + np.radians(curr_lat))/2))**2 + (np.radians(lats[idx[0], idx[1]]) - np.radians(curr_lat))**2

        # store vars: lat	lon	radar	vis	windspeed	temp2m	RH	Uwind	Vwind	precip	FRP	bright_t13	confidence	mask	id	sat	dateTime
        # note that windspeed is GUST SPEED for RAP files        
        
        # check if current FRP point, if so then make FRP a sum and update count, else store FRP
        # FRP sum will end up being one var, but there's a temp var that sums up the polars to be
        # injected into FRP since the polar's are less common occurances but need summations, not averages

#        mergedf[(0, idx[0], idx[1])] =  # GOES FRP        
        mergedf[(1, idx[0], idx[1])] = frp_regrid.lat[i]
        mergedf[(2, idx[0], idx[1])] = frp_regrid.lon[i]
        mergedf[(3, idx[0], idx[1])] = frp_regrid.temp2m[i]
        mergedf[(4, idx[0], idx[1])] = frp_regrid.RH[i]
        mergedf[(5, idx[0], idx[1])] = frp_regrid.Uwind[i]
        mergedf[(6, idx[0], idx[1])] = frp_regrid.Vwind[i]
        mergedf[(7, idx[0], idx[1])] = frp_regrid.windspeed[i] #wind gust
        mergedf[(8, idx[0], idx[1])] = frp_regrid.precip[i] # instant
        mergedf[(9, idx[0], idx[1])] = frp_regrid.vis[i] # instant        
        mergedf[(10, idx[0], idx[1])] = frp_regrid.FRP[i] #this will be the 24 hour average to current
#        mergedf[(11, idx[0], idx[1])] =  #this will be yesterdays 24 hour average
#        mergedf[(12, idx[0], idx[1])] =  #count for mergedf[0] current hour goes avg value
#        mergedf[(13, idx[0], idx[0])] = #POLAR FRP -- summation of polar FRP points
#        mergedf[(14, idx[0], idx[0])] = #BOTH FRP -- avgd polar and goes FRP points
        
        # check if there exists an FRP in the RAP grid cell yet
        if mergedf[(12, idx[0], idx[1])] > 0:
            # is it a geo or polar satellite?
            if 'goes' in frp_regrid.sat[i]:
                mergedf[(0, idx[0], idx[1])] = mergedf[(0, idx[0], idx[1])] + frp_regrid.FRP[i] #sum for averaging goes
#            print("check non-zero FRP value: ", frp_regrid.FRP[i])
#            print ("FRP RAP grid total summation: ", mergedf[(0, idx[0], idx[1])])
                mergedf[(12, idx[0], idx[1])] = mergedf[(12, idx[0], idx[1])] + 1 #update count for averaging goes
#            print("checking count value: ", mergedf[(9, idx[0], idx[1])])
#            print("check the count", mergedf[(9, idx[0], idx[1])])
                frp_list.append(idx)    # this guy hangs on to the spots with goes frps
            # it's a polar satellite    
            else:
                mergedf[(13, idx[0], idx[1])] = mergedf[(13, idx[0], idx[1])] + frp_regrid.FRP[i] #sum polar FRPs
        # Otherwise, there is no FRP in the desired RAP grid for that hour yet
        else:
            # is it from goes?
            if 'goes' in frp_regrid.sat[i]:
                mergedf[(0, idx[0], idx[1])] = frp_regrid.FRP[i]
                mergedf[(12, idx[0], idx[1])] = mergedf[(12, idx[0], idx[1])] + 1
                frp_list.append(idx) # this guy hangs on to the spots with goes frps
            # else, it's a polar point
            else:
                mergedf[(13, idx[0], idx[1])] = mergedf[(13, idx[0], idx[1])] + frp_regrid.FRP[i] #sum polar FRPs
        print('frp checkpoint: ', mergedf[(0, idx[0], idx[1])]) 
            
    # remove duplicate goes enteries in the list
    new_frp_list = set(map(tuple, frp_list))
    
    # loop through goes list and calculate an average FRP
    for i in new_frp_list:
        frp_avg = mergedf[(0, i[0], i[1])] / mergedf[(12, i[0], i[1])]
        mergedf[(0, i[0], i[1])] = frp_avg
        print('frp averaged to grid checkpoint: ', mergedf[(0, idx[0], idx[1])])
    
    # Look if there is both a polar and GOES FRP. If so, take average of both and replace
    polar_goes_values = polar_goes_combo_FRP(mergedf)
    mergedf[14, :, :] = polar_goes_values
    print('max of goes_polar_frp :', np.max(mergedf[14, :, :]))
        
    # Look in previous files for up to 24hours of files in past
    prev_FRP_values = prev_FRP(mergedf, date, prev_24_hr_date)
    mergedf[10, :, :] = prev_FRP_values
    print('max of prev_24hour :', np.max(mergedf[10, :, :]))
    
    # Look in previous files for the previous day's average FRP
    prev_day_values = prev_day_frp(mergedf, date, yesterday_date)
    mergedf[11, :, :] = prev_day_values
    print('max of prev_day_frp :', np.max(mergedf[11, :, :]))

    # takes care of the outside bounds FRP points that appended to RAP files anyways
    mergedf[:,0,:] = 0
    mergedf[:,-1,:] = 0
    mergedf[:,:,0] = 0
    mergedf[:,:,-1] = 0
        
    # export the file into numpy array AND CSV
#    numpy.savetxt('%s_RAP_FRP_proc.csv' % (date.strftime("%y%m%d%H")), mergedf, delimiter=",")
    save("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (date.strftime("%y%m%d%H")), frp_regrid)
    print('saved npy file')
    # reshape 3D 10x337x452 to 2D 10x152324
#    flat_grid = mergedf.reshape(10,151987)
#    savetxt("./RAP_proc_files/%s_RAP_FRP_proc.csv" % (date.strftime("%y%m%d%H")), flat_grid, delimiter=',')
    # check file 
    # with open("18061103_RAP_FRP_proc.csv", 'r') as f:
    #     prev_FRP_file = list(csv.reader(f, delimiter=","))
    # prev_FRP_file = np.array(prev_FRP_file[0:], dtype=np.float)

    return mergedf

# define polar FRP gridcell summation -- INCORRECT
def polar_goes_combo_FRP(mergedf):
    
# new way
#    # copy the polar and goes grids into arrays
#    p_frp = mergedf[13,:,:]
#    g_frp = mergedf[0,:,:]
#    # set the zeros to nan
#    p_frp[p_frp == 0] = 'nan'
#    g_frp[g_frp == 0] = 'nan'    
#    # take a numpy nan mean  
#    m = np.array([p_frp, p_frp])
#    np.nanmean(m, axis=0)
#    m[m == 'nan'] = 0
#    polar_goes_values = m
	# if there's both goes and frp, add and sum, otherwise take the non-zero item
    # nested loops not ideal but just going to loop it at this point in the project
    polar_goes_values = np.zeros((grid[0], grid[1]))
    for ii in range(grid[0]):
        for jj in range(grid[1]):
            if mergedf[0,ii,jj] > 0 and mergedf[13,ii,jj] > 0:
                polar_goes_values[ii,jj] = ((mergedf[0, ii, jj] + mergedf[13, ii, jj])/2)
            elif mergedf[13,ii,jj] > 0:
                polar_goes_values[ii,jj] = mergedf[13,ii,jj]
            else:
                polar_goes_values[ii,jj] = mergedf[0,ii,jj]
    # compare the enteries
    polar_goes_values = mergedf[0, :, :] + mergedf[13, :, :]
    return polar_goes_values
    
# define geostationary gridcell 24hour averaged FRP
def prev_FRP(mergedf, date, prev_24_hr_date):
    # find files from up to 24 hours in past (keep count)
    # while the date is not bigger than 24 hours ago, then open files, extract the goods for FRP only
    # if there's an FRP already in the grid, update a count and add FRP to summating in temp var 
    time_traveler = date - timedelta(hours=1)
    prev_FRP_values = mergedf[10,:,:]
    file_count = np.zeros((grid[0], grid[1])) + 1  #initial FRP count will be 1 from FRP at current time
    summed_frp = mergedf[10,:,:]
    tt = 1
    #while time_traveler is not prev_24_hr_date:
    while tt <= 24:
        if os.path.exists("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (time_traveler.strftime("%y%m%d%H"))):
            file_past = np.load("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (time_traveler.strftime("%y%m%d%H")))
            frp_past = file_past[0,:,:]
            ff = frp_past > 0
            ff.astype(np.int)
#temp            file_count = ff + file_count
            file_count = ff.astype(np.int) + file_count
            summed_frp = summed_frp + frp_past
            # print('max of summed: ', np.max(summed_frp))
            # print('max of count: ', np.max(file_count))
        time_traveler = time_traveler - timedelta(hours=1)
        tt = tt+1
    prev_FRP_values = (summed_frp) / file_count
    # open just the FRP values (so 0 entry in the 10x337x451 array)
    # create variable size 337x451 that takes a rolling sum of all FRP
    # divide by number of files
    # return the array prev_FRP_values 
    return prev_FRP_values

# define geostationary gridcell yesterday averaged FRP
def prev_day_frp(mergedf, date, yesterday_date):
    # define first time of yesterday at 0 utc
    # define 23 utc of same day
    # pull files and make average
    # store into variable prev_day_frp_acg
#    yesterday_date = yesterday_date.replace(hour=0)
    # set timetraveler to add an hour
#    time_traveler_yester = yesterday_date
    # Initialize assuming that there are no FRP values from yesterday
#    prev_day_values = np.zeros((grid[0], grid[1]))
#    file_count_yester = np.zeros((grid[0], grid[1])) + 1  #initial FRP count will be 1 from FRP at current time
#    summed_frp_yester = mergedf[10,:,:]
#    ttt = 0
#    if date.hour == 0:
#    #while time_traveler is not prev_24_hr_date:
#        while ttt <= 23:
#            if os.path.exists("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (time_traveler_yester.strftime("%y%m%d%H"))):
#                file_past_2 = np.load("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (time_traveler_yester.strftime("%y%m%d%H")))
#                frp_past_2 = file_past_2[14,:,:]
#                ff = frp_past_2 > 0
#                ff.astype(np.int)
#                file_count_yester = ff.astype(np.int) + file_count_yester
#                summed_frp_yester = summed_frp_yester + frp_past_2
#                # print('max of summed: ', np.max(summed_frp_yester))
#                # print('max of count: ', np.max(file_count_yester))
#            time_traveler_yester = time_traveler_yester + timedelta(hours=1)
#            ttt = ttt+1
#        prev_day_values = (summed_frp_yester) / file_count_yester  
#    else:
#        if os.path.exists("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (time_traveler_yester.strftime("%y%m%d%H"))):
#            file_past_2 = np.load("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (time_traveler_yester.strftime("%y%m%d%H")))
#            frp_past_2 = file_past_2[11,:,:]
#            prev_day_values = frp_past_2
#        else:
#            while ttt <= 23:
#                if os.path.exists("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (time_traveler_yester.strftime("%y%m%d%H"))):
#                    file_past_2 = np.load("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (time_traveler_yester.strftime("%y%m%d%H")))
#                    frp_past_2 = file_past_2[14,:,:]
#                    ff = frp_past_2 > 0
#                    ff.astype(np.int)
#                    file_count_yester = ff.astype(np.int) + file_count_yester
#                    summed_frp_yester = summed_frp_yester + frp_past_2
#                    # print('max of summed: ', np.max(summed_frp_yester))
#                    # print('max of count: ', np.max(file_count_yester))
#                time_traveler_yester = time_traveler_yester + timedelta(hours=1)
#                ttt = ttt+1
#        prev_day_values = (summed_frp_yester) / file_count_yester              
   
#old way    
     # set yesterday date to 0utc
     yesterday_date = yesterday_date.replace(hour=0)
     # set timetraveler to add an hour
     time_traveler_yester = yesterday_date
     # Initialize assuming that there are no FRP values from yesterday
     prev_day_values = np.zeros((grid[0], grid[1]))
     file_count_yester = np.zeros((grid[0], grid[1])) + 1  #initial FRP count will be 1 from FRP at current time
     summed_frp_yester = mergedf[10,:,:]
     ttt = 0
     #while time_traveler is not prev_24_hr_date:
     while ttt < 23:
         if os.path.exists("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (time_traveler_yester.strftime("%y%m%d%H"))):
             file_past_2 = np.load("./polar_goes_rap_proc/%s_RAP_FRP_proc.npy" % (time_traveler_yester.strftime("%y%m%d%H")))
             frp_past_2 = file_past_2[14,:,:]
             ff = frp_past_2 > 0
             ff.astype(np.int)
 #temp            file_count = ff + file_count
             file_count_yester = ff.astype(np.int) + file_count_yester
             summed_frp_yester = summed_frp_yester + frp_past_2
             # print('max of summed: ', np.max(summed_frp_yester))
             # print('max of count: ', np.max(file_count_yester))
         time_traveler_yester = time_traveler_yester + timedelta(hours=1)
         ttt = ttt+1
     prev_day_values = (summed_frp_yester) / file_count_yester
     return prev_day_values
       
# define a main function %
def main():
    
    for m in range(1, 13):
        for d in range(1, 32):
            for h in range(0,24):
                    
    ## To make work with future polar orbitors ##
    # for m in range(1, 13):
    #     for d in range(1, 32):
    #         for h in range(24):
                try:
                    date = datetime(year, m, d, h)
                    next_date = date + timedelta(hours=1)
                    prev_24_hr_date = date - timedelta(hours=24)
                    yesterday_date = date - timedelta(hours=24)
                    # create a list of all the files with this exact date
#                    date_files = glob.glob('scratch2/BMC/gsd-hpcs/Christina.E.Kumler/machine_learning/frp_project/*_RAP_processed/%s_*.csv' % (date.strftime("%y%m%d%H")))
                    #test on local
#old for single sat                    date_files = glob.glob('/Users/christina.bonfanti/Documents/machine_learning/FRP/process_rap_goes_polar/*_rap_processed/%s/%s_*.csv' 
#                                           % (date.strftime("%Y%m%d")), (date.strftime("%y%m%d%H")))
                    modis_date_files = glob.glob('./modis_rap_processed/%s/%s_*.csv' 
                                                 % (date.strftime("%Y%m%d"), date.strftime("%y%m%d%H")))
                    jpss_date_files = glob.glob('./jpss_rap_processed/%s/%s_*.csv' 
                                                 % (date.strftime("%Y%m%d"), date.strftime("%y%m%d%H")))
                    noaa_date_files = glob.glob('./noaa_rap_processed/%s/%s_*.csv' 
                                                 % (date.strftime("%Y%m%d"), date.strftime("%y%m%d%H")))
                    goes_date_files = glob.glob('./skip_jan_goes_rap_processed/%s/%s_*.csv' 
                                                 % (date.strftime("%Y%m%d"), date.strftime("%y%m%d%H")))
                    # loop through these file directories based on date and hour, concat the data into one dataframe
                    li = []
                    #jpss
                    for file in jpss_date_files:
                        df = pd.read_csv(file, index_col=None, header=0, parse_dates=['dateTime'])
                        li.append(df)
                        print('Processing File: ', file)
                    #noaa
                    for file in noaa_date_files:
                        dff = pd.read_csv(file, index_col=None, header=0, parse_dates=['dateTime'])
                        li.append(dff)
                        print('Processing File: ', file)
                    #goes
                    for file in goes_date_files:
                        dfff = pd.read_csv(file, index_col=None, header=0, parse_dates=['dateTime'])
                        li.append(dfff)
                        print('Processing File: ', file)
                    #modis - who is stored by day instead of hour
                    for file in modis_date_files:
                        dm = pd.read_csv(file, index_col=None, header=0, parse_dates=['dateTime'])
                        # if hour is desired hour, store the appropriate entry...
                        masky = (dm['dateTime'] >= date) & (dm['dateTime'] <= date)
                        li.append(dm.loc[masky])
#                        print('Processing File: ', file)
                    frp_all = pd.concat(li, axis=0, ignore_index=True)
#                    size_frp_all = frp_all.shape
                    # Run through a short loop to remove lat/lon points outside of RAP domain
                    #check if lats are greater or less than a value
                        #then check if lons are greater or less than a value
                    date_str = date.strftime("%Y-%m-%d %H:%M:%S")
                    date_str2 = next_date.strftime("%Y-%m-%d %H:%M:%S")
                    mask = (frp_all['dateTime'] >= date_str) & (frp_all['dateTime'] < date_str2)
                    frp_regrid = frp_all.loc[mask]
                    frp_regrid.reset_index(drop=True, inplace=True)
                    
                    # csv file outputted for rounded hour's FRP points all sats
                    frp_all.to_csv('./polar_goes_rap_proc/RAP_polar_goes_FRP_%s.csv' %date.strftime("%y%m%d%H"), index = False)
                    
                    # Create hour file
                    print(" Working on date: ", date)
                                  

                    create_proc(date, frp_regrid, prev_24_hr_date, yesterday_date)
                except ValueError as e:
                    "date doesn't exist"
    

if __name__ == '__main__':
    main()
