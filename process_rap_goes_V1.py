#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 14:21:11 2020

@author: christina.bonfanti
"""

# Code to read the RAP files and polar orbiter satellites
# saves into csv files
# This code reads by FRP polar file and finds the corresponding RAP


#!/usr/bin/env python

import pandas as pd
import numpy as np
import pygrib
import datetime
import netCDF4 as nc
from pyproj import Proj
import os
#from datetime import datetime, timedelta
from scipy.spatial.distance import cdist

#rap_directory = '/Users/christina.bonfanti/Documents/machine_learning/FRP/proc_local_RAP_GOES/RAP/'
#data_directory = '/Users/christina.bonfanti/Documents/machine_learning/FRP/proc_local_RAP_GOES/goes/'
#out_directory = '/Users/christina.bonfanti/Documents/machine_learning/FRP/proc_local_RAP_GOES/goes_rap_processed/'
#goes_map = 'goes_lat_lons.csv'

#uncomment for Hera
rap_directory = '/scratch1/RDARCH/rda-aidata/rap/grib2/'
data_directory = '/scratch2/BMC/gsd-hpcs/Christina.E.Kumler/machine_learning/frp_project/goesABI_2019/fdcc/001/'
out_directory = '/scratch2/BMC/gsd-hpcs/Christina.E.Kumler/machine_learning/frp_project/skip_jan_goes_rap_processed/'
goes_map = '/scratch2/BMC/gsd-hpcs/Christina.E.Kumler/machine_learning/frp_project/goes_lat_lons.csv'

columns = ['dateTime', 'sat', 'id', 'lat', 'lon', 'FRP', 'radar', 'vis', 'windspeed', 'temp2m', 'RH', 'Uwind', 'Vwind', 'precip', 'temp_sat']

def open_map(m):
    d = {}
    with open(m, 'r') as f:
        for l in f:
            li = l.split()
            if not int(li[5]):
                d[int(li[0])] = [int(li[3]), int(li[4])]
    return d

# old way lat_lon_map = open_map('lat_lon_map_RAP.txt')


def gatherData(time, f, outf):

    rap_file = os.path.join(rap_directory, time.strftime("%Y%m%d"), time.strftime("%y%j%H000000"))

    print("I found the RAP data: ", rap_file)

#    jpss_dir = os.path.join(data_directory)
#    rap_file = os.path.join(rap_directory,  time.strftime("%y%j%H000000"))
#    rap_file = os.path.join(rap_directory, time.strftime("%Y%m%d"),  time.strftime("%y%j%H000000"))


    fdcc_file = os.path.join(data_directory, f)

    print("I got to the satellite files: ", fdcc_file)

    df = gatherRAPandGOES(time, rap_file, fdcc_file)

    df.to_csv(outf, index=False)


def gatherRAPandGOES(time, rap_f, closest_fdcc_file):

#    print("I got into gatherHRRRandGOES")
        # open map with header ,gridlat,i,j,gridlon
    goes_lat_lon = pd.read_csv(goes_map)
    
    # get the hour from the file
    f_date_string = closest_fdcc_file.split('_')[-1][1:14]
    f_date = datetime.datetime.strptime(f_date_string, "%Y%j%H%M%S")

    df = pd.DataFrame(columns=columns)

    grbs = pygrib.open(rap_f)   
#    print("I opened the RAP file")

    fdcc_ds = nc.Dataset(closest_fdcc_file)

    fdcc_frp = np.asarray(fdcc_ds.variables["Power"][:])
    fdcc_temp = np.asarray(fdcc_ds.variables["Temp"][:])
    fdcc_frp = fdcc_frp.flatten()
    fdcc_temp = fdcc_temp.flatten()
    
    print('here is the shape of fdcc_frp: ', fdcc_frp.shape)

    #stores all the RAP lats and lons into single dataframe for cdist

    raplats = grbs[1].latitudes.flatten()
    raplons = grbs[1].longitudes.flatten()-360 
    rap_map = pd.DataFrame({"raplat": raplats, "raplon": raplons})

    first = int(0)
    count = int(0)
    for i in fdcc_frp: 
        ind = count
        #needs to be greater than zero because there are many -9 values
        if i >= 0:
        
        # single file test
#        grbs = pygrib.open('/Users/christina.bonfanti/Documents/machine_learning/FRP/proc_local_RAP_polar/RAP/1823216000000')
#        jpss_ds = pd.read_csv('/Users/christina.bonfanti/Documents/machine_learning/FRP/proc_local_RAP_polar/polar_modis/MODIS_C6_Global_MCD14DL_NRT_2018225')
#        rap_lat_lon = pd.read_csv('concat_lat_lon_RAP.csv')
        
            lat_test = grbs[1].latitudes.flatten()
            lon_test = grbs[1].longitudes.flatten()-360
            rap_map = pd.DataFrame({"raplat": lat_test, "raplon": lon_test})
        
        # x == 1500 y=2500 in radians?
        # GOES fixed grid projection x-coordinate
            frp_lat = goes_lat_lon['lat']
            frp_lon = goes_lat_lon['lon']
            frp_pt = pd.DataFrame({"Lat": frp_lat, "Lon": frp_lon})
            single = np.array((frp_pt.at[ind,'Lat'], frp_pt.at[ind,'Lon']))
            single = single.reshape(1,2)
            print('single: ', single)
            loc_rap = cdist(rap_map, single)
            rap_close = loc_rap.argmin()
            if np.isnan(frp_lat[ind]):
                rap_close = float("NAN")
            print('rap_close is: ', rap_close)
#        print("this is ind: ", ind)
#        print("this is first: ", first)
        
            print('here is the i with an frp: ', i, 'and index ', ind)        
            if first == 0:
                first = int(1)
 #           print("I went into first = 0 ")  
            # if in SH, add Nan's and move on
                if np.isnan(rap_close):
                    data_df = pd.DataFrame({"lat": float("NAN"),
                                            "lon": float("NAN"),
                                            "radar": float("NAN"),
                                            "vis": float("NAN"),
                                            "windspeed": float("NAN"),
                                            "temp2m": float("NAN"),
                                            "RH": float("NAN"),
                                            "Uwind": float("NAN"),
                                            "Vwind": float("NAN"),
                                            "precip": float("NAN"),
                                            "FRP": float("NAN"),
                                            "temp_sat": float("NAN"),
                                            "id": float("NAN"),
                                            "sat": float("NAN"),
                                            "dateTime": float("NAN")}, index=[0])
                else:
                    data_df = pd.DataFrame({"lat": grbs[1].latitudes[rap_close],
                                            "lon": grbs[1].longitudes[rap_close]-360,
                                            "radar": grbs[1].values.flatten()[rap_close],
                                            "vis": grbs[2].values.flatten()[rap_close],
                                            "windspeed": grbs[3].values.flatten()[rap_close],
                                            "temp2m": grbs[4].values.flatten()[rap_close],
                                            "RH": grbs[5].values.flatten()[rap_close],
                                            "Uwind": grbs[6].values.flatten()[rap_close],
                                            "Vwind": grbs[7].values.flatten()[rap_close],
                                            "precip": grbs[8].values.flatten()[rap_close],
                                            "FRP": i,
                                            "temp_sat": fdcc_temp[ind],
                                            "id": rap_close,
                                            "sat": 'goes',
                                            "dateTime": [f_date]}, index=[0])
                df = pd.DataFrame(data=data_df, index=[0])
 #           print("I made a df this size: ", df.shape)
                del data_df
            else:
 #           print("I went into first /= 0 ")
            # Append FRP and RAP together in a dataframe
                if np.isnan(rap_close):
                    data_df2 = pd.DataFrame({"lat": float("NAN"),
                                            "lon": float("NAN"),
                                            "radar": float("NAN"),
                                            "vis": float("NAN"),
                                            "windspeed": float("NAN"),
                                            "temp2m": float("NAN"),
                                            "RH": float("NAN"),
                                            "Uwind": float("NAN"),
                                            "Vwind": float("NAN"),
                                            "precip": float("NAN"),
                                            "FRP": float("NAN"),
                                            "temp_sat": float("NAN"),
                                            "id": float("NAN"),
                                            "sat": float("NAN"),
                                            "dateTime": float("NAN")}, index=[0])
                else:      
                    data_df2 = pd.DataFrame({"lat": grbs[1].latitudes[rap_close],
                                            "lon": grbs[1].longitudes[rap_close]-360,
                                            "radar": grbs[1].values.flatten()[rap_close],
                                            "vis": grbs[2].values.flatten()[rap_close],
                                            "windspeed": grbs[3].values.flatten()[rap_close],
                                            "temp2m": grbs[4].values.flatten()[rap_close],
                                            "RH": grbs[5].values.flatten()[rap_close],
                                            "Uwind": grbs[6].values.flatten()[rap_close],
                                            "Vwind": grbs[7].values.flatten()[rap_close],
                                            "precip": grbs[8].values.flatten()[rap_close],
                                            "FRP": i,
                                            "temp_sat": fdcc_temp[ind],
                                            "id": rap_close,
                                            "sat": 'goes',
                                            "dateTime": [f_date]}, index=[0])          
                hold_df_2 = pd.DataFrame(data=data_df2, index=[0])                        
                df = df.append(hold_df_2, ignore_index=True)
 #           print("I appended a df this size: ", df.shape)        
                del hold_df_2
        count = count + 1
    grbs.close()
    return df

def roundTime(dt=None, roundTo=60):
   """Round a datetime object to any time lapse in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   if dt == None : dt = datetime.datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)


def main():

    goes_dir = os.path.join(data_directory)
    count = 1 #new
    sorted_goes_dir = sorted(os.listdir(goes_dir))
    for f in sorted_goes_dir:
        f_date_string = f.split('_')[-1][1:14]
        date = datetime.datetime.strptime(f_date_string, "%Y%j%H%M%S")

        # Polars need to be rounded to nearest hour
        date = roundTime(date, roundTo=60*60)
        if date.month > 7:
#        if date.month > 1:
            # following sequence deals with multiple polar obs in an hour
            if count == 1: #new
                date_remember = date #new
                version = 0 #new
                count = 2
            else: #new
                if date_remember == date: #new
                    version = version+1 #new
                else: #new
                    date_remember = date #new
                    version = 0 #new
    #        date = datetime.strptime(f_date_string, "%Y%m%d%H%M")
    
            print("your file: ", f, "your date: ", date)
            try:
#                check_file = os.path.join(rap_directory, date.strftime("%y%j%H000000"))
                check_file = os.path.join(rap_directory, date.strftime("%Y%m%d"), date.strftime("%y%j%H000000"))
                if date.month == '1':
                    continue
                print('check_file needs to be: ', check_file)
                if os.path.exists(check_file):
                     if not os.path.isdir(os.path.join(out_directory, date.strftime("%Y%m%d"))):
                          print ("%s doesn't exist, creating directory" % os.path.join(out_directory, date.strftime("%Y%m%d")))
                          os.mkdir(os.path.join(out_directory, date.strftime("%Y%m%d")))
                     print("I am going to run outfile now!")
                     outfile = os.path.join(out_directory, "%s/%s%s%s.csv" % (date.strftime("%Y%m%d"), date.strftime("%y%m%d%H"), "_V_", version))
                     if os.path.isfile(outfile): # skip this file
                         print('this exists!')
                     else:
                         print ("Gathering data for %s" % date.strftime("%Y-%m-%d %H:%M:%S"))
                         gatherData(date, f, outfile)
                         print ("Wrote to %s" % outfile)
            except ValueError as e:
                pass
                print("date doesn't exist")


if __name__ == '__main__':

    main()

