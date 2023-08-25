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


rap_directory = '/scratch1/RDARCH/rda-aidata/rap/grib2'
data_directory = '/scratch2/BMC/gsd-hpcs/Christina.E.Kumler/machine_learning/frp_project/polar_orbitors/MODIS/global/'
out_directory = '/scratch2/BMC/gsd-hpcs/Christina.E.Kumler/machine_learning/frp_project/modis_rap_processed'

#rap_directory = '/Users/christina.bonfanti/Documents/machine_learning/FRP/proc_local_RAP_polar/RAP/'
#data_directory = '/Users/christina.bonfanti/Documents/machine_learning/FRP/proc_local_RAP_polar/polar_modis/'
#out_directory = '/Users/christina.bonfanti/Documents/machine_learning/FRP/proc_local_RAP_polar/processed_modis/'

columns = ['dateTime', 'sat', 'id', 'lat', 'lon', 'FRP', 'bright_t13', 'radar', 'vis', 'windspeed', 'temp2m', 'RH', 'Uwind', 'Vwind', 'precip', 'mask', 'confidence']

def open_map(m):
    d = {}
    with open(m, 'r') as f:
        for l in f:
            li = l.split()
            if not int(li[5]):
                d[int(li[0])] = [int(li[3]), int(li[4])]
    return d

# old way lat_lon_map = open_map('lat_lon_map_RAP.txt')

# open map with header ,gridlat,i,j,gridlon
rap_lat_lon = pd.read_csv('concat_lat_lon_RAP.csv')


def gatherData(time, f, outf):

#    rap_file = os.path.join(rap_directory, time.strftime("%y%j%H000000"))

#    print("I found the RAP data: ", rap_file)

#    jpss_dir = os.path.join(data_directory)
    jpss_file = os.path.join(data_directory, f)
#    jpss_dir = os.path.join(data_directory, '/polar_orbitors/JPSS-1/fire')

    print("I got to the satellite data: ", jpss_file)

    #df = gatherGOES(closest_fdcc_file, closest_acmc_file, df, outf)
#    df = gatherHRRRandGOES(time, rap_file, closest_jpss_file)
    df = gatherRAPandPOLAR(time, jpss_file)
#    df = gatherRAPandPOLAR(time, rap_file, jpss_file)


    df.to_csv(outf, index=False)


def gatherRAPandPOLAR(time, jpss_f):

#    print("I got into gatherHRRRandGOES")

    df = pd.DataFrame(columns=columns)
#    rap_f = os.path.join(rap_directory, time.strftime("%y%j%H000000"))

#    grbs = pygrib.open(rap_f)   
#    print("I opened the RAP file")

    jpss_ds = pd.read_csv(jpss_f)   
#    print("I opened the current polar file")
  
    #stores all the RAP lats and lons into single dataframe for cdist

#    raplats = grbs[1].latitudes.flatten()
#    raplons = grbs[1].longitudes.flatten()-360 
#    rap_map = pd.DataFrame({"raplat": raplats, "raplon": raplons})

    first = int(0)
    for ind in jpss_ds.index: 
        
        # single file test
#        grbs = pygrib.open('/Users/christina.bonfanti/Documents/machine_learning/FRP/proc_local_RAP_polar/RAP/1823216000000')
#        jpss_ds = pd.read_csv('/Users/christina.bonfanti/Documents/machine_learning/FRP/proc_local_RAP_polar/polar_modis/MODIS_C6_Global_MCD14DL_NRT_2018225')
#        rap_lat_lon = pd.read_csv('concat_lat_lon_RAP.csv')
        
        # MODIS is 24 hours, so need to pull RAP file each new hour
        modis_hour_str = jpss_ds['acq_time'][ind]
        mn = int(modis_hour_str[3:5])
        hr = int(modis_hour_str[0:2])
        time = time.replace(hour=hr, minute=mn)
        time = roundTime(time, roundTo=60*60)
        
        rap_f = os.path.join(rap_directory, time.strftime("%Y%m%d"), time.strftime("%y%j%H000000"))
        if os.path.exists(rap_f):

            grbs = pygrib.open(rap_f)   
        
            lat_test = grbs[1].latitudes.flatten()
            lon_test = grbs[1].longitudes.flatten()-360
            rap_map = pd.DataFrame({"raplat": lat_test, "raplon": lon_test})
        
            frp_lat = jpss_ds['latitude']
            frp_lon = jpss_ds['longitude']
            frp_pt = pd.DataFrame({"Lat": frp_lat, "Lon": frp_lon})
            single = np.array((frp_pt.at[ind,'Lat'], frp_pt.at[ind,'Lon']))
            single = single.reshape(1,2)
            loc_rap = cdist(rap_map, single)
            rap_close = loc_rap.argmin()
            if np.isnan(frp_lat[ind]):
                rap_close = float("NAN")
            print("this is ind: ", ind)
            print("this is first: ", first)
        
        
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
                                            "bright_t13": float("NAN"),
                                            "confidence": float("NAN"),
                                            "mask": float("NAN"),
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
                                            "FRP": jpss_ds['frp'][ind],
                                            "bright_t13": jpss_ds['bright_t31'][ind],
                                            "confidence": jpss_ds['confidence'][ind],
                                            "mask": jpss_ds['version'][ind],
                                            "id": rap_close,
                                            "sat": 'modis',
                                            "dateTime": [time]}, index=[0])
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
                                             "bright_t13": float("NAN"),
                                             "confidence": float("NAN"),
                                             "mask": float("NAN"),
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
                                             "FRP": jpss_ds['frp'][ind],
                                             "bright_t13": jpss_ds['bright_t31'][ind],
                                             "confidence": jpss_ds['confidence'][ind],
                                             "mask": jpss_ds['version'][ind],
                                             "id": rap_close,
                                             "sat": 'modis',
                                             "dateTime": [time]}, index=[0])          
                hold_df_2 = pd.DataFrame(data=data_df2, index=[0])                        
                df = df.append(hold_df_2, ignore_index=True)
 #           print("I appended a df this size: ", df.shape)        
                del hold_df_2

            # else it was a missing RAP and should be NAN
        else:
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
                                    "bright_t13": float("NAN"),
                                    "confidence": float("NAN"),
                                    "mask": float("NAN"),
                                    "id": float("NAN"),
                                    "sat": float("NAN"),  
                                    "dateTime": float("NAN")}, index=[0])

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

    jpss_dir = os.path.join(data_directory)
    count = 1 #new
    sorted_dir = sorted(os.listdir(jpss_dir))
    for f in sorted_dir:
        f_date_string = f[-11:-4]
        date = datetime.datetime.strptime(f_date_string, "%Y%j")

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

#        print("your file: ", f, "your date: ", f_date)
        try:
            check_file = os.path.join(rap_directory, date.strftime("%Y%m%d"), date.strftime("%y%j%H000000"))      
            if os.path.exists(check_file):
                 if not os.path.isdir(os.path.join(out_directory, date.strftime("%Y%m%d"))):
                      print ("%s doesn't exist, creating directory" % os.path.join(out_directory, date.strftime("%Y%m%d")))
                      os.mkdir(os.path.join(out_directory, date.strftime("%Y%m%d")))

                 outfile = os.path.join(out_directory, "%s/%s%s%s.csv" % (date.strftime("%Y%m%d"), date.strftime("%y%m%d%H"), "_V_", version))

                 if os.path.isfile(outfile):
                     print('This file exists already!')
                 else:
                     print ("Gathering data for %s" % date.strftime("%Y-%m-%d %H:%M:%S"))
                     gatherData(date, f, outfile)
                     print ("Wrote to %s" % outfile)
        except ValueError as e:
            pass
            print("date doesn't exist")
        


if __name__ == '__main__':

    main()

