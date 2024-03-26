#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:07:32 2024

@author: kangsun
"""

import sys,os,glob
sys.path.append('/home/kangsun/Oversampling_matlab')
from popy import Level3_List
sys.path.append('/home/kangsun/IDS/IDS')
from IDS import AgriRegion
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#%% download nei, only need once
from IDS import Inventory
nei = Inventory()
nei.download_NEI_ag('/home/kangsun/data/NEI')
#%%
# l3 data downloaded from https://uofi.app.box.com/s/wqmsg4wd605g0vk467czrcjvtw4m88ts
l3_path_pattern = '/home/kangsun/data/IASINH3/L3/IASINH3L3_flux01/CONUS_%Y_%m_%d.nc'
nei_dir = '/home/kangsun/data/NEI'
start_dt = dt.datetime(2020,1,1)
end_dt = dt.datetime(2021,12,31)
west,east,south,north = -105,-83,37,45
# create a region with lat/lon bounds
mdwst = AgriRegion(start_dt=start_dt,end_dt=end_dt,
                    west=west,east=east,south=south,north=north)
# load emissions in the region
mdwst.get_region_emission(l3_path_pattern=l3_path_pattern,
                          masking_kw=dict(
                              nei_dir=nei_dir,
                              max_topo_emission=1e-9,
                              max_chem_emission=1e-9
                              ),
                          fit_topo_kw=dict(
                              resample_rule='1M',
                              max_iter=1,
                              min_windtopo=-np.inf,
                              max_windtopo=np.inf
                              ),
                          fit_chem_kw=dict(
                              resample_rule='1M',
                              max_iter=1,
                              min_windtopo=-np.inf,
                              max_windtopo=np.inf,
                              max_wind_column=np.inf))
#%%
plt.close('all')
fig,axs = plt.subplots(4,2,sharex=True,sharey=True,constrained_layout=True,
                       subplot_kw=dict(projection=ccrs.PlateCarree()),
                       figsize=(15,10))
flds = ['column_amount','wind_column','wind_column_topo','wind_column_topo_chem',
        'surface_altitude','NEI','topo_mask','chem_mask']
vlims = [(0,.5e-3),(-1e-8,1e-8),(-1e-8,1e-8),(-1e-8,1e-8),
         (0,1500),(-1e-8,1e-8),(0,1),(0,1)]

for ax,fld,vlim in zip(axs.ravel(order='F'),flds,vlims):
    figout = mdwst.l3all.plot(fld,vmin=vlim[0],vmax=vlim[1],existing_ax=ax)