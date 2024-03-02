import pandas as pd
import os
import netCDF4
from popy1 import popy,datetime2datenum,datedev_py,Level3_Data, Level3_List  # Replace with your actual module and class names
import datetime as dt
import zipfile, requests, os, sys, glob
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
import pickle
import matplotlib.pyplot as plt
from matplotlib import path
from scipy.signal import savgol_filter
from scipy.ndimage import percentile_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import kendalltau
from astropy.convolution import convolve_fft
from netCDF4 import Dataset
import statsmodels.formula.api as smf
import logging
from shapely.ops import unary_union

# Identify the L3 data NetCDF files
# l3_files = []
# dt_array = pd.period_range(start='2019-09-22',end='2019-09-30',freq='1D')
# for ip,p in enumerate(dt_array):
#     l3_fn = p.strftime(l3_path_pattern)
#     if os.path.exists(l3_fn):
#         l3_files.append(l3_fn)
#     else:
#         continue

# # Instantiate the Level3_List object
# level3_list = Level3_List(dt_array=dt_array,west=-128,east=-65,south=24,north=50) # CONUS
# level3_list.read_nc_pattern(l3_path_pattern=l3_path_pattern)




class Ammonia():
    '''class for Level4 data (NH3 emission) calculated from Level3 data'''
    def __init__(self,geometry=None,xys=None,start_dt=None,end_dt=None,
                 west=None,east=None,south=None,north=None):
        '''
        geometry:
            a list of tuples for the polygon, e.g., [(xarray,yarray)], or geometry in a gpd row
        start/end_dt:
            datetime objects accurate to the day
        west, east, south, north:
            boundary of the region
        '''
        import shapely
        self.logger = logging.getLogger(__name__)
        self.start_dt = start_dt or dt.datetime(2018,5,1)
        self.end_dt = end_dt or dt.datetime.now()
        if geometry is None and xys is not None:
            geometry = xys
        if isinstance(geometry,list):
            xys = geometry
            self.west = west or np.min([np.min(xy[0]) for xy in xys])
            self.east = east or np.max([np.max(xy[0]) for xy in xys])
            self.south = south or np.min([np.min(xy[1]) for xy in xys])
            self.north = north or np.max([np.max(xy[1]) for xy in xys])
            self.xys = xys
        elif isinstance(geometry,shapely.geometry.multipolygon.MultiPolygon):
            bounds = geometry.bounds
            self.west = west or bounds[0]
            self.east = east or bounds[2]
            self.south = south or bounds[1]
            self.north = north or bounds[3]
            self.xys = [g.exterior.xy for g in geometry.geoms] #zitong edit
        elif isinstance(geometry,shapely.geometry.polygon.Polygon):
            bounds = geometry.bounds
            self.west = west or bounds[0]
            self.east = east or bounds[2]
            self.south = south or bounds[1]
            self.north = north or bounds[3]
            self.xys = [geometry.exterior.xy]
        elif geometry is None:
            self.west = west
            self.east = east
            self.south = south
            self.north = north
            self.xys = [([west,west,east,east],[south,north,north,south])]

    def get_l3(self,l3_path_pattern,file_freq='1D',lonlat_margin=0.5,xgb_pblh_path_pattern=None,
               topo_kw=None,if_smooth_X=True,X_smooth_kw=None,cp_kw=None,chem_kw=None):
        '''interface popy level 3 objects. get daily clean/polluted vcd and sfc conc.
        l3_path_pattern:
            time pattern of level3 files, e.g., '/projects/bbkc/zitong/Data/IASINH3L3_flux02/CONUS_%Y_%m_%d.nc'
        file_freq:
            frequency code by which l3 files are saved, e.g., 1D
        lonlat_margin:
            Level3_List will be trimmed this amount broader than Ammonia boundaries
        xgb_pblh_path_pattern:
            add sfc ppb estimated using xgb/amdar-based pblh instead of era5_blh,
            see https://doi.org/10.5194/amt-16-563-2023
        topo_kw:
            key word arguments for fit_topo related functions, may include min/max to mask l3 and args for fit_topography
        if_smooth_X:
            when the l3 record is short, it is better not to smooth scale height
        X_smooth_kw:
            key word arguments to smooth inverse scale height (X), default using savgol, window_length=5;polyorder=3
        cp_kw:
            key word arguments to separate clean/polluted and covert from vcd to sfc conc seperately
        '''
        topo_kw = topo_kw or {}
        chem_kw = chem_kw or {}

        # load level 3 files
        ewsn_dict = dict(west=self.west-lonlat_margin,east=self.east+lonlat_margin,
                            south=self.south-lonlat_margin,north=self.north+lonlat_margin)
        l3ds = Level3_List(pd.period_range(self.start_dt,self.end_dt,freq=file_freq),**ewsn_dict)
        # dt_array = pd.period_range(start=self.start_dt,end=self.end_dt,freq=file_freq)
        # l3ds = Level3_List(dt_array=dt_array,west=-128,east=-65,south=24,north=50)
        fields_name = ['column_amount','wind_topo','wind_column','surface_altitude']
        l3ds.read_nc_pattern(l3_path_pattern,
                            fields_name=fields_name.copy())
        l3 = l3ds.aggregate()
        l3ms,_ = l3ds.resample(rule=topo_kw['resample_rule']) # zitong edit

        # create region mask
        lonmesh,latmesh = np.meshgrid(l3['xgrid'],l3['ygrid'])
        region_mask = np.zeros(l3['num_samples'].shape,dtype=bool)
        for xy in self.xys:
            boundary_polygon = path.Path([(x,y) for x,y in zip(*xy)])
            all_points = np.column_stack((lonmesh.ravel(),latmesh.ravel()))
            region_mask = region_mask | boundary_polygon.contains_points(all_points).reshape(lonmesh.shape)

        if topo_kw is not None:
            if 'max_iter' not in topo_kw.keys():
                topo_kw['max_iter'] = 1

            # create topo mask
            min_windtopo = topo_kw.pop('min_windtopo',0.001)
            max_windtopo = topo_kw.pop('max_windtopo',0.1)
            min_H = topo_kw.pop('min_H',0.1)
            max_H = topo_kw.pop('max_H',2000)
            max_wind_column = topo_kw.pop('max_wind_column',1e-9) #???????
            topo_mask = region_mask &\
            (np.abs(l3['wind_topo']/l3['column_amount'])>=min_windtopo) &\
            (np.abs(l3['wind_topo']/l3['column_amount'])<=max_windtopo) &\
            (l3['surface_altitude']>=min_H) &\
            (l3['surface_altitude']<=max_H) &\
            (l3['wind_column']<=max_wind_column)
            if 'mask' in topo_kw.keys():
                topo_kw['mask'] = topo_kw['mask'] & topo_mask
            else:
                topo_kw['mask'] = topo_mask

            # l3ms,_ = l3ds.resample(rule=topo_kw['resample_rule']) # zitong edit
            l3ms.fit_topography(**topo_kw)
            self.topo_mask = topo_mask
            fields_name.append('wind_column_topo')
            

        if chem_kw is not None:
            print('1')
            max_windtopo = chem_kw.pop('max_windtopo',0.001)
            # print(np.nanmax(l3['column_amount'].reshape(-1, 1)))
            # print(np.nanmin(l3['column_amount'].reshape(-1, 1)))
            min_column_amount = chem_kw.pop('min_column_amount',2.5e-5) # not sure
            max_wind_column = chem_kw.pop('max_wind_column',1e-9) # not sure
            chem_mask = region_mask &\
            (l3['column_amount']>=min_column_amount) &\
            (np.abs(l3['wind_topo']/l3['column_amount'])<=max_windtopo) &\
            (l3['wind_column']<=max_wind_column)
            if 'mask' in chem_kw.keys():
                chem_kw['mask'] = chem_kw['mask'] & chem_mask
            else:
                chem_kw['mask'] = chem_mask

            # l3ms,_ = l3ds.resample(rule=topo_kw['resample_rule']) # zitong edit
            l3ms.fit_chemistry(**chem_kw)
            self.chem_mask = chem_mask
            fields_name.append('wind_column_topo_chem')            

        # attach results to self
        self.l3ds = l3ds
        tmpdf = l3ms.df.copy()
        l3ms,_ = l3ds.resample(rule=topo_kw['resample_rule'])
        l3ms.df = tmpdf
        self.l3ms = l3ms
        self.l3 = l3ds.aggregate()
        self.topo_mask = topo_mask
        self.region_mask = region_mask

# make the mask E~=0
shp_file = "/projects/bbkc/zitong/Data/tl_2019_us_county/tl_2019_us_county.shp"
gdf = gpd.read_file(shp_file)
excel_file = "/projects/bbkc/zitong/Data/county_ammonia_2020.xlsx"
df = pd.read_excel(excel_file)
merged_gdf = gdf.merge(df, left_on='NAME', right_on='County')
extracted_polygons = merged_gdf[merged_gdf['Emission2020'] <= 400]
mask_polygon = unary_union(extracted_polygons.geometry)

# NH3_emission
l3_path_pattern= '/projects/bbkc/zitong/Data/IASINH3L3_flux004/CONUS_%Y_%m_%d.nc'  # flux004：flux_grid_size=0.04
ammonia = Ammonia(geometry=mask_polygon,start_dt=dt.datetime(2019,9,23),end_dt=dt.datetime(2021,10,14),west=-128,east=-65,south=24,north=50)
topo_kw = {'resample_rule': '1M'}
chem_kw = {'resample_rule': '1M'}
ammonia.get_l3(l3_path_pattern,topo_kw=topo_kw,chem_kw=chem_kw)

# Plot scale height and lifetime
months = pd.period_range(start='2019-09', end='2021-10', freq='M')
timestamps = months.to_timestamp()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
axes[0].plot(timestamps,ammonia.l3ms.df['topo_scale_height']) 
axes[0].set_ylabel('Scale height (m)')
axes[1].plot(timestamps,ammonia.l3ms.df['chem_lifetime'])
axes[1].set_ylabel('Lifetime (h)')
plt.tight_layout()
plt.show()
plt.savefig('Xtao_flux004.png')