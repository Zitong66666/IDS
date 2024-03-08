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
from test_nei import Inventory
import shapely
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes

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
def F_center2edge(lon,lat):
    '''
    function to shut up complain of pcolormesh like 
    MatplotlibDeprecationWarning: shading='flat' when X and Y have the same dimensions as C is deprecated since 3.3.
    create grid edges from grid centers
    '''
    res=np.mean(np.diff(lon))
    lonr = np.append(lon-res/2,lon[-1]+res/2)
    res=np.mean(np.diff(lat))
    latr = np.append(lat-res/2,lat[-1]+res/2)
    return lonr,latr



class Agri_region():
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
        region_shpfilename:
            filename of regions shp
        '''
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

    def process_by_region(self,l3_path_pattern,topo_kw=None,chem_kw=None,
                 region_shpfilename=None, region_num=None):
        
        '''
        calculate scale heights and lifetimes by regions

        region_shpfilename: multipolygon shp filename of regions

        region_num: if not None, calculate for one specific given region
        '''

        topo_scale_heights = {}
        chem_lifetimes = {}
        wind_column_topo = []     

        # for one region
        if region_shpfilename is not None and region_num is not None: 
            gdf = gpd.read_file(region_shpfilename)
            gdf_region = gdf[gdf['Region'] == region_num]
            region_polygon = unary_union(gdf_region.geometry)
            if isinstance(region_polygon,shapely.geometry.polygon.Polygon):
                self.region_xys = [region_polygon.exterior.xy]
            elif isinstance(region_polygon,shapely.geometry.multipolygon.MultiPolygon):
                self.region_xys = [g.exterior.xy for g in region_polygon.geoms]

            self.get_l3(l3_path_pattern=l3_path_pattern,topo_kw=topo_kw,chem_kw=chem_kw, region_xys=self.region_xys)
            topo_scale_heights = self.l3ms.df['topo_scale_height']
            chem_lifetimes = self.l3ms.df['chem_lifetime']
            # wind_column_topo = self.l3ms.df['wind_column_topo']

        # for all regions
        region_names = ['Pacific','Mountain','Northern Plains','Southern Plains','Lake States',
                        'Corn Belt','Delta States','Southeast','Appalachia','Northeast']   
        if region_shpfilename is not None and region_num is None: 
            gdf = gpd.read_file(region_shpfilename)
            region_polygons = unary_union(gdf.geometry)
            region_xys = []
            for i in range(0,11):
                if isinstance(region_polygons,shapely.geometry.polygon.Polygon):
                    region_xys.append([region_polygons.exterior.xy])
                elif isinstance(region_polygons,shapely.geometry.multipolygon.MultiPolygon):
                    region_xys.append([g.exterior.xy for g in region_polygons.geoms])
            self.region_xys = region_xys

            for i in range(0,11):
                region_xy = self.region_xys[i]
                self.get_l3(l3_path_pattern=l3_path_pattern,topo_kw=topo_kw,chem_kw=chem_kw, region_xy=region_xy)
                topo_scale_heights[region_names[i]] = self.l3ms.df['topo_scale_height']
                chem_lifetimes[region_names[i]] = self.l3ms.df['chem_lifetime']
                wind_column_topo = wind_column_topo + self.l3ms.df['wind_column_topo']

        self.topo_scale_heights = topo_scale_heights
        self.chem_lifetimes = chem_lifetimes     

        
        for l3 in self.l3ds:
            month = l3.start_python_datetime.strftime('%Y-%m')
            sh = self.topo_scale_heights[month]
            cl = self.chem_lifetimes[month]
            wc_topo = l3['wind_column']-sh*l3['wind_topo']
            l3['wind_column_topo'] = wc_topo*self.region_mask
            wc_chem = l3['wind_column_topo']-cl*l3['column_amount']
            l3['wind_column_topo_chem'] = wc_chem*self.region_mask
            l3['month'] = month
            l3['date'] = l3.start_python_datetime
        
        l3ds_df = pd.DataFrame(self.l3ds)
        self.l3ds_df = l3ds_df
          
    
    def get_l3(self,l3_path_pattern,file_freq='1D',lonlat_margin=0.5,xgb_pblh_path_pattern=None,
               topo_kw=None,if_smooth_X=True,X_smooth_kw=None,cp_kw=None,chem_kw=None,
               region_xys = None):
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

        # create mask
        lonmesh,latmesh = np.meshgrid(l3['xgrid'],l3['ygrid'])
        region_mask = np.zeros(l3['num_samples'].shape,dtype=bool)
        # create region mask (by polygon)
        for xy in region_xys:
            # boundary_polygon = path.Path(xy)
            boundary_polygon = path.Path([(x,y) for x,y in zip(*xy)])
            all_points = np.column_stack((lonmesh.ravel(),latmesh.ravel()))
            region_mask = region_mask | boundary_polygon.contains_points(all_points).reshape(lonmesh.shape)
        print(np.sum(region_mask==True))

        E0_mask = np.zeros(l3['num_samples'].shape,dtype=bool)
        # create E=0 mask (by raster) based on region mask
        for xy in self.xys:
            boundary_polygon = path.Path(xy)
            # boundary_polygon = path.Path([(x,y) for x,y in zip(*xy)])
            all_points = np.column_stack((lonmesh.ravel(),latmesh.ravel()))
            E0_mask = E0_mask | boundary_polygon.contains_points(all_points).reshape(lonmesh.shape)
        E0_mask = region_mask & E0_mask
        print(np.sum(E0_mask==True))
        if topo_kw is not None:
            if 'max_iter' not in topo_kw.keys():
                topo_kw['max_iter'] = 1

            # create topo mask
            min_windtopo = topo_kw.pop('min_windtopo',0.001)
            max_windtopo = topo_kw.pop('max_windtopo',0.1)
            min_H = topo_kw.pop('min_H',0.1)
            max_H = topo_kw.pop('max_H',2000)
            max_wind_column = topo_kw.pop('max_wind_column',1e-9) #???????
            topo_mask = E0_mask &\
            (np.abs(l3['wind_topo']/l3['column_amount'])>=min_windtopo) &\
            (np.abs(l3['wind_topo']/l3['column_amount'])<=max_windtopo) &\
            (l3['surface_altitude']>=min_H) &\
            (l3['surface_altitude']<=max_H) &\
            (l3['wind_column']<=max_wind_column)
            if 'mask' in topo_kw.keys():
                topo_kw['mask'] = topo_kw['mask'] & topo_mask
            else:
                topo_kw['mask'] = topo_mask

            topo_kw['region_mask'] = region_mask
            # l3ms,_ = l3ds.resample(rule=topo_kw['resample_rule']) # zitong edit
            l3ms.fit_topography(**topo_kw)
            self.topo_mask = topo_mask
            fields_name.append('wind_column_topo')
            

        if chem_kw is not None:
            max_windtopo = chem_kw.pop('max_windtopo',0.001)
            # print(np.nanmax(l3['column_amount'].reshape(-1, 1)))
            # print(np.nanmin(l3['column_amount'].reshape(-1, 1)))
            min_column_amount = chem_kw.pop('min_column_amount',2.5e-5) # not sure
            max_wind_column = chem_kw.pop('max_wind_column',1e-9) # not sure

            chem_mask = E0_mask &\
            (l3['column_amount']>=min_column_amount) &\
            (np.abs(l3['wind_topo']/l3['column_amount'])<=max_windtopo) &\
            (l3['wind_column']<=max_wind_column)
            if 'mask' in chem_kw.keys():
                chem_kw['mask'] = chem_kw['mask'] & chem_mask
            else:
                chem_kw['mask'] = chem_mask

            chem_kw['region_mask'] = region_mask
            # l3ms,_ = l3ds.resample(rule=topo_kw['resample_rule']) # zitong edit
            l3ms.fit_chemistry(**chem_kw)
            self.chem_mask = chem_mask
            fields_name.append('wind_column_topo_chem')            


        # attach results to self
        self.l3ds = l3ds
        tmpdf = l3ms.df.copy()
        l3ms,_ = l3ds.resample(rule=topo_kw['resample_rule'])
        l3ms.df = tmpdf
        self.l3ms = l3ms # L3 list
        self.l3 = l3ds.aggregate() # L3 Data
        self.topo_mask = topo_mask
        self.region_mask = region_mask


# make the mask E~=0
# shp_file = "/projects/bbkc/zitong/Data/tl_2019_us_county/tl_2019_us_state.shp"
# gdf = gpd.read_file(shp_file)
# excel_file = "/projects/bbkc/zitong/Data/county_ammonia_2020.xlsx"
# df = pd.read_excel(excel_file)
# merged_gdf = gdf.merge(df, left_on='NAME', right_on='County')
# extracted_polygons = merged_gdf[merged_gdf['Emission2020'] <= 400]
# mask_polygon = unary_union(extracted_polygons.geometry)        
your_path = '/projects/bbkc/zitong/Data/NEI_gridded'
monthly_filenames = [os.path.join(your_path,f) for f in os.listdir(your_path) if os.path.isfile(os.path.join(your_path, f))]
nei = Inventory().read_NEI_ag(monthly_filenames)
nei.create_mask_from_sum_NH3(minNH3th = 50)


# NH3_emission
l3_path_pattern= '/projects/bbkc/zitong/Data/IASINH3L3_flux02/CONUS_%Y_%m_%d.nc'  # flux004：flux_grid_size=0.04
region_shp_file = "/projects/bbkc/zitong/Data/tl_2019_us_county/tl_2019_us_state.shp" ## process_by_region
agri_region = Agri_region(geometry=nei['boundary_mask'],start_dt=dt.datetime(2019,9,23),end_dt=dt.datetime(2021,10,14),west=-128,east=-65,south=24,north=50)#, region_num = 7
topo_kw = {'resample_rule': '1M'}
chem_kw = {'resample_rule': '1M'}
agri_region.process_by_region(l3_path_pattern,topo_kw=topo_kw,chem_kw=chem_kw,
                  region_shpfilename = region_shp_file, region_num = 6)
print(agri_region.topo_scale_heights)
print(agri_region.chem_lifetimes)
# l3.plot(plot_field='wind_column_topo_chem',saving_path='temp1.png')

# Plot scale height and lifetime
months = pd.period_range(start='2019-09', end='2021-10', freq='M')
timestamps = months.to_timestamp()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
axes[0].plot(timestamps,agri_region.l3ms.df['topo_scale_height']) 
axes[0].set_ylabel('Scale height (m)')
axes[1].plot(timestamps,agri_region.l3ms.df['chem_lifetime'])
axes[1].set_ylabel('Lifetime (h)')
plt.tight_layout()
plt.show()
# plt.savefig('Xtao_region6.png')

# Spatial plot
df = agri_region.l3ds_df
filtered_df = df[df['date'] == '2020-04-01']
# plot monthly total emission of year 2020
months = pd.period_range(start='2020-01', end='2020-12', freq='M')
for month in months:    
    t = df[df['month']==str(month)]
    arrays = t['wind_column_topo_chem'].values[0:len(t['wind_column_topo_chem'])]
    list = arrays.tolist()
    t1 = np.nansum(list, axis=0)
    t1 = np.where(t1 == 0, np.nan, t1)

    # matrix = filtered_df['wind_column_topo_chem'].values[0]
    xgrid = filtered_df['xgrid'].iloc[0]
    ygrid = filtered_df['ygrid'].iloc[0]
    kwargs = {}
    kwargs['cmap'] = 'jet'
    kwargs['alpha'] = 1.
    kwargs['shrink'] = 0.75
    kwargs['vmin'] = np.nanmin(t1)
    kwargs['vmax'] = 3 #np.nanmax(t1)
    
    fig,ax = plt.subplots(1,1,figsize=(10,5),subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([min(filtered_df['xgrid'].iloc[0]), max(filtered_df['xgrid'].iloc[0]), min(filtered_df['ygrid'].iloc[0]), max(filtered_df['ygrid'].iloc[0])], ccrs.Geodetic())
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='k',linewidth=1)
    pc = ax.pcolormesh(*F_center2edge(xgrid,ygrid),t1,transform=ccrs.PlateCarree(),
                               alpha=kwargs['alpha'],cmap=kwargs['cmap'],vmin=kwargs['vmin'],vmax=kwargs['vmax'])
    cb = plt.colorbar(pc,ax=ax,label='wind_column_topo_chem',shrink=kwargs['shrink'])

    # plt.savefig()
    