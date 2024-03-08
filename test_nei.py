from netCDF4 import Dataset
import sys, os, glob
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import logging
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.path import Path

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

class Inventory(dict):
    '''class based on dict, representing a gridded emission inventory'''
    def __init__(self,name=None,west=-180,east=180,south=-90,north=90):
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.west = west
        self.east = east
        self.south = south
        self.north = north
    
    def read_NEI_ag(self,monthly_filenames,field='NH3',unit='nmol/m2/s'):
        '''read a list of monthly NEI inventory files
        monthly_filenames:
            a list of file paths
        field:
            data field to read from the nc file
        unit:
            emission will be converted from kg/m2/s (double check) to this unit
        '''
        mons = []
        for i,filename in enumerate(monthly_filenames):
            nc = Dataset(filename)
            if i == 0:
                monthly_fields = np.zeros((nc.dimensions['lat'].size,
                                           nc.dimensions['lon'].size,
                                           len(monthly_filenames)))
                xgrid = nc['lon'][:].data
                ygrid = nc['lat'][:].data
                xgrid_size = np.abs(np.nanmedian(np.diff(xgrid)))
                ygrid_size = np.abs(np.nanmedian(np.diff(ygrid)))
                self.xgrid_size = xgrid_size
                self.ygrid_size = ygrid_size
                if not np.isclose(xgrid_size,ygrid_size,rtol=1e-03):
                    self.logger.warning(f'x grid size {xgrid_size} does not equal to y grid size {ygrid_size}')
                self.grid_size = (xgrid_size+ygrid_size)/2
                xmask = (xgrid >= self.west) & (xgrid <= self.east)
                ymask = (ygrid >= self.south) & (ygrid <= self.north)
                self['xgrid'] = xgrid[xmask]
                self['ygrid'] = ygrid[ymask]
                xmesh,ymesh = np.meshgrid(self['xgrid'],self['ygrid'])
                self['grid_size_in_m2'] = np.cos(np.deg2rad(ymesh/180*np.pi))*np.square(self.grid_size*111e3)
                nc_unit = nc[field].units
                if nc_unit == 'kg/m2/s' and unit=='nmol/m2/s' and field in ['NH3','NH3_FERT']:
                    self.logger.warning(f'unit of {field} will be converted from {nc_unit} to {unit}')
                    self[f'{field} unit'] = unit
                    unit_factor = 1e9/0.017
                else:
                    self.logger.info('no unit conversion is done')
                    self[f'{field} unit'] = nc_unit
                    unit_factor = 1.
            monthly_fields[:,:,i] = unit_factor*nc[field][:].filled(np.nan)[0,0,:,:]# time and lev are singular dimensions
            mons.append(dt.datetime(int(str(nc.SDATE)[0:4]),1,1)+dt.timedelta(days=-1+int(str(nc.SDATE)[-3:])))
            nc.close()
        self[field] = monthly_fields
        self['mons'] = pd.to_datetime(mons).to_period('1M')
        self['sum_NH3'] = np.sum(monthly_fields, axis=2) # zitong add: sum for 12 months
            
        return self
    
    def plot_all_months(self,field='NH3',nrow=1,ncol=1,figsize=(10,5),**kwargs):
        fig,axs = plt.subplots(nrow,ncol,figsize=figsize,constrained_layout=True,
                               subplot_kw={"projection": ccrs.PlateCarree()})
        if not hasattr(axs, '__iter__'):
            axs=[axs]
        for i,ax in enumerate(axs):
            pc = ax.pcolormesh(*F_center2edge(self['xgrid'],self['ygrid']),self[field][...,i],**kwargs)
            ax.coastlines(resolution='50m', color='black', linewidth=1)
            ax.add_feature(cfeature.STATES.with_scale('50m'), facecolor='None',
                           edgecolor='black', linewidth=1)
            ax.set_title(self['mons'][i].strftime('%Y%m'))
            fig.colorbar(pc,ax=ax,label=self[f'{field} unit'])
            plt.savefig('5.png')
    
    def plot_sum_months(self,field='sum_NH3',nrow=1,ncol=1,figsize=(10,5),**kwargs):
        # zitong add
        fig,ax = plt.subplots(nrow,ncol,figsize=figsize,constrained_layout=True,
                               subplot_kw={"projection": ccrs.PlateCarree()})
        pc = ax.pcolormesh(*F_center2edge(self['xgrid'],self['ygrid']),self[field],**kwargs)
        ax.coastlines(resolution='50m', color='black', linewidth=1)
        ax.add_feature(cfeature.STATES.with_scale('50m'), facecolor='None',
                        edgecolor='black', linewidth=1)
        ax.set_title(field)
        fig.colorbar(pc,ax=ax,label='nmol/m2/s')
        plt.savefig('6.png')

    def create_mask_from_sum_NH3(self, minNH3th):
        # zitong add: make the square boundary of mask E~0
        sumNH3 = self['sum_NH3']
        sumNH3 = sumNH3[sumNH3>0]
        print(np.percentile(sumNH3, minNH3th))
        mask = np.logical_and(self['sum_NH3']<np.percentile(sumNH3, minNH3th), self['sum_NH3']>0)    
        print('wow')
        print(np.sum(mask==True))    
        xyz = []
        for i in range(mask.shape[1]):
            for j in range(mask.shape[0]):
                if mask[j,i]:
                    # Calculate the bottom left corner of the square
                    bottom_left_lon = self['xgrid'][i] - self.xgrid_size / 2
                    bottom_left_lat = self['ygrid'][j] - self.ygrid_size / 2
                    # Define square boundaries (bottom_left, bottom_right, top_right, top_left)
                    square_boundaries = [
                        (bottom_left_lon, bottom_left_lat),
                        (bottom_left_lon + self.xgrid_size, bottom_left_lat),
                        (bottom_left_lon + self.xgrid_size, bottom_left_lat + self.ygrid_size),
                        (bottom_left_lon, bottom_left_lat + self.ygrid_size)
                    ]
                    xyz.append(square_boundaries)
        self['boundary_mask'] = xyz

# example to run the code
# your_path = '/projects/bbkc/zitong/Data/NEI_gridded'
# monthly_filenames = [os.path.join(your_path,f) for f in os.listdir(your_path) if os.path.isfile(os.path.join(your_path, f))]
# nei = Inventory().read_NEI_ag(monthly_filenames)
# plt.close('all')
# nei.plot_all_months(figsize=(10,8),nrow=6)
# nei.plot_sum_months(figsize=(10,8))
# nei.create_mask_from_sum_NH3(minNH3=10e-10)
# df = pd.DataFrame(nei['sum_NH3'])
# file_name = 'sum_NH3_2016.xlsx'
# df.to_excel(file_name, index=False)  
