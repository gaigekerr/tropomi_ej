#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module reads in  S5P/TROPOMI granules for given dates and interpolates
from the S5P time-varying grid to a fixed grid.

Todo:
    * Figure out time vs time_utc
    * Include netCDF4 attributes/description in output file
    * Output file name should state resolution, QA      COMPLETE - 5 July 2020
    * Convert to molec cm-2? (x 6.02214e+19)            COMPLETE - 5 July 2020
"""

# DDIR = '/mnt/scratch3/gaige/TROPOMI/'
# DDIR_OUT = '/mnt/scratch3/gaige/TROPOMI/'
DDIR = '/Users/ghkerr/GW/data/'
DDIR_OUT = '/Users/ghkerr/GW/data/'

def load_tropomi(startdate, enddate, fstr, var):
    """Read S5P/TROPOMI granules for a given range of dates and extract 
    variable of interest along with QA and dimensional information (i.e., 
    time, latitude, and longitude)

    Parameters
    ----------
    startdate : str
        Start date of time period of interest
    enddate : str
        End date of time period of interest
    fstr : str
        The prefix for the S5P/TROPOMI graunles that specifies version (Near 
        Real-Time (NRTI) or Offline (OFFL)), dataset, and level
    var : str
        Variable of interest that will be extracted from S5P/TROPOMI granules

    Returns
    -------
    outdict : dict
        Dictionary containing variable(s) of interest, QA, and dimensional
        information
    """
    import numpy as np
    import netCDF4 as nc
    from datetime import datetime
    import os
    import pandas as pd    
    # Change date to convention of TROPOMI files (i.e., YYYYMMDD)
    startdate = datetime.strptime(startdate,'%Y-%m-%d')
    enddate = datetime.strptime(enddate,'%Y-%m-%d')
    # Create date range 
    dates = []
    tmp = pd.date_range(startdate, enddate).tolist()
    for t in tmp: 
        dates.append(str(t.year)+str(t.month).zfill(2)+str(t.day).zfill(2))
    del tmp    
    # Find all TROPOMI files within temporal range of interest (must remove 
    # fstr, which specifies product/level/gas)
    files = [fn for fn in os.listdir(DDIR) if 
        any(fn.replace(fstr,'').startswith(prefix) for prefix in dates)]
    # Variables of interest will be extracted and stored in a dictionary
    outdict = {key:[] for key in ['qa_value', 'latitude', 'longitude', 
        'time', var]}
    for file in files:
        infile = nc.Dataset(DDIR+file,'r')
        # Grid changes with each granule, so this needs to be saved off at
        # every iteration
        lat = infile.groups['PRODUCT'].variables['latitude'][0].data
        # n.b., longitude is in units of -180 to 180 deg
        lng = infile.groups['PRODUCT'].variables['longitude'][0].data
        # Extract time; units are seconds since 2010-01-01
        time = infile.groups['PRODUCT'].variables['time'][:]
        time_unit = infile.groups['PRODUCT'].variables['time'].units
        try:
            time_cal = infile.groups['PRODUCT'].variables['time'].calendar
        except AttributeError: # Attribute doesn't exist
            time_cal = u'gregorian' # Or standard
        time = nc.num2date(time, units=time_unit, calendar=time_cal)[0]
        qa_value = infile.groups['PRODUCT'].variables['qa_value'][0].data
        # Extract variables of interest and append to dictionary containing
        # dimensional information
        vararr = infile.groups['PRODUCT'].variables[var][0].data
        fill_value = infile.groups['PRODUCT'].variables[var][0].fill_value
        vararr[vararr==fill_value] = np.nan
        outdict[var].append(vararr)
        outdict['latitude'].append(lat)
        outdict['longitude'].append(lng)
        outdict['time'].append(time)
        outdict['qa_value'].append(qa_value)
    return outdict
        
def apply_qa_value(tropomi, var, threshold):
    """Function searches data quality values (varies between 0 (no data) and 1 
    (full quality data)) for occurrences under the threshold of interest. 
    TROMPOMI retrievals at indices under the threshold are set to NaN

    Parameters
    ----------    
    tropomi : dict
        Dictionary containing TROPOMI/S5P variable(s) of interest, QA, and 
        dimensional information
    var : str
        Variable of interest that will be extracted from S5P/TROPOMI granules        
    threshold : float
        Data quality threshold below which data will be screen (recommended to
        ignore data with QA values < 0.5)

    Returns
    -------
    tropomi : dict
        Dictionary containing TROPOMI/S5P variable(s) of interest, QA, and 
        dimensional information screened for non-quality data    
    """
    import numpy as np
    # Loop through variables/time of interest 
    tmp = tropomi[var]
    for g in np.arange(0,len(tropomi['time']),1):
        tmp_g = tmp[g]
        qa_value_g = tropomi['qa_value'][g]
        # Set as NaN where QA array is less than threshold
        underthresh = np.where(qa_value_g < threshold)
        tmp_g[underthresh] = np.nan
        # Replace in dictionary 
        tropomi[var][g] = tmp_g
    return tropomi

def regrid(tropomi, var, crop, lonss, latss, nchunk):
    """Regrid TROPOMI data to a consistent rectlinear grid separated by a 
    constant spacing
    
    Parameters
    ----------    
    tropomi : dict 
        Dictionary containing TROPOMI/S5P variable(s) of interest, QA, and 
        dimensional information    
    var : str
        TROPOMI/S5P variable name (should be a key in variable "tropomi")
    crop : list
        Bounds (x0,x1,y0,y1) roughly corresponding to TROPOMI focus region
    lonss : float
        Longitude step size, i.e. grid resolution, units of degrees
    latss : float
        Latitude step size, i.e. grid resolution, units of degrees
    nchunk : int
        Number of chunks that the time-varying lat/lon grid for the satellite
        retrievals will be split into. Note that this function splits the 
        lat/lon grid by the y dimension since the satellite swaths are 
        predominantly N/S (if split on the x dimension, there will be instances
        where there are no retrivals in a chunk and xESMF will throw an error).

    Returns
    -------    
    interpolated : numpy.ndarray
        Interpolated TROPOMI retrievals, [time, lat, lon]
    lat_out : numpy.ndarray
         Rectilinear latitude spine with grid spacing given by variable 
         "latss," [lat,]
    lon_out : numpy.ndarray
         Rectilinear longitude spine with grid spacing given by variable 
         "lonss," [lon,]
    """
    from timeit import default_timer as timer
    from datetime import timedelta
    start = timer()
    import numpy as np
    import xarray as xr
    import xesmf as xe
    import cartopy.crs as ccrs
    interpolated = []
    def chunk_it(length, num):
        """For a given dimension size, get the start/stop indices of quasi equal-
        sized chunks. Adapted from stackoverflow.com/questions/2130016/splitting-
        a-list-into-n-parts-of-approximately-equal-length
        """
        import numpy as np
        avg = length / float(num)
        seq = np.arange(0,length,1)
        out = []
        last = 0.0
        while last < length:
            out.append(seq[int(last):int(last + avg)])
            last += avg
        return [(x[0], x[-1]+1) for x in out]  
    # Constract 2D rectilinear grid, cropped over region of interest so there
    # are no issues with buffers and to avoid creating large ESMF grid objects 
    # (see https://github.com/JiaweiZhuang/xESMF/issues/29)
    ds_out = xe.util.grid_2d(crop[0], crop[1], lonss, crop[2], crop[3], latss)
    ds_out = ds_out.drop(labels=['lon_b','lat_b'])
    # Extract latitude/longitude spines
    lat_out = np.nanmean(ds_out.lat.data, axis=1)
    lon_out = np.nanmean(ds_out.lon.data, axis=0)
    # Loop through files (each one representing a specific granule; see 
    # https://sentinel.esa.int/documents/247904/2474726/Sentinel-5P-Level-2-
    # Product-User-Manual-NPP-Cloud-product)
    for g in np.arange(0,len(tropomi['time']),1):
        var_g = tropomi[var][g]
        lat_g = tropomi['latitude'][g]
        lng_g = tropomi['longitude'][g]
        # List to fill with interpolated values for particular granule "g"
        interpolated_g = []
        # Make xarray dataset for variable/time of interest 
        ds = xr.Dataset({var: (['x', 'y'],  var_g)}, 
            coords={'lon': (['x', 'y'], lng_g), 'lat': (['x', 'y'], lat_g)})
        # Chunk up TROPOMI data so xESMF doesn't crash 
        for chunk in chunk_it(ds_out.y.shape[0], nchunk):
            ds_out_chunk = ds_out.isel(y=slice(chunk[0],chunk[-1]))
            # Only examine retrievals that are approximately within the 
            # chunk             
            ds_chunk = ds.where(
                (ds.lon > ds_out_chunk.lon.data.min()) & 
                (ds.lon < ds_out_chunk.lon.data.max()) & 
                (ds.lat > ds_out_chunk.lat.data.min()) & 
                (ds.lat < ds_out_chunk.lat.data.max()), drop=True)
            dr = ds_chunk[var]
            # Only perform regridding if there are points in chunk to regrid;
            # There are issues when there are chunked sections of the high 
            # resolution grid that don't overlap with any TROPOMI retrivals
            # (see https://github.com/JiaweiZhuang/xESMF/issues/36), if the 
            # shape of the retrieval within the chunk is (0,0). Then fill the 
            # output interpolated array with NaNs
            if ds_chunk[var].shape == (0,0):
                dr_out = np.empty(shape=(ds_out_chunk.y.shape[0],
                    ds_out_chunk.x.shape[0]))
                dr_out[:] = np.nan
                interpolated_g.append(dr_out)            
            else: 
                # Note that xESMF thinks that some grid cells (maybe near edges
                # of the domain) are triangles are instead of quadrilaters 
                # (i.e., degenerated); see https://github.com/JiaweiZhuang/xESMF/
                # issues/60                
                regridder = xe.Regridder(ds_chunk, ds_out_chunk, 'bilinear', 
                    ignore_degenerate=True)
                dr_out = regridder(dr)
                interpolated_g.append(dr_out.data)
                regridder.clean_weight_file()  # clean-up
        # There appears to be issues with cells that are outide the old grid's 
        # domain (see https://github.com/JiaweiZhuang/xESMF/issues/15) are all 
        # set to 0.0 when regridding rather than nan. Force any interpolated 
        # grid cell that is identically equal to 0.0 to be NaNs 
        interpolated_g = np.vstack(interpolated_g)
        interpolated_g[interpolated_g==0.0] = np.nan
        # # # # Optional: check regridding 
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,4))
        ax1 = plt.subplot2grid((2,2),(0,0),colspan=2, 
            projection=ccrs.PlateCarree())
        ax2 = plt.subplot2grid((2,2),(1,0),colspan=2, 
            projection=ccrs.PlateCarree())    
        mb1 = ax1.pcolormesh(lng_g, lat_g, var_g, cmap=plt.get_cmap(
            'gist_earth'), vmin=np.nanpercentile(interpolated_g, 20),
            vmax=np.nanpercentile(interpolated_g, 20))
        ax1.coastlines()
        plt.colorbar(mb1, ax=ax1)
        # dr.plot.pcolormesh(ax=ax1, x='lon', y='lat', levels=np.linspace(
        #     np.nanpercentile(dr.data, 20), np.nanpercentile(dr.data, 80), 
        #     10), cmap=plt.get_cmap('gist_earth'));
        mb2 = ax2.pcolormesh(lon_out, lat_out, interpolated_g, 
            cmap=plt.get_cmap('gist_earth'), vmin=
            np.nanpercentile(interpolated_g, 20), 
            vmax=np.nanpercentile(interpolated_g, 20))
        plt.colorbar(mb2, ax=ax2, extend='both')
        ax2.coastlines()
        for ax in [ax1, ax2]:
            ax.set_extent(crop)
        plt.show()
        # # # # 
        # Append interpolated grid to multi-granule list
        interpolated.append(interpolated_g)
    interpolated = np.stack(interpolated)
    end = timer()
    print('Interpolated in...', timedelta(seconds=end-start))
    return interpolated, lat_out, lon_out

def regrid_tropomi(startdate, enddate, crop, lonss, latss, nchunk, qa=0.75):
    """Read SP5/TROPOMI L2 OFFL NO2, conduct quality assurance on data, and 
    interpolate to a standard rectilinear grid. Each interpolated granuleand 
    the date on which it was retrieved is saved in the output file. 
    
    Parameters
    ----------    
    startdate : str
        Start date of time period of interest
    enddate : str
        End date of time period of interest
    crop : list
        Bounds (x0,x1,y0,y1) roughly corresponding to TROPOMI focus region
    lonss : float
        Longitude step size, i.e. grid resolution, units of degrees
    latss : float
        Latitude step size, i.e. grid resolution, units of degrees
    nchunk : int
        Number of chunks to split fine resolution grid into for regridding
        (if the resolution of output grid is, say, 1 deg, then this could be 
        1)
    qa : optional, float
        Data quality threshold below which data will be screen (recommended to
        ignore data with QA values < 0.5)
        
    Returns
    -------    
    None
    """
    import calendar
    import netCDF4 as nc
    from datetime import datetime
    fstr = 'S5P_OFFL_L2__NO2____'
    var = 'nitrogendioxide_tropospheric_column'
    # Load data for selected time period
    tropomi = load_tropomi(startdate, enddate, fstr, var)
    # Filter out bad/erroneous data
    tropomi = apply_qa_value(tropomi, var, qa)
    # Interpolate to rectilinear grid
    interpolated, lat, lon = regrid(tropomi, var, crop, lonss, latss, nchunk)
    # Create output file which specifies the version, constituent, operation, 
    # and start/end dates
    root_grp = nc.Dataset(DDIR_OUT+
        'S5P_NO2_%s-%s_%.2fgrid_QA%d.nc'%(
        datetime.strftime(datetime.strptime(startdate,'%Y-%m-%d'),'%Y%m%d'),
        datetime.strftime(datetime.strptime(enddate,'%Y-%m-%d'),'%Y%m%d'),
        latss, int(qa*100)), 'w', format='NETCDF4')
    root_grp.title = u'TROPOMI/S5P %s '%(var)
    root_grp.history = 'created %d %s %d by Gaige Kerr (gaigekerr@gwu.edu)'\
        %(datetime.today().day, calendar.month_abbr[datetime.today().month],
        datetime.today().year)
    root_grp.description = 'Data are for %s to %s '%(startdate,enddate)+\
        'and are filtered such that only qa_value > %.2f are included. '%(qa)+\
        'Data regridded to %.2f deg longitude x %.2f deg latitude grid'\
        %(lonss, latss)
    # Dimensions
    root_grp.createDimension('time', None)
    root_grp.createDimension('longitude', len(lon))
    root_grp.createDimension('latitude', len(lat))
    # Variables
    # Longitude
    var_lon = root_grp.createVariable('longitude', 'f4', ('longitude',))
    var_lon[:] = lon
    var_lon.long_name = 'longitude'
    var_lon.units = 'degrees east'
    # Latitude
    var_lat = root_grp.createVariable('latitude', 'f4', ('latitude',))
    var_lat[:] = lat
    var_lat.long_name = 'latitude'
    var_lat.units = 'degrees north'
    # TROPOMI extract
    var_out = root_grp.createVariable(var, 
        'f8', ('time', 'latitude', 'longitude',))
    var_out[:] = (interpolated*6.02214e+19)
    var_out.long_name = var
    var_out.units = 'molecules_percm2'
    # Time
    var_t = root_grp.createVariable('time', 'int32', ('time',))
    var_t.setncattr('units', 'seconds since 2010-01-01 00:00:00 UTC')
    ntime = nc.date2num(tropomi['time'], var_t.units)
    var_t[:] = ntime
    # Closing
    root_grp.close()
    return

startdate = '2019-04-01'
enddate = '2019-04-30'

crop = [-125, -65, 23, 52]
lonss = 0.01
latss = 0.01
nchunk = 4
regrid_tropomi(startdate, enddate, crop, lonss, latss, nchunk)

# import numpy as np
# fig = plt.figure()
# ax = plt.subplot2grid((1,1), (0,0))
# mb = ax.pcolormesh(tropomi['longitude'][0], tropomi['latitude'][0], 
#     tropomi['nitrogendioxide_tropospheric_column'][0], vmin=0., vmax=0.0001)
# plt.colorbar(mb)
# ax.set_xlim([-170,-110])
# ax.set_ylim([0, 75])





# import matplotlib.pyplot as plt
# import netCDF4 as nc
# import numpy as np
# import matplotlib as mpl
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# gg = nc.Dataset('/Users/ghkerr/Desktop/S5P_NO2_20190401-20190430_0.25grid_QA75.nc', 'r')
# no2 = gg.variables['nitrogendioxide_tropospheric_column'][:]
# lat = gg.variables['latitude'][:]
# lon = gg.variables['longitude'][:]
# clevs = np.linspace(0, 8e14, 9)
# fig = plt.figure()
# ax = plt.subplot2grid((1,2), (0,0), colspan=2,
#     projection=ccrs.PlateCarree(central_longitude=0.))
# ax.set_extent([-125,-65,25,50], crs=ccrs.PlateCarree())
# # Add coasts 
# ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='50m',
#     facecolor='none', name='coastline', lw=0.5, 
#     edgecolor='w'))
# # Add national borders
# ax.add_feature(cfeature.NaturalEarthFeature('cultural', 
#     'admin_0_countries', '50m', edgecolor='w', lw=0.5, 
#     facecolor='none'))
# # Add states
# ax.add_feature(cfeature.NaturalEarthFeature('cultural', 
#     'admin_1_states_provinces_lines', '50m', edgecolor='w', lw=0.5, 
#     facecolor='none'))
# # Add lakes
# ax.add_feature(cfeature.NaturalEarthFeature('physical', 
#     'lakes', '50m', edgecolor='w', lw=0.5, 
#     facecolor='none'))
# cmap = plt.get_cmap('Blues')
# cm = ax.contourf(lon, lat, np.nanmean(no2,axis=0), clevs, cmap=cmap, 
#     transform=ccrs.PlateCarree())
# plt.colorbar(cm, extend='max')
# plt.show()


# clevs = np.linspace(0, 8e15, 9)
# danavg = nc.Dataset('/Users/ghkerr/GW/data/'+
#     'Tropomi_NO2_griddedon0.01grid_042019_QA75.ncf', 'r')
# danavg_no2 = danavg.variables['NO2'][:]
# danavg_grid = nc.Dataset('/Users/ghkerr/GW/data/'+
#     'LatLonGrid.ncf', 'r')
# danavg_lng = danavg_grid.variables['LON'][:]
# danavg_lat = danavg_grid.variables['LAT'][:]

# fig = plt.figure()
# ax = plt.subplot2grid((1,2), (0,0), colspan=2,
#     projection=ccrs.PlateCarree(central_longitude=0.))
# ax.set_extent([-125,-65,25,50], crs=ccrs.PlateCarree())
# # Add coasts 
# ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='50m',
#     facecolor='none', name='coastline', lw=0.5, 
#     edgecolor='w'))
# # Add national borders
# ax.add_feature(cfeature.NaturalEarthFeature('cultural', 
#     'admin_0_countries', '50m', edgecolor='w', lw=0.5, 
#     facecolor='none'))
# # Add states
# ax.add_feature(cfeature.NaturalEarthFeature('cultural', 
#     'admin_1_states_provinces_lines', '50m', edgecolor='w', lw=0.5, 
#     facecolor='none'))
# # Add lakes
# ax.add_feature(cfeature.NaturalEarthFeature('physical', 
#     'lakes', '50m', edgecolor='w', lw=0.5, 
#     facecolor='none'))
# cm = ax.contourf(danavg_lng[::10], danavg_lat[::10], danavg_no2[::10,::10], 
#     clevs, cmap=cmap, transform=ccrs.PlateCarree())
# plt.colorbar(cm, extend='max')
