#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module loads census tract-level demographic data (e.g., income, race, 
etc.) and pre- and post-lockdown NO2. The mean NO2 column densities averaged 
over each census tract in a state are found (both pre- and post-lockdowns) and 
combined with demographic information in a pandas dataframe that is output 
to a .csv file. 

Todo:
    Automate processing different states
"""
import math 
import numpy as np
# DIR_NO2 = '/Users/ghkerr/GW/data/'
# DIR_CENSUS = '/Users/ghkerr/GW/data/census_no2_harmonzied/'
# DIR_SHAPEFILE = '/Users/ghkerr/GW/data/geography/tigerline/'
# DIR_OUT = '/Users/ghkerr/GW/data/census_no2_harmonzied/'
DIR_NO2 = '/mnt/scratch1/gaige/data/tropomi_ej/'
DIR_CENSUS = '/mnt/scratch1/gaige/data/tropomi_ej/'
DIR_SHAPEFILE = '/mnt/scratch1/gaige/data/tropomi_ej/tigerline/'
DIR_OUT = '/mnt/scratch1/gaige/data/tropomi_ej/'

def geo_idx(dd, dd_array):
    """Function searches for nearest decimal degree in an array of decimal 
    degrees and returns the index. np.argmin returns the indices of minimum 
    value along an axis. So subtract dd from all values in dd_array, take 
    absolute value and find index of minimum. n.b. Function edited on 
    5 Dec 2018 to calculate the resolution using the mode. Before this it 
    used the simple difference which could be erroraneous because of 
    longitudes in (-180-180) coordinates (the "jump" from -180 to 180 yielded 
    a large resolution).
    
    Parameters
    ----------
    dd : int
        Latitude or longitude whose index in dd_array is being sought
    dd_array : numpy.ndarray 
        1D array of latitude or longitude 
    
    Returns
    -------
    geo_idx : int
        Index of latitude or longitude in dd_array that is closest in value to 
        dd
    """
    import numpy as np   
    from scipy import stats
    geo_idx = (np.abs(dd_array - dd)).argmin()
    # if distance from closest cell to intended value is 2x the value of the
    # spatial resolution, raise error 
    res = stats.mode(np.diff(dd_array))[0][0]
    if np.abs(dd_array[geo_idx] - dd) > (2 * res):
        print('Closet index far from intended value!')
        return 
    return geo_idx

def find_grid_in_bb(ingrid, lat, lng, left, right, down, up): 
    """Given a bounding box (i.e., coordinates of minimum and maximum latitudes
    and longitudes), reduce a given grid and the dimensional coordinates to 
    only that focus region.
    
    Parameters
    ----------
    ingrid : numpy.ndarray
        Input grid
    lat : numpy.ndarray
        Latitude coordinates, units of degrees north, [lat,]
    lng : numpy.ndarray
        Longitude coordinates, units of degrees east, [lng,]
    left : float
        Longitude coordinate of the left side (minimum) of the bounding box 
        containing the focus region, units of degrees east        
    right : float 
        Longitude coordinate of the right side (maximum) of the bounding box 
        containing the focus region, units of degrees east        
    down : float
        Latitude coordinate of the bottom side (minimum) of the bounding box 
        containing the focus region, units of degrees north            
    up : float 
        Latitude coordinate of the top side (maximum) of the bounding box 
        containing the focus region, units of degrees north    

    Returns
    -------
    outgrid : numpy.ndarray
        Output grid
    lat : numpy.ndarray
        Focus region latitude coordinates, units of degrees north, [lat,]
    lng : numpy.ndarray
        Focus region longitude coordinates, units of degrees east, [lng,]
    """
    import numpy as np
    # Find spines of focus region 
    left = geo_idx(left, lng)
    right = geo_idx(right, lng)
    up = geo_idx(up, lat)
    down = geo_idx(down, lat)
    # Reduce grid to only focus region
    lng_dim = np.where(np.array(ingrid.shape) == lng.shape[0])[0][0]
    lat_dim = np.where(np.array(ingrid.shape) == lat.shape[0])[0][0]
    # Eventually fix this section? Depending on whether the grid has a time
    # dimension or if lat/lng are reversed, indexing the grid will be handled
    # differently
    if (lat_dim == 2) and (lng_dim == 3):
        outgrid = ingrid[:, :, down:up+1, left:right+1]        
    elif (lat_dim == 1) and (lng_dim == 2):
        outgrid = ingrid[:, down:up+1, left:right+1]    
    elif (lat_dim == 0) and (lng_dim == 1):
        outgrid = ingrid[down:up+1, left:right+1]    
    elif (lat_dim == 1) and (lng_dim == 0):
        outgrid = ingrid[left:right+1, down:up+1]        
    else: 
        print('Dimensional information does not match!'+
              ' Cannot reduce grid.')    
    # Reduce lat/lng to focus region 
    lat = lat[down:up+1]
    lng = lng[left:right+1]
    return outgrid, lat, lng

def harvesine(lon1, lat1, lon2, lat2):
    """Distance calculation, degree to km (Haversine method)

    Parameters
    ----------
    lon1 : float
        Longitude of point A
    lat1 : float
        Latitude of point A
    lon2 : float
        Longitude of point B
    lat2 : float
        Latitude of point A

    Returns
    -------
    d : float
        Distance between points A and B, units of degrees
    """
    rad = math.pi / 180  # degree to radian
    R = 6378.1  # earth average radius at equador (km)
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = (math.sin(dlat / 2)) ** 2 + math.cos(lat1 * rad) * \
        math.cos(lat2 * rad) * (math.sin(dlon / 2)) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d

def idwr(x, y, z, xi, yi):
    """Inverse distance weighting for interpolating missing TROPOMI NO2 
    column densities for census tracts that are too small to intersect with
    the regridded NO2 fields.

    Parameters
    ----------
    x : list
        Longitude
    y : list
        Latitudes
    z : list
        NO2 column densities
    xi : list
        Unknown longitude
    yi : TYPE
        Unknown latitude

    Returns
    -------
    lstxyzi : list
        List comprised of unknown longitude, latitude, and interpolated 
        TROPOMI column density for small census tract 
    """
    lstxyzi = []
    for p in range(len(xi)):
        lstdist = []
        for s in range(len(x)):
            d = (harvesine(x[s], y[s], xi[p], yi[p]))
            lstdist.append(d)
        sumsup = list((1 / np.power(lstdist, 2)))
        suminf = np.sum(sumsup)
        sumsup = np.sum(np.array(sumsup) * np.array(z))
        u = sumsup / suminf
        xyzi = [xi[p], yi[p], u]
        lstxyzi.append(xyzi)
    return lstxyzi

def harmonize_tropomino2_census(sFIPS, censusfile, checkplot=False):
    """For a particular state, function extracts census tracts and determines
    the pre- and post-COVID lockdown NO2 retrievals and demographic information
    within the tract. Note that the variable "var_extract" should be 
    changed depending on which census variables are desired (e.g., race, 
    income, age).

    Parameters
    ----------
    sFIPS : str
        Two digit numeric code defined in U.S. Federal Information Processing 
        Standard Publication to identify U.S. states. The TROPOMI NO2 and 
        census data will be extract for the state of the FIPS code supplied to
        function. For a list of FIPS codes, see: 
        https://en.wikipedia.org/wiki/
        Federal_Information_Processing_Standard_state_code
    censusfile : str
        Name of file containing NHGIS-created file. Name of file appears to 
        be related to the number of files downloaded from NHGIS (nhgisXXXX, 
        where XXXX is the number of files downloaded on account). The key to 
        the unique, NHGIS-created column names is found in the Codebook file(s) 
        automatically downloaded with data extract. See: 
        https://www.nhgis.org/support/faq#what_do_field_names_mean
    checkplot : bool
        If true, a multi-figure PDF will be output with maps of each extracted
        variable at the tract-level and gridded NO2 and tract-level NO2.

    Returns
    -------        
    None
    """
    import time
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pandas as pd
    import numpy as np
    import netCDF4 as nc
    import cartopy.io.shapereader as shpreader
    import cartopy.crs as ccrs
    import shapefile
    from shapely.geometry import shape, Point
    # Read in census tract information from https://www.nhgis.org
    # Name of file containing NHGIS-created file. Name of file appears to 
    # be related to the number of files downloaded from NHGIS (nhgisXXXX, 
    # where XXXX is the number of files downloaded on account). The key to 
    # the unique, NHGIS-created column names is found in the Codebook file(s) 
    # automatically downloaded with data extract. See: 
    # https://www.nhgis.org/support/faq#what_do_field_names_mean
    #----------------------
    nhgis = pd.read_csv(DIR_CENSUS+censusfile, delimiter=',', header=0, 
        engine='python')
    # Variables to be extracted from census tract file (see .txt file for 
    # codebook for NHGIS data file). This part can be modified to extract
    # additional variables from NHGIS file
    var_extract = ['AJWBE001', # Total population
        'AJZAE001', # Median household income in the past 12 months (in 2018 
        # inflation-adjusted dollars)
        'AJYPE001', # Total
        'AJYPE002', # No schooling completed
        'AJYPE003', # Nursery school
        'AJYPE004', # Kindergarten
        'AJYPE005', # 1st grade
        'AJYPE006', # 2nd grade
        'AJYPE007', # 3rd grade
        'AJYPE008', # 4th grade
        'AJYPE009', # 5th grade
        'AJYPE010', # 6th grade
        'AJYPE011', # 7th grade
        'AJYPE012', # 8th grade
        'AJYPE013', # 9th grade
        'AJYPE014', # 10th grade
        'AJYPE015', # 11th grade
        'AJYPE016', # 12th grade, no diploma
        'AJYPE017', # Regular high school diploma
        'AJYPE018', # GED or alternative credential
        'AJYPE019', # Some college, less than 1 year
        'AJYPE020', # Some college, 1 or more years, no degree
        'AJYPE021', # Associate's degree
        'AJYPE022', # Bachelor's degree
        'AJYPE023', # Master's degree
        'AJYPE024', # Professional school degree
        'AJYPE025', # Doctorate degree
        'AJWNE001', # Total
        'AJWNE002', # White alone
        'AJWNE003', # Black or African American alone
        'AJWNE004', # American Indian and Alaska Native alone
        'AJWNE005', # Asian alone
        'AJWNE006', # Native Hawaiian and Other Pacific Islander alone
        'AJWNE007', # Some other race alone
        'AJWNE008', # Two or more races
        'AJWNE009', # Two or more races: Two races including Some other race
        'AJWNE010', # Two or more races: Two races excluding S
        'AJWWE001', # Total
        'AJWWE002', # Not Hispanic or Latino
        'AJWWE003' # Hispanic or Latino
        # # Note that not all insurance-related variables are needed for the 
        # # insurance as some further subdivide insurance holders into how many/
        # # the types of insurance they have
        # 'AJ35E001', # Total
        # 'AJ35E003', # Under 19 years: With one type of health insurance 
        # # coverage
        # 'AJ35E010', # Under 19 years: With two or more types of health 
        # # insurance coverage
        # 'AJ35E017', # Under 19 years: No health insurance coverage
        # 'AJ35E019', # 19 to 34 years: With one type of health insurance     
        # # coverage
        # 'AJ35E026', # 19 to 34 years: With two or more types of health 
        # # insurance coverage
        # 'AJ35E033', # 19 to 34 years: No health insurance coverage
        # 'AJ35E035', # 35 to 64 years: With one type of health insurance 
        # # coverage
        # 'AJ35E042', # 35 to 64 years: With two or more types of health 
        # # insurance coverage
        # 'AJ35E050', # 35 to 64 years: No health insurance coverage
        # 'AJ35E052', # 65 years and over: With one type of health insurance 
        # # coverage
        # 'AJ35E058', # 65 years and over: With two or more types of health 
        # # insurance coverage
        # 'AJ35E066', # 65 years and over: No health insurance coverage
        # 'AJWVE001', # Total
        # 'AJWVE002', # Not Hispanic or Latino
        # 'AJWVE003', # Not Hispanic or Latino: White alone
        # 'AJWVE004', # Not Hispanic or Latino: Black or African American 
        # # alone
        # 'AJWVE005', # Not Hispanic or Latino: American Indian and Alaska 
        # # Native alone
        # 'AJWVE006', # Not Hispanic or Latino: Asian alone
        # 'AJWVE007', # Not Hispanic or Latino: Native Hawaiian and Other 
        # # Pacific Islander alone
        # 'AJWVE008', # Not Hispanic or Latino: Some other race alone
        # 'AJWVE009', # Not Hispanic or Latino: Two or more races
        # 'AJWVE012', # Hispanic or Latino
        # 'AJWVE013', # Hispanic or Latino: White alone
        # 'AJWVE014', # Hispanic or Latino: Black or African American alone
        # 'AJWVE015', # Hispanic or Latino: American Indian and Alaska Native 
        # # alone
        # 'AJWVE016', # Hispanic or Latino: Asian alone
        # 'AJWVE017', # Hispanic or Latino: Native Hawaiian and Other Pacific 
        # # Islander alone
        # 'AJWVE018', # Hispanic or Latino: Some other race alone
        # 'AJWVE019' # Hispanic or Latino: Two or more races
        # 'AJWCE001', # Median age: Total
        # 'AJWBE002', # Male
        # 'AJWBE026', # Female
        # 'AJWCE002', # Median age: Male
        # 'AJWCE003', # Median age: Female        
        # 'AJ0EE001', # Per capita income in the past 12 months (in 2018 
        # # inflation-adjusted dollars)
        # 'AJZVE001', # Total
        # 'AJZVE002', # With cash public assistance or Food Stamps/SNAP
        # 'AJZVE003', # No cash public assistance or Food Stamps/SNAP        
        ]
    print('# # # # # NHGIS census tract-level information read! # # # # #')
    
    # Read in census tract boundaries      
    #----------------------
    # How to download: https://www.census.gov/cgi-bin/geo/shapefiles/
    # index.php?year=2019&layergroup=Census+Tracts
    # Information about TIGER/Line shapefiles: https://www2.census.gov/geo/
    # pdfs/maps-data/data/tiger/tgrshp2019/TGRSHP2019_TechDoc.pdf
    fname = DIR_SHAPEFILE+'tl_2019_%s_tract/tl_2019_%s_tract.shp'%(sFIPS,sFIPS)
    r = shapefile.Reader(fname)
    # Get shapes, records
    tracts = r.shapes()
    records = r.records()
    print('# # # # # Census tracts for FIPS %s read! # # # # #'%sFIPS)    
    
    # Read in gridded pre-/post-lock down NO2 values
    #----------------------
    preNO2 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_Mar13-Jun132019_precovid19_QA75.ncf')
    postNO2 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_Mar13-Jun132020_postcovid19_QA75.ncf')
    preNO2apr = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_Apr01-Jun302019_precovid19_QA75.ncf')
    postNO2apr = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_Apr01-Jun302020_postcovid19_QA75.ncf') 
    preNO2marsep = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_Mar13-Sep132019_precovid19_QA75.ncf')
    postNO2marsep = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_Mar13-Sep132020_postcovid19_QA75.ncf') 
    allNO2 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_allyears_QA75.ncf')
    preNO2 = preNO2.variables['NO2'][:]
    postNO2 = postNO2.variables['NO2'][:]    
    preNO2apr = preNO2apr['NO2'][:]
    postNO2apr = postNO2apr['NO2'][:]
    preNO2marsep = preNO2marsep['NO2'][:]
    postNO2marsep = postNO2marsep['NO2'][:]
    allNO2 = allNO2['NO2'][:]
    # Dimensional information (lat/lon)
    grid = nc.Dataset(DIR_NO2+'LatLonGrid.ncf')
    lng_full = grid.variables['LON'][:]
    lat_full = grid.variables['LAT'][:]
    
    # Select ~state from gridded NO2 retrievals 
    #----------------------
    # Find min/max of each shapes in file for subsetting NO2 grids
    max_lats, min_lats, max_lngs, min_lngs = [],[], [], []
    for t in tracts:
        # Longitudes
        max_lngs.append(np.nanmax(np.array(t.points)[:,0]))
        min_lngs.append(np.nanmin(np.array(t.points)[:,0]))
        # Latitude
        max_lats.append(np.nanmax(np.array(t.points)[:,1]))
        min_lats.append(np.nanmin(np.array(t.points)[:,1]))
    # Find global min/max
    max_lats = np.nanmax(max_lats)
    min_lats = np.nanmin(min_lats)
    max_lngs = np.nanmax(max_lngs)
    min_lngs = np.nanmin(min_lngs)
    # MN and ME (and maybe OR) break the code
    if (max_lats+0.25 > lat_full[-1]):
        preNO2, lat, lng = find_grid_in_bb(preNO2, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, lat_full[-1])
        postNO2, lat, lng = find_grid_in_bb(postNO2, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, lat_full[-1])
        preNO2apr, lat, lng = find_grid_in_bb(preNO2apr, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, lat_full[-1])
        postNO2apr, lat, lng = find_grid_in_bb(postNO2apr, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, lat_full[-1])
        preNO2marsep, lat, lng = find_grid_in_bb(preNO2marsep, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, lat_full[-1])
        postNO2marsep, lat, lng = find_grid_in_bb(postNO2marsep, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, lat_full[-1])
        allNO2, lat, lng = find_grid_in_bb(allNO2, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, lat_full[-1])
    elif (min_lats-0.25 < lat_full[0]):
        preNO2, lat, lng = find_grid_in_bb(preNO2, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        postNO2, lat, lng = find_grid_in_bb(postNO2, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        preNO2apr, lat, lng = find_grid_in_bb(preNO2apr, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        postNO2apr, lat, lng = find_grid_in_bb(postNO2apr, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        preNO2marsep, lat, lng = find_grid_in_bb(preNO2marsep, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        postNO2marsep, lat, lng = find_grid_in_bb(postNO2marsep, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        allNO2, lat, lng = find_grid_in_bb(allNO2, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
    elif (max_lngs+0.25 > lng_full[-1]):
        preNO2, lat, lng = find_grid_in_bb(preNO2, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        postNO2, lat, lng = find_grid_in_bb(postNO2, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        preNO2apr, lat, lng = find_grid_in_bb(preNO2apr, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        postNO2apr, lat, lng = find_grid_in_bb(postNO2apr, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        preNO2marsep, lat, lng = find_grid_in_bb(preNO2marsep, lat_full, 
            lng_full, min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        postNO2marsep, lat, lng = find_grid_in_bb(postNO2marsep, lat_full, 
            lng_full, min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)        
        allNO2, lat, lng = find_grid_in_bb(allNO2, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
    elif (min_lngs-0.25 < lng_full[0]):
        preNO2, lat, lng = find_grid_in_bb(preNO2, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        postNO2, lat, lng = find_grid_in_bb(postNO2, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        preNO2apr, lat, lng = find_grid_in_bb(preNO2apr, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        postNO2apr, lat, lng = find_grid_in_bb(postNO2apr, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        preNO2marsep, lat, lng = find_grid_in_bb(preNO2marsep, lat_full, 
            lng_full, lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        postNO2marsep, lat, lng = find_grid_in_bb(postNO2marsep, lat_full, 
            lng_full, lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)        
        allNO2, lat, lng = find_grid_in_bb(allNO2, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)        
    else:    
        preNO2, lat, lng = find_grid_in_bb(preNO2, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        postNO2, lat, lng = find_grid_in_bb(postNO2, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        preNO2apr, lat, lng = find_grid_in_bb(preNO2apr, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        postNO2apr, lat, lng = find_grid_in_bb(postNO2apr, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        preNO2marsep, lat, lng = find_grid_in_bb(preNO2marsep, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        postNO2marsep, lat, lng = find_grid_in_bb(postNO2marsep, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        allNO2, lat, lng = find_grid_in_bb(allNO2, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, max_lats+0.25)        
    print('# # # # # TROPOMI NO2 files read! # # # # #')
    time.sleep(2)
    
    # Loop through shapes (each shape corresponds a census tract) append
    # GEOIDs, NO2 concentrations, and demographic information to list
    #----------------------
    df = []
    for tract in tqdm(np.arange(0, len(tracts), 1), leave=True):
        # Extract metadata (e.g., STATEFP|State FIPS Code, 
        # COUNTYFP|County FIPS Code, GEOID|Geographic Identifier, 
        # NAMELSAD|Full Name)
        record = records[tract]
        # Build a shapely polygon from shape
        tract = shape(tracts[tract]) 
        # Extract GEOID of record
        geoid = record['GEOID']
        # Find grid cells within polygon for NO2 in polygon 
        i_inside, j_inside = [], []
        for i, ilat in enumerate(lat):
            for j, jlng in enumerate(lng): 
                point = Point(jlng, ilat)
                if tract.contains(point) is True:
                    # Fill lists with indices in grid within polygon
                    i_inside.append(i)
                    j_inside.append(j)
        # Find census tract information from NHGIS by "converting" GEOID 
        # to GISJOIN identifier 
        # GISJOIN identifiers match the identifiers used in NHGIS data 
        # tables and boundary files. A block GISJOIN concatenates these codes:
        #    "G" prefix         This prevents applications from automatically 
        #                       reading the identifier as a number and, in 
        #                       effect, dropping important leading zeros
        #    State NHGIS code	3 digits (FIPS + "0"). NHGIS adds a zero to 
        #                       state FIPS codes to differentiate current 
        #                       states from historical territories.
        #    County NHGIS code	4 digits (FIPS + "0"). NHGIS adds a zero to 
        #                       county FIPS codes to differentiate current 
        #                       counties from historical counties.
        #    Census tract code	6 digits for 2000 and 2010 tracts. 1990 tract 
        #                       codes use either 4 or 6 digits.
        #    Census block code	4 digits for 2000 and 2010 blocks. 1990 block
        #                       codes use either 3 or 4 digits.
        # GEOID identifiers correspond to the codes used in most current 
        # Census sources (American FactFinder, TIGER/Line, Relationship Files, 
        # etc.). A block GEOID concatenates these codes:
        #    State FIPS code	2 digits
        #    County FIPS code	3 digits
        #    Census tract code	6 digits. 1990 tract codes that were 
        #                       originally 4 digits (as in NHGIS files) are 
        #                       extended to 6 with an appended "00" (as in 
        #                       Census Relationship Files).
        #    Census block code	4 digits for 2000 and 2010 blocks. 1990 block 
        #                       codes use either 3 or 4 digits.
        geoid_converted = 'G'+geoid[:2]+'0'+geoid[2:5]+'0'+geoid[5:]
        tract_nhgis = nhgis.loc[nhgis['GISJOIN']==geoid_converted]
        # If there are no TROPOMI grid cells within census tract, fill row 
        # corresponding to tract with NaNs for all variables
        if (len(i_inside)==0) or (len(j_inside)==0):
            # # # # Option 1: Treat small tracts as missing data/NaN for NO2
            # dicttemp = {'GEOID':geoid, 'PRENO2':np.nan, 'POSTNO2':np.nan, 
            #     'PRENO2APR':np.nan, 'POSTNO2APR':np.nan, 
            #     'PRENO2MAR-SEP':np.nan, 'POSTNO2MAR-SEP':np.nan, 
            #     'ALLNO2':np.nan}
            # for var in var_extract:
            #     dicttemp[var] = tract_nhgis[var].values[0]        
            # df.append(dicttemp)   
            # del dicttemp
            # # # # Option 2: Interpolate using inverse distance weighting 
            # https://rafatieppo.github.io/post/2018_07_27_idw2pyr/
            tract_latcentroid = tract.centroid.y
            tract_lngcentroid = tract.centroid.x           
            idx_latnear = geo_idx(tract_latcentroid, lat)
            idx_lngnear = geo_idx(tract_lngcentroid, lng)
            # Indices for 8 nearby points
            lng_idx = [idx_lngnear-1, idx_lngnear, idx_lngnear+1, 
                idx_lngnear-1, idx_lngnear+1, idx_lngnear-1, idx_lngnear, 
                idx_lngnear+1]
            lat_idx = [idx_latnear+1, idx_latnear+1, idx_latnear+1, 
                idx_latnear, idx_latnear, idx_latnear-1, idx_latnear-1, 
                idx_latnear-1]
            # Known coordinates
            x = lng[lng_idx].data
            y = lat[lat_idx].data
            # Known NO2 values 
            z_preNO2 = preNO2[lat_idx,lng_idx].data.flatten()
            z_postNO2 = postNO2[lat_idx,lng_idx]
            z_preNO2apr = preNO2apr[lat_idx,lng_idx]
            z_postNO2apr = postNO2apr[lat_idx,lng_idx]
            z_preNO2marsep = preNO2marsep[lat_idx,lng_idx]
            z_postNO2marsep = postNO2marsep[lat_idx,lng_idx]
            z_allNO2 = allNO2[lat_idx,lng_idx]
            dicttemp = {'GEOID':geoid, 
                'PRENO2':idwr(x,y,z_preNO2,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'POSTNO2':idwr(x,y,z_postNO2,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1], 
                'PRENO2APR':idwr(x,y,z_preNO2apr,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1], 
                'POSTNO2APR':idwr(x,y,z_postNO2apr,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1], 
                'PRENO2MAR-SEP':idwr(x,y,z_preNO2marsep,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1], 
                'POSTNO2MAR-SEP':idwr(x,y,z_postNO2marsep,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1], 
                'ALLNO2':idwr(x,y,z_allNO2,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1]}
            for var in var_extract:
                dicttemp[var] = tract_nhgis[var].values[0]        
            df.append(dicttemp)   
            del dicttemp   
        # For census tracts with TROPOMI grid cells, find grid cell-averaged
        # NO2 and demographic information from NHGIS census files
        else: 
            preNO2_inside = np.nanmean(preNO2[i_inside, j_inside])
            postNO2_inside = np.nanmean(postNO2[i_inside, j_inside])
            preNO2apr_inside = np.nanmean(preNO2apr[i_inside, j_inside])
            postNO2apr_inside = np.nanmean(postNO2apr[i_inside, j_inside])
            preNO2marsep_inside = np.nanmean(preNO2marsep[i_inside, j_inside])
            postNO2marsep_inside = np.nanmean(postNO2marsep[i_inside, j_inside])            
            allNO2_inside = np.nanmean(allNO2[i_inside, j_inside])
            dicttemp = {'GEOID':geoid, 'PRENO2':preNO2_inside, 
                'POSTNO2':postNO2_inside, 'PRENO2APR':preNO2apr_inside,
                'POSTNO2APR':postNO2apr_inside, 
                'PRENO2MAR-SEP':preNO2marsep_inside,
                'POSTNO2MAR-SEP':postNO2marsep_inside, 'ALLNO2':allNO2_inside
                }
            for var in var_extract:
                dicttemp[var] = tract_nhgis[var].values[0]
            df.append(dicttemp)               
            del dicttemp
    df = pd.DataFrame(df)
    # Set GEOID as column 
    df = df.set_index('GEOID')
    # Add FIPS code as column
    df.insert(1, 'FIPS', sFIPS)    
    print('# # # # # TROPOMI/census harmonized at tract level! # # # # #')
    
    # Save DataFrame
    #----------------------
    df = df.replace('NaN', '', regex=True)
    df.to_csv(DIR_OUT+'Tropomi_NO2_interpolated_%s_%s'%(sFIPS,censusfile), 
        sep = ',')
    print('# # # # # Output file written! # # # # #')  
    
    # If desired, maps of all variables will be plotted 
    #----------------------
    if checkplot==True:
        import matplotlib.backends.backend_pdf
        plt.rcParams.update({'figure.max_open_warning': 0})
        tracts = list(shpreader.Reader(fname).geometries())
        records = list(shpreader.Reader(fname).records())
        # Define coordinate reference system; all Census Bureau generated 
        # shapefiles are in Global Coordinate System North American Datum of 
        # 1983 (GCS NAD83). Each .prj file contains the following:
        # GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",
        # SPHEROID["GRS_1980",6 378137,298.257222101]],PRIMEM["Greenwich",0],
        # UNIT["Degree",0.017453292519943295]]
        # From https://www2.census.gov/geo/pdfs/maps-data/data/tiger/tgrshp2017/
        # TGRSHP2017_TechDoc_Ch2.pdf
        proj = ccrs.PlateCarree(globe=ccrs.Globe(datum='NAD83'))
        # Create figure, output file, and subplot
        pdf = matplotlib.backends.backend_pdf.PdfPages(DIR_OUT+
            'plotcheck_%s.pdf'%sFIPS)
        # Normalize colormap, see https://stackoverflow.com/questions/
        # 58379841/continuous-color-map-on-cartopy
        for var in var_extract:
            f, ax = plt.subplots()
            f.set_size_inches([10, 6])
            ax = plt.axes(projection=proj)
            ax.set_title(var, fontsize=16)
            norm = plt.Normalize(np.nanpercentile(df[var].values, 10), 
                np.nanpercentile(df[var].values, 90))
            cmap = plt.get_cmap('Blues')
            for tract, record in zip(tracts, records):
                gi = record.attributes['GEOID']
                var_tract = df.loc[df.index==gi]
                ax.add_geometries(tract, proj, edgecolor='k', facecolor=cmap(
                    norm(var_tract[var].values[0])), alpha=1.)        
            bnd = np.array([i.bounds for i in tracts])
            x0,x1 = np.min(bnd[:,0]),np.max(bnd[:,2])
            y0,y1 = np.min(bnd[:,1]),np.max(bnd[:,3])
            ax.set_extent([x0,x1,y0,y1],proj)    
            # Colorbar
            # sm = plt.cm.ScalarMappable(cmap=cmap)
            # sm._A = []
            # plt.colorbar(sm)
            cax = f.add_axes([0.85, 0.1, 0.0225, 0.775])
            cb = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=norm, 
                spacing='proportional', extend='both')
            pdf.savefig(f, dpi=200)
        # Plot gridded NO2
        f, ax = plt.subplots()
        f.set_size_inches([10, 6])
        ax = plt.axes(projection=proj)
        norm = plt.Normalize(np.nanpercentile(df.POSTNO2-df.PRENO2, 10), 
            np.nanpercentile(df.POSTNO2-df.PRENO2, 90))        
        ax.set_title('NO$_{2}$ (0.01$^{\circ}$ x 0.01$^{\circ}$)', fontsize=16)        
        cmap = plt.get_cmap('PuBu_r', 10)
        mb = ax.pcolormesh(lng, lat, postNO2-preNO2, cmap=cmap, vmin=
            np.nanpercentile(df.POSTNO2-df.PRENO2, 10), vmax=np.nanpercentile(
            df.POSTNO2-df.PRENO2, 90), transform=proj, alpha=1.)
        cb = plt.colorbar(mb, extend='both', label='molec cm$^{-2}$')
        ax.add_geometries(tracts, proj, edgecolor='k', facecolor='None', 
            alpha=1.)
        bnd = np.array([i.bounds for i in tracts])
        x0,x1 = np.min(bnd[:,0]),np.max(bnd[:,2])
        y0,y1 = np.min(bnd[:,1]),np.max(bnd[:,3])
        ax.set_extent([x0,x1,y0,y1],proj)
        pdf.savefig(f, dpi=200)    
        # Plot tract-level NO2 
        f, ax = plt.subplots()
        f.set_size_inches([10, 6])
        ax = plt.axes(projection=proj)
        ax.set_title('NO$_{2}$ (tract-level)', fontsize=16)
        for tract, record in zip(tracts, records):
            # Find GEOID of tract
            gi = record.attributes['GEOID']
            # Look up NO2 change for tract
            dno2 =(df.loc[df.index==gi].POSTNO2-df.loc[df.index==gi].PRENO2)
            dno2 = dno2.values[0]
            if np.isnan(dno2)==True:
                ax.add_geometries(tract, proj, edgecolor='k', facecolor='grey', 
                    alpha=1.)        
            else:
                ax.add_geometries(tract, proj, edgecolor='k', facecolor=cmap(
                    norm(dno2)), alpha=1.)
        bnd = np.array([i.bounds for i in tracts])
        x0,x1 = np.min(bnd[:,0]),np.max(bnd[:,2])
        y0,y1 = np.min(bnd[:,1]),np.max(bnd[:,3])
        ax.set_extent([x0,x1,y0,y1],proj)    
        cax = f.add_axes(cb.ax.get_position())
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, 
            spacing='proportional', extend='both', label='molec cm$^{-2}$')
        pdf.savefig(f, dpi=200)
        pdf.close()
    print('# # # # # checkplotfile saved! # # # # #')  
    return

def harmonize_tropomino2_census_metsens(sFIPS, censusfile):
    """Same as function "harmonize_tropomino2_census" but to harmonize 
    census demographics with baseline periods of various lengths (7 day, 
    14 day, 31 day, etc.)

    Parameters
    ----------
    sFIPS : str
        Two digit numeric code defined in U.S. Federal Information Processing 
        Standard Publication to identify U.S. states. The TROPOMI NO2 and 
        census data will be extract for the state of the FIPS code supplied to
        function. For a list of FIPS codes, see: 
        https://en.wikipedia.org/wiki/
        Federal_Information_Processing_Standard_state_code
    censusfile : str
        Name of file containing NHGIS-created file. Name of file appears to 
        be related to the number of files downloaded from NHGIS (nhgisXXXX, 
        where XXXX is the number of files downloaded on account). The key to 
        the unique, NHGIS-created column names is found in the Codebook file(s) 
        automatically downloaded with data extract. See: 
        https://www.nhgis.org/support/faq#what_do_field_names_mean

    Returns
    -------        
    None
    """
    import time
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    import netCDF4 as nc
    import shapefile
    from shapely.geometry import shape, Point
    # Read in census tract information from https://www.nhgis.org
    # Name of file containing NHGIS-created file. Name of file appears to 
    # be related to the number of files downloaded from NHGIS (nhgisXXXX, 
    # where XXXX is the number of files downloaded on account). The key to 
    # the unique, NHGIS-created column names is found in the Codebook file(s) 
    # automatically downloaded with data extract. See: 
    # https://www.nhgis.org/support/faq#what_do_field_names_mean
    #----------------------
    nhgis = pd.read_csv(DIR_CENSUS+censusfile, delimiter=',', header=0, 
        engine='python')
    # Variables to be extracted from census tract file (see .txt file for 
    # codebook for NHGIS data file). This part can be modified to extract
    # additional variables from NHGIS file
    var_extract = ['AJWBE001', # Total population
        'AJZAE001', # Median household income in the past 12 months (in 2018 
        # inflation-adjusted dollars)
        'AJYPE001', # Total
        'AJYPE002', # No schooling completed
        'AJYPE003', # Nursery school
        'AJYPE004', # Kindergarten
        'AJYPE005', # 1st grade
        'AJYPE006', # 2nd grade
        'AJYPE007', # 3rd grade
        'AJYPE008', # 4th grade
        'AJYPE009', # 5th grade
        'AJYPE010', # 6th grade
        'AJYPE011', # 7th grade
        'AJYPE012', # 8th grade
        'AJYPE013', # 9th grade
        'AJYPE014', # 10th grade
        'AJYPE015', # 11th grade
        'AJYPE016', # 12th grade, no diploma
        'AJYPE017', # Regular high school diploma
        'AJYPE018', # GED or alternative credential
        'AJYPE019', # Some college, less than 1 year
        'AJYPE020', # Some college, 1 or more years, no degree
        'AJYPE021', # Associate's degree
        'AJYPE022', # Bachelor's degree
        'AJYPE023', # Master's degree
        'AJYPE024', # Professional school degree
        'AJYPE025', # Doctorate degree
        'AJWNE001', # Total
        'AJWNE002', # White alone
        'AJWNE003', # Black or African American alone
        'AJWNE004', # American Indian and Alaska Native alone
        'AJWNE005', # Asian alone
        'AJWNE006', # Native Hawaiian and Other Pacific Islander alone
        'AJWNE007', # Some other race alone
        'AJWNE008', # Two or more races
        'AJWNE009', # Two or more races: Two races including Some other race
        'AJWNE010', # Two or more races: Two races excluding S
        'AJWWE001', # Total
        'AJWWE002', # Not Hispanic or Latino
        'AJWWE003' # Hispanic or Latino
        ]
    print('# # # # # NHGIS census tract-level information read! # # # # #')
    # Read in census tract boundaries      
    #----------------------
    # How to download: https://www.census.gov/cgi-bin/geo/shapefiles/
    # index.php?year=2019&layergroup=Census+Tracts
    # Information about TIGER/Line shapefiles: https://www2.census.gov/geo/
    # pdfs/maps-data/data/tiger/tgrshp2019/TGRSHP2019_TechDoc.pdf
    fname = DIR_SHAPEFILE+'tl_2019_%s_tract/tl_2019_%s_tract.shp'%(sFIPS,sFIPS)
    r = shapefile.Reader(fname)
    # Get shapes, records
    tracts = r.shapes()
    records = r.records()
    print('# # # # # Census tracts for FIPS %s read! # # # # #'%sFIPS)    
    # 7 day averages
    trop7_031519 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_7dayavg_starting03152019_QA75.ncf')
    trop7_031519 = trop7_031519.variables['NO2'][:]
    trop7_032219 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_7dayavg_starting03222019_QA75.ncf')
    trop7_032219 = trop7_032219.variables['NO2'][:]
    trop7_032919 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_7dayavg_starting03292019_QA75.ncf')
    trop7_032919 = trop7_032919.variables['NO2'][:]
    trop7_040519 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_7dayavg_starting04052019_QA75.ncf')
    trop7_040519 = trop7_040519.variables['NO2'][:]
    trop7_041219 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_7dayavg_starting04122019_QA75.ncf')
    trop7_041219 = trop7_041219.variables['NO2'][:]
    trop7_041919 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_7dayavg_starting04192019_QA75.ncf')
    trop7_041919 = trop7_041919.variables['NO2'][:]
    trop7_042619 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_7dayavg_starting04262019_QA75.ncf')
    trop7_042619 = trop7_042619.variables['NO2'][:]
    trop7_050319 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_7dayavg_starting05032019_QA75.ncf')
    trop7_050319 = trop7_050319.variables['NO2'][:]
    trop7_051019 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_7dayavg_starting05102019_QA75.ncf')
    trop7_051019 = trop7_051019.variables['NO2'][:]
    trop7_051719 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_7dayavg_starting05172019_QA75.ncf')
    trop7_051719 = trop7_051719.variables['NO2'][:]
    trop7_052419 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_7dayavg_starting05242019_QA75.ncf')
    trop7_052419 = trop7_052419.variables['NO2'][:]
    trop7_053119 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_7dayavg_starting05312019_QA75.ncf')
    trop7_053119 = trop7_053119.variables['NO2'][:]
    trop7_060719 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_7dayavg_starting06072019_QA75.ncf')
    trop7_060719 = trop7_060719.variables['NO2'][:]
    # 14 day averages
    trop14_031519 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_14dayavg_starting03152019_QA75.ncf')
    trop14_031519 = trop14_031519.variables['NO2'][:]
    trop14_032919 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_14dayavg_starting03292019_QA75.ncf')
    trop14_032919 = trop14_032919.variables['NO2'][:]
    trop14_041219 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_14dayavg_starting04122019_QA75.ncf')
    trop14_041219 = trop14_041219.variables['NO2'][:]
    trop14_042619 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_14dayavg_starting04262019_QA75.ncf')
    trop14_042619 = trop14_042619.variables['NO2'][:]
    trop14_051019 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_14dayavg_starting05102019_QA75.ncf')
    trop14_051019 = trop14_051019.variables['NO2'][:]
    trop14_052419 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_14dayavg_starting05242019_QA75.ncf')
    trop14_052419 = trop14_052419.variables['NO2'][:]
    trop14_060719 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_14dayavg_starting06072019_QA75.ncf')
    trop14_060719 = trop14_060719.variables['NO2'][:]
    # 31 day averages
    trop31_031319 = nc.Dataset(DIR_NO2+ 
        'Tropomi_NO2_griddedon0.01grid_31dayavg_starting03132019_QA75.ncf')
    trop31_031319 = trop31_031319.variables['NO2'][:]
    trop31_041319 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_31dayavg_starting04132019_QA75.ncf')
    trop31_041319 = trop31_041319.variables['NO2'][:]
    trop31_051419 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_31dayavg_starting05142019_QA75.ncf')
    trop31_051419 = trop31_051419.variables['NO2'][:]
    # Baseline average
    trop_base = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_Mar13-Jun132019_precovid19_QA75.ncf')
    trop_base = trop_base.variables['NO2'][:]
    # 2019 average
    trop_2019 = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_2019_QA75.ncf')
    trop_2019 = trop_2019.variables['NO2'][:]
    # All 
    trop_all = nc.Dataset(DIR_NO2+
        'Tropomi_NO2_griddedon0.01grid_allyears_QA75.ncf')
    trop_all = trop_all.variables['NO2'][:]
    # Dimensional information (lat/lon)
    grid = nc.Dataset(DIR_NO2+'LatLonGrid.ncf')
    lng_full = grid.variables['LON'][:]
    lat_full = grid.variables['LAT'][:]
        
    
    # Select ~state from gridded NO2 retrievals 
    #----------------------
    # Find min/max of each shapes in file for subsetting NO2 grids
    max_lats, min_lats, max_lngs, min_lngs = [],[], [], []
    for t in tracts:
        # Longitudes
        max_lngs.append(np.nanmax(np.array(t.points)[:,0]))
        min_lngs.append(np.nanmin(np.array(t.points)[:,0]))
        # Latitude
        max_lats.append(np.nanmax(np.array(t.points)[:,1]))
        min_lats.append(np.nanmin(np.array(t.points)[:,1]))
    # Find global min/max
    max_lats = np.nanmax(max_lats)
    min_lats = np.nanmin(min_lats)
    max_lngs = np.nanmax(max_lngs)
    min_lngs = np.nanmin(min_lngs)
    
    # MN and ME (and maybe OR) break the code
    if (max_lats+0.25 > lat_full[-1]):
        trop7_031519, lat, lng = find_grid_in_bb(trop7_031519, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop7_032219, lat, lng = find_grid_in_bb(trop7_032219, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop7_032919, lat, lng = find_grid_in_bb(trop7_032919, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])                                                 
        trop7_040519, lat, lng = find_grid_in_bb(trop7_040519, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])                                                 
        trop7_041219, lat, lng = find_grid_in_bb(trop7_041219, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])                                                 
        trop7_041919, lat, lng = find_grid_in_bb(trop7_041919, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])                                                 
        trop7_042619, lat, lng = find_grid_in_bb(trop7_042619, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop7_050319, lat, lng = find_grid_in_bb(trop7_050319, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop7_051019, lat, lng = find_grid_in_bb(trop7_051019, lat_full,  
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop7_051719, lat, lng = find_grid_in_bb(trop7_051719, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop7_052419, lat, lng = find_grid_in_bb(trop7_052419, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop7_053119, lat, lng = find_grid_in_bb(trop7_053119, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop7_060719, lat, lng = find_grid_in_bb(trop7_060719, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop14_031519, lat, lng = find_grid_in_bb(trop14_031519, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])                                                  
        trop14_032919, lat, lng = find_grid_in_bb(trop14_032919, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop14_041219, lat, lng = find_grid_in_bb(trop14_041219, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop14_042619, lat, lng = find_grid_in_bb(trop14_042619, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop14_051019, lat, lng = find_grid_in_bb(trop14_051019, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop14_052419, lat, lng = find_grid_in_bb(trop14_052419, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop14_060719, lat, lng = find_grid_in_bb(trop14_060719, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop31_031319, lat, lng = find_grid_in_bb(trop31_031319, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])                                                  
        trop31_041319, lat, lng = find_grid_in_bb(trop31_041319, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])
        trop31_051419, lat, lng = find_grid_in_bb(trop31_051419, lat_full,
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            lat_full[-1])                                                  
        trop_base, lat, lng = find_grid_in_bb(trop_base, lat_full, lng_full,
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, lat_full[-1])
        trop_2019, lat, lng = find_grid_in_bb(trop_2019, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, lat_full[-1])
        trop_all, lat, lng = find_grid_in_bb(trop_all, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, lat_full[-1])    
    elif (min_lats-0.25 < lat_full[0]):
        trop7_031519, lat, lng = find_grid_in_bb(trop7_031519, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop7_032219, lat, lng = find_grid_in_bb(trop7_032219, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)    
        trop7_032919, lat, lng = find_grid_in_bb(trop7_032919, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop7_040519, lat, lng = find_grid_in_bb(trop7_040519, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop7_041219, lat, lng = find_grid_in_bb(trop7_041219, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop7_042619, lat, lng = find_grid_in_bb(trop7_042619, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop7_041919, lat, lng = find_grid_in_bb(trop7_041919, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop7_050319, lat, lng = find_grid_in_bb(trop7_050319, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop7_051019, lat, lng = find_grid_in_bb(trop7_051019, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop7_051719, lat, lng = find_grid_in_bb(trop7_051719, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop7_052419, lat, lng = find_grid_in_bb(trop7_052419, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop7_053119, lat, lng = find_grid_in_bb(trop7_053119, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop7_060719, lat, lng = find_grid_in_bb(trop7_060719, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop14_031519, lat, lng = find_grid_in_bb(trop14_031519, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop14_032919, lat, lng = find_grid_in_bb(trop14_032919, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop14_041219, lat, lng = find_grid_in_bb(trop14_041219, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop14_042619, lat, lng = find_grid_in_bb(trop14_042619, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop14_051019, lat, lng = find_grid_in_bb(trop14_051019, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop14_052419, lat, lng = find_grid_in_bb(trop14_052419, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop14_060719, lat, lng = find_grid_in_bb(trop14_060719, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop31_031319, lat, lng = find_grid_in_bb(trop31_031319, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)    
        trop31_041319, lat, lng = find_grid_in_bb(trop31_041319, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop31_051419, lat, lng = find_grid_in_bb(trop31_051419, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop_base, lat, lng = find_grid_in_bb(trop_base, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop_2019, lat, lng = find_grid_in_bb(trop_2019, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
        trop_all, lat, lng = find_grid_in_bb(trop_all, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, lat_full[0], max_lats+0.25)
    elif (max_lngs+0.25 > lng_full[-1]):
        trop7_031519, lat, lng = find_grid_in_bb(trop7_031519, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop7_032219, lat, lng = find_grid_in_bb(trop7_032219, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop7_032919, lat, lng = find_grid_in_bb(trop7_032919, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop7_040519, lat, lng = find_grid_in_bb(trop7_040519, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop7_041219, lat, lng = find_grid_in_bb(trop7_041219, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop7_041919, lat, lng = find_grid_in_bb(trop7_041919, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop7_042619, lat, lng = find_grid_in_bb(trop7_042619, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop7_050319, lat, lng = find_grid_in_bb(trop7_050319, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop7_051019, lat, lng = find_grid_in_bb(trop7_051019, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop7_051719, lat, lng = find_grid_in_bb(trop7_051719, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop7_052419, lat, lng = find_grid_in_bb(trop7_052419, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop7_053119, lat, lng = find_grid_in_bb(trop7_053119, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop7_060719, lat, lng = find_grid_in_bb(trop7_060719, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop14_031519, lat, lng = find_grid_in_bb(trop14_031519, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop14_032919, lat, lng = find_grid_in_bb(trop14_032919, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop14_041219, lat, lng = find_grid_in_bb(trop14_041219, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop14_042619, lat, lng = find_grid_in_bb(trop14_042619, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop14_051019, lat, lng = find_grid_in_bb(trop14_051019, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop14_052419, lat, lng = find_grid_in_bb(trop14_052419, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop14_060719, lat, lng = find_grid_in_bb(trop14_060719, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop31_031319, lat, lng = find_grid_in_bb(trop31_031319, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop31_041319, lat, lng = find_grid_in_bb(trop31_041319, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop31_051419, lat, lng = find_grid_in_bb(trop31_051419, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop_base, lat, lng = find_grid_in_bb(trop_base, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
        trop_2019, lat, lng = find_grid_in_bb(trop_2019, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)    
        trop_all, lat, lng = find_grid_in_bb(trop_all, lat_full, lng_full, 
            min_lngs-0.25, lng_full[-1], min_lats-0.25, max_lats+0.25)
    elif (min_lngs-0.25 < lng_full[0]):
        trop7_031519, lat, lng = find_grid_in_bb(trop7_031519, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop7_032219, lat, lng = find_grid_in_bb(trop7_032219, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)   
        trop7_032919, lat, lng = find_grid_in_bb(trop7_032919, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop7_040519, lat, lng = find_grid_in_bb(trop7_040519, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop7_041219, lat, lng = find_grid_in_bb(trop7_041219, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop7_041919, lat, lng = find_grid_in_bb(trop7_041919, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop7_042619, lat, lng = find_grid_in_bb(trop7_042619, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop7_050319, lat, lng = find_grid_in_bb(trop7_050319, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop7_051019, lat, lng = find_grid_in_bb(trop7_051019, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop7_051719, lat, lng = find_grid_in_bb(trop7_051719, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop7_052419, lat, lng = find_grid_in_bb(trop7_052419, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop7_053119, lat, lng = find_grid_in_bb(trop7_053119, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop7_060719, lat, lng = find_grid_in_bb(trop7_060719, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop14_031519, lat, lng = find_grid_in_bb(trop14_031519, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)       
        trop14_032919, lat, lng = find_grid_in_bb(trop14_032919, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)    
        trop14_041219, lat, lng = find_grid_in_bb(trop14_041219, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)    
        trop14_042619, lat, lng = find_grid_in_bb(trop14_042619, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)    
        trop14_051019, lat, lng = find_grid_in_bb(trop14_051019, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)    
        trop14_052419, lat, lng = find_grid_in_bb(trop14_052419, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)    
        trop14_060719, lat, lng = find_grid_in_bb(trop14_060719, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)    
        trop31_031319, lat, lng = find_grid_in_bb(trop31_031319, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)    
        trop31_041319, lat, lng = find_grid_in_bb(trop31_041319, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)    
        trop31_051419, lat, lng = find_grid_in_bb(trop31_051419, lat_full, lng_full, 
            lng_full[0], max_lngs+0.25, min_lats-0.25, max_lats+0.25)    
        trop_base, lat, lng = find_grid_in_bb(trop_base, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop_2019, lat, lng = find_grid_in_bb(trop_2019, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop_all, lat, lng = find_grid_in_bb(trop_all, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, max_lats+0.25)
    else:    
        trop7_031519, lat, lng = find_grid_in_bb(trop7_031519, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop7_032219, lat, lng = find_grid_in_bb(trop7_032219, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop7_032919, lat, lng = find_grid_in_bb(trop7_032919, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop7_040519, lat, lng = find_grid_in_bb(trop7_040519, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop7_041219, lat, lng = find_grid_in_bb(trop7_041219, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop7_041919, lat, lng = find_grid_in_bb(trop7_041919, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop7_042619, lat, lng = find_grid_in_bb(trop7_042619, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop7_050319, lat, lng = find_grid_in_bb(trop7_050319, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop7_051019, lat, lng = find_grid_in_bb(trop7_051019, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop7_051719, lat, lng = find_grid_in_bb(trop7_051719, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop7_052419, lat, lng = find_grid_in_bb(trop7_052419, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop7_053119, lat, lng = find_grid_in_bb(trop7_053119, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop7_060719, lat, lng = find_grid_in_bb(trop7_060719, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop14_031519, lat, lng = find_grid_in_bb(trop14_031519, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop14_032919, lat, lng = find_grid_in_bb(trop14_032919, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop14_041219, lat, lng = find_grid_in_bb(trop14_041219, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop14_042619, lat, lng = find_grid_in_bb(trop14_042619, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop14_051019, lat, lng = find_grid_in_bb(trop14_051019, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop14_052419, lat, lng = find_grid_in_bb(trop14_052419, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop14_060719, lat, lng = find_grid_in_bb(trop14_060719, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop31_031319, lat, lng = find_grid_in_bb(trop31_031319, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop31_041319, lat, lng = find_grid_in_bb(trop31_041319, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop31_051419, lat, lng = find_grid_in_bb(trop31_051419, lat_full, 
            lng_full, min_lngs-0.25, max_lngs+0.25, min_lats-0.25, 
            max_lats+0.25)
        trop_base, lat, lng = find_grid_in_bb(trop_base, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop_2019, lat, lng = find_grid_in_bb(trop_2019, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, max_lats+0.25)
        trop_all, lat, lng = find_grid_in_bb(trop_all, lat_full, lng_full, 
            min_lngs-0.25, max_lngs+0.25, min_lats-0.25, max_lats+0.25)    
    print('# # # # # TROPOMI NO2 files read! # # # # #')
    time.sleep(2)
    # Loop through shapes (each shape corresponds a census tract) append
    # GEOIDs, NO2 concentrations, and demographic information to list
    #----------------------
    df = []
    for tract in tqdm(np.arange(0, len(tracts), 1), leave=True):
        # Extract metadata (e.g., STATEFP|State FIPS Code, 
        # COUNTYFP|County FIPS Code, GEOID|Geographic Identifier, 
        # NAMELSAD|Full Name)
        record = records[tract]
        # Build a shapely polygon from shape
        tract = shape(tracts[tract]) 
        # Extract GEOID of record
        geoid = record['GEOID']
        # Find grid cells within polygon for NO2 in polygon 
        i_inside, j_inside = [], []
        for i, ilat in enumerate(lat):
            for j, jlng in enumerate(lng): 
                point = Point(jlng, ilat)
                if tract.contains(point) is True:
                    # Fill lists with indices in grid within polygon
                    i_inside.append(i)
                    j_inside.append(j)
        # Find census tract information from NHGIS by "converting" GEOID 
        # to GISJOIN identifier 
        # GISJOIN identifiers match the identifiers used in NHGIS data 
        # tables and boundary files. A block GISJOIN concatenates these codes:
        #    "G" prefix         This prevents applications from automatically 
        #                       reading the identifier as a number and, in 
        #                       effect, dropping important leading zeros
        #    State NHGIS code	3 digits (FIPS + "0"). NHGIS adds a zero to 
        #                       state FIPS codes to differentiate current 
        #                       states from historical territories.
        #    County NHGIS code	4 digits (FIPS + "0"). NHGIS adds a zero to 
        #                       county FIPS codes to differentiate current 
        #                       counties from historical counties.
        #    Census tract code	6 digits for 2000 and 2010 tracts. 1990 tract 
        #                       codes use either 4 or 6 digits.
        #    Census block code	4 digits for 2000 and 2010 blocks. 1990 block
        #                       codes use either 3 or 4 digits.
        # GEOID identifiers correspond to the codes used in most current 
        # Census sources (American FactFinder, TIGER/Line, Relationship Files, 
        # etc.). A block GEOID concatenates these codes:
        #    State FIPS code	2 digits
        #    County FIPS code	3 digits
        #    Census tract code	6 digits. 1990 tract codes that were 
        #                       originally 4 digits (as in NHGIS files) are 
        #                       extended to 6 with an appended "00" (as in 
        #                       Census Relationship Files).
        #    Census block code	4 digits for 2000 and 2010 blocks. 1990 block 
        #                       codes use either 3 or 4 digits.
        geoid_converted = 'G'+geoid[:2]+'0'+geoid[2:5]+'0'+geoid[5:]
        tract_nhgis = nhgis.loc[nhgis['GISJOIN']==geoid_converted]
        # If there are no TROPOMI grid cells within census tract, fill row 
        # corresponding to tract with NaNs for all variables
        if (len(i_inside)==0) or (len(j_inside)==0):
            # # # # Option 1: Treat small tracts as missing data/NaN for NO2
            # dicttemp = {'GEOID':geoid, 'PRENO2':np.nan, 'POSTNO2':np.nan, 
            #     'PRENO2APR':np.nan, 'POSTNO2APR':np.nan, 
            #     'PRENO2MAR-SEP':np.nan, 'POSTNO2MAR-SEP':np.nan, 
            #     'ALLNO2':np.nan}
            # for var in var_extract:
            #     dicttemp[var] = tract_nhgis[var].values[0]        
            # df.append(dicttemp)   
            # del dicttemp
            # # # # Option 2: Interpolate using inverse distance weighting 
            # https://rafatieppo.github.io/post/2018_07_27_idw2pyr/
            tract_latcentroid = tract.centroid.y
            tract_lngcentroid = tract.centroid.x           
            idx_latnear = geo_idx(tract_latcentroid, lat)
            idx_lngnear = geo_idx(tract_lngcentroid, lng)
            # Indices for 8 nearby points
            lng_idx = [idx_lngnear-1, idx_lngnear, idx_lngnear+1, 
                idx_lngnear-1, idx_lngnear+1, idx_lngnear-1, idx_lngnear, 
                idx_lngnear+1]
            lat_idx = [idx_latnear+1, idx_latnear+1, idx_latnear+1, 
                idx_latnear, idx_latnear, idx_latnear-1, idx_latnear-1, 
                idx_latnear-1]
            # Known coordinates
            x = lng[lng_idx].data
            y = lat[lat_idx].data
            # Known NO2 values 
            # z_preNO2 = preNO2[lat_idx,lng_idx].data.flatten()
            z_trop7_031519 = trop7_031519[lat_idx,lng_idx]
            z_trop7_032219 = trop7_032219[lat_idx,lng_idx]
            z_trop7_032919 = trop7_032919[lat_idx,lng_idx]
            z_trop7_040519 = trop7_040519[lat_idx,lng_idx]
            z_trop7_041219 = trop7_041219[lat_idx,lng_idx]
            z_trop7_041919 = trop7_041919[lat_idx,lng_idx]        
            z_trop7_042619 = trop7_042619[lat_idx,lng_idx]        
            z_trop7_050319 = trop7_050319[lat_idx,lng_idx]
            z_trop7_051019 = trop7_051019[lat_idx,lng_idx]
            z_trop7_051719 = trop7_051719[lat_idx,lng_idx]
            z_trop7_052419 = trop7_052419[lat_idx,lng_idx]
            z_trop7_053119 = trop7_053119[lat_idx,lng_idx]
            z_trop7_060719 = trop7_060719[lat_idx,lng_idx]
            z_trop14_031519 = trop14_031519[lat_idx,lng_idx]
            z_trop14_032919 = trop14_032919[lat_idx,lng_idx]
            z_trop14_041219 = trop14_041219[lat_idx,lng_idx]
            z_trop14_042619 = trop14_042619[lat_idx,lng_idx]
            z_trop14_051019 = trop14_051019[lat_idx,lng_idx]
            z_trop14_052419 = trop14_052419[lat_idx,lng_idx]
            z_trop14_060719 = trop14_060719[lat_idx,lng_idx]
            z_trop31_031319 = trop31_031319[lat_idx,lng_idx]
            z_trop31_041319 = trop31_041319[lat_idx,lng_idx]
            z_trop31_051419 = trop31_051419[lat_idx,lng_idx]
            z_trop_base = trop_base[lat_idx,lng_idx]
            z_trop_2019 = trop_2019[lat_idx,lng_idx]
            z_trop_all = trop_all[lat_idx,lng_idx]
            dicttemp = {'GEOID':geoid, 
                'trop7_031519':idwr(x,y,z_trop7_031519,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop7_032219':idwr(x,y,z_trop7_032219,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop7_032919':idwr(x,y,z_trop7_032919,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop7_040519':idwr(x,y,z_trop7_040519,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop7_041219':idwr(x,y,z_trop7_041219,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop7_041919':idwr(x,y,z_trop7_041919,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop7_042619':idwr(x,y,z_trop7_042619,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop7_050319':idwr(x,y,z_trop7_050319,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop7_051019':idwr(x,y,z_trop7_051019,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop7_051719':idwr(x,y,z_trop7_051719,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop7_052419':idwr(x,y,z_trop7_052419,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop7_053119':idwr(x,y,z_trop7_053119,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop7_060719':idwr(x,y,z_trop7_060719,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop14_031519':idwr(x,y,z_trop14_031519,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop14_032919':idwr(x,y,z_trop14_032919,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop14_041219':idwr(x,y,z_trop14_041219,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop14_042619':idwr(x,y,z_trop14_042619,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop14_051019':idwr(x,y,z_trop14_051019,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop14_052419':idwr(x,y,z_trop14_052419,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop14_060719':idwr(x,y,z_trop14_060719,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop31_031319':idwr(x,y,z_trop31_031319,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop31_041319':idwr(x,y,z_trop31_041319,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop31_051419':idwr(x,y,z_trop31_051419,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop_base':idwr(x,y,z_trop_base,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop_2019':idwr(x,y,z_trop_2019,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1],
                'trop_all':idwr(x,y,z_trop_all,[tract_lngcentroid],
                              [tract_latcentroid])[0][-1]}
            for var in var_extract:
                dicttemp[var] = tract_nhgis[var].values[0]        
            df.append(dicttemp)   
            del dicttemp   
        # For census tracts with TROPOMI grid cells, find grid cell-averaged
        # NO2 and demographic information from NHGIS census files
        else: 
            trop7_031519_inside = np.nanmean(trop7_031519[i_inside, j_inside])
            trop7_032219_inside = np.nanmean(trop7_032219[i_inside, j_inside])
            trop7_032919_inside = np.nanmean(trop7_032919[i_inside, j_inside])
            trop7_040519_inside = np.nanmean(trop7_040519[i_inside, j_inside])
            trop7_041219_inside = np.nanmean(trop7_041219[i_inside, j_inside])
            trop7_041919_inside = np.nanmean(trop7_041919[i_inside, j_inside])        
            trop7_042619_inside = np.nanmean(trop7_042619[i_inside, j_inside])
            trop7_050319_inside = np.nanmean(trop7_050319[i_inside, j_inside])
            trop7_051019_inside = np.nanmean(trop7_051019[i_inside, j_inside])        
            trop7_051719_inside = np.nanmean(trop7_051719[i_inside, j_inside])
            trop7_052419_inside = np.nanmean(trop7_052419[i_inside, j_inside])
            trop7_053119_inside = np.nanmean(trop7_053119[i_inside, j_inside])
            trop7_060719_inside = np.nanmean(trop7_060719[i_inside, j_inside])
            trop14_031519_inside = np.nanmean(trop14_031519[i_inside, j_inside])
            trop14_032919_inside = np.nanmean(trop14_032919[i_inside, j_inside])
            trop14_041219_inside = np.nanmean(trop14_041219[i_inside, j_inside])
            trop14_042619_inside = np.nanmean(trop14_042619[i_inside, j_inside])
            trop14_051019_inside = np.nanmean(trop14_051019[i_inside, j_inside])
            trop14_052419_inside = np.nanmean(trop14_052419[i_inside, j_inside])
            trop14_060719_inside = np.nanmean(trop14_060719[i_inside, j_inside])
            trop31_031319_inside = np.nanmean(trop31_031319[i_inside, j_inside])
            trop31_041319_inside = np.nanmean(trop31_041319[i_inside, j_inside])
            trop31_051419_inside = np.nanmean(trop31_051419[i_inside, j_inside])
            trop_base_inside = np.nanmean(trop_base[i_inside, j_inside])
            trop_2019_inside = np.nanmean(trop_2019[i_inside, j_inside])
            trop_all_inside = np.nanmean(trop_all[i_inside, j_inside])
            dicttemp = {'GEOID':geoid, 
                'trop7_031519':trop7_031519_inside,
                'trop7_032219':trop7_032219_inside,
                'trop7_032919':trop7_032919_inside,
                'trop7_040519':trop7_040519_inside,
                'trop7_041219':trop7_041219_inside,
                'trop7_041919':trop7_041919_inside,
                'trop7_042619':trop7_042619_inside,
                'trop7_050319':trop7_050319_inside,
                'trop7_051019':trop7_051019_inside,
                'trop7_051719':trop7_051719_inside,
                'trop7_052419':trop7_052419_inside,
                'trop7_053119':trop7_053119_inside,
                'trop7_060719':trop7_060719_inside,
                'trop14_031519':trop14_031519_inside,
                'trop14_032919':trop14_032919_inside,
                'trop14_041219':trop14_041219_inside,
                'trop14_042619':trop14_042619_inside,
                'trop14_051019':trop14_051019_inside,
                'trop14_052419':trop14_052419_inside,
                'trop14_060719':trop14_060719_inside,
                'trop31_031319':trop31_031319_inside,
                'trop31_041319':trop31_041319_inside,
                'trop31_051419':trop31_051419_inside,
                'trop_base':trop_base_inside,
                'trop_2019':trop_2019_inside,
                'trop_all':trop_all_inside}        
            for var in var_extract:
                dicttemp[var] = tract_nhgis[var].values[0]
            df.append(dicttemp)               
            del dicttemp
    df = pd.DataFrame(df)
    # Set GEOID as column 
    df = df.set_index('GEOID')
    # Add FIPS code as column
    df.insert(1, 'FIPS', sFIPS)    
    print('# # # # # TROPOMI/census harmonized at tract level! # # # # #')
    # Save DataFrame
    #----------------------
    df = df.replace('NaN', '', regex=True)
    df.to_csv(DIR_OUT+'Tropomi_NO2_interpolated_%s_metsens_%s'%(sFIPS,
        censusfile), sep = ',')
    print('# # # # # Output file written! # # # # #')  
    return 
# # # # To harmonize TROPOMI NO2 with census 
# censusfile = 'nhgis0003_ds239_20185_2018_tract.csv'
# FIPS = ['01', '04', '05', '06', '08', '09', '10', '11', '12', '13', '16', 
#     '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
#     '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', 
#     '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50',
#     '51', '53', '54', '55', '56']
# for state in FIPS:
#     harmonize_tropomino2_census(state, censusfile)

# # # # To assess meteorological sensitivity
# censusfile = 'nhgis0003_ds239_20185_2018_tract.csv'
# FIPS = ['01', '04', '05', '06', '08', '09', '10', '11', '12', '13', '16']
# for state in FIPS:
#     harmonize_tropomino2_census_metsens(state, censusfile)
# FIPS = ['17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27']
# for state in FIPS:
#     harmonize_tropomino2_census_metsens(state, censusfile)
# FIPS = ['28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38']
# for state in FIPS:
#     harmonize_tropomino2_census_metsens(state, censusfile)
# FIPS = ['39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50']
# for state in FIPS:
#     harmonize_tropomino2_census_metsens(state, censusfile)
# FIPS = ['51', '53', '54', '55', '56']
# for state in FIPS:
#     harmonize_tropomino2_census_metsens(state, censusfile)

