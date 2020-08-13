#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module calculates a psuedo-road density in each census tract and the vehicle
ownership in the tract"""

# DIR_NO2 = '/Users/ghkerr/GW/data/'
# DIR_CENSUS = '/Users/ghkerr/GW/data/census_no2_harmonzied/'
# DIR_SHAPEFILE = '/Users/ghkerr/GW/data/geography/tigerline/'
# DIR_OUT = '/Users/ghkerr/GW/data/census_no2_harmonzied/'
DIR_NO2 = '/mnt/scratch1/gaige/data/tropomi_ej/'
DIR_CENSUS = '/mnt/scratch1/gaige/data/tropomi_ej/'
DIR_SHAPEFILE = '/mnt/scratch1/gaige/data/tropomi_ej/'
DIR_OUT = '/mnt/scratch1/gaige/data/tropomi_ej/'

def harmonized_vehicleownership_roaddensity(FIPS):
    """function loads primary U.S. road geography and census tract geography 
    for the specified states. The centroids of each road segment and census 
    tract are found, and the number of roads within 1km, 5km, 10km, 20km, and 
    50km are calculated. Additionally, census information on tenure by 
    vehicles available and renting/owning households is harmonized with the 
    road density at the tract level.     

    Parameters
    ----------
    FIPS : list
        Contains FIPS code(s) for the state(s) of interest

    Returns
    -------
    None
    """
    import numpy as np
    import pandas as pd
    from cartopy.io import shapereader
    # Constants
    R = 6373.0
    searchrad = 2.
    # Lists will be filled with requested information about nearby 
    # road density and 
    geoids = []
    within50, within20, within10, within5, within1 = [], [], [], [], []
    frac_nocars, frac_rent, frac_own = [], [], []
    # Open ACS tract-level information about vehicle ownership
    vehicleown = pd.read_csv(DIR_CENSUS+'nhgis0004_ds239_20185_2018_tract.csv', 
       delimiter=',', header=0, engine='python')
    # Replace GISJOIN identified with GEOID for easy look-up
    gisjoin_to_geoid = [x[1:3]+x[4:7]+x[8:] for x in vehicleown['GISJOIN']]
    vehicleown['GEOID'] = gisjoin_to_geoid
    # Make GEOID a string and index row 
    vehicleown = vehicleown.set_index('GEOID')
    vehicleown.index = vehicleown.index.map(str)
    # Make other columns floats
    # Tenure by Vehicles Available
    # Universe:    Occupied housing units
    # Source code: B25044
    # NHGIS code:  AJ2W
    # AJ2WM001:    Total
    # AJ2WM002:    Owner occupied
    # AJ2WM003:    Owner occupied: No vehicle available
    # AJ2WM009:    Renter occupied
    # AJ2WM010:    Renter occupied: No vehicle available
    for col in ['AJ2WE001','AJ2WE002','AJ2WE003','AJ2WE004','AJ2WE005',
        'AJ2WE006','AJ2WE007','AJ2WE008','AJ2WE009','AJ2WE010','AJ2WE011',
    	'AJ2WE012','AJ2WE013','AJ2WE014','AJ2WE015']:
        vehicleown[col] = vehicleown[col].astype(float)
    # Replace tracts with 0 houses with NaN to deal with division by zero 
    # (otherwise if there are 0 houses and, for example, 0 rental properties, 
    # there will be a RuntimeWarning)
    vehicleown.loc[vehicleown['AJ2WE001'] == 0.] = np.nan
    # Open nationwide primary road shapefiles and records. There are 18565
    # roads and the following types (attribute RTTYP):
    # C (county) - 16/18565
    # I (interstate) - 5699/18565
    # M (common name) - 5741/18565
    # O (other) - 26/18565
    # S (state recognized) - 3218/18565
    # U (U.S.) - 3865/18565
    shp = shapereader.Reader(DIR_SHAPEFILE+'tl_2018_us_primaryroads/'+
        'tl_2018_us_primaryroads')
    roads = list(shp.geometries())
    # # If desired, specific road type can be subset
    # roads_rttyp = [x.attributes['RTTYP'] for x in roads_records]
    # where_interstate = np.where(np.array(roads_rttyp)=='I')[0]
    # roads_i = []
    # roads_i += [roads[x] for x in where_interstate]
    # roads = cfeature.ShapelyFeature(roads_i, proj)
    # Find centroid of roads 
    road_centroid_lat, road_centroid_lng = [], []
    for r in roads: 
        road_centroid_lat.append(r.centroid.xy[1][0])
        road_centroid_lng.append(r.centroid.xy[0][0])
    # Create DataFrame from road centroids for easy indexing
    road_centroids = pd.DataFrame({'RoadLat':road_centroid_lat,
        'RoadLng':road_centroid_lng})
    for FIPS_i in FIPS: 
        print(FIPS_i)
        # Tigerline shapefile for state
        shp = shapereader.Reader(DIR_SHAPEFILE+'tl_2019_%s_tract/'%FIPS_i+    
            'tl_2019_%s_tract'%FIPS_i)
        tracts_records = list(shp.records())
        # Loop through tracts and find current latitude and longitude of the
        # internal point of tract
        for t in tracts_records:
            geoid = t.attributes['GEOID']
            tract_intptlat = float(t.attributes['INTPTLAT'])
            tract_intptlon = float(t.attributes['INTPTLON'])
            # Find vehicle ownership in tract
            tract_vehicleown = vehicleown.loc[vehicleown.index==geoid]
            # Slice roads ~near tract center
            road_centroids_attract = road_centroids.loc[
                (road_centroids['RoadLat']>tract_intptlat-searchrad) &
                (road_centroids['RoadLat']<tract_intptlat+searchrad) & 
                (road_centroids['RoadLng']>tract_intptlon-searchrad) &
                (road_centroids['RoadLng']<tract_intptlon+searchrad)]
            # Loop through roads within search radius and calculate haversine 
            # distance
            dists = []
            for r in np.arange(0,len(road_centroids_attract),1):
                lon1, lat1, lon2, lat2 = map(np.radians, [
                    road_centroids_attract.iloc[r]['RoadLng'],
                    road_centroids_attract.iloc[r]['RoadLat'],
                    tract_intptlon, tract_intptlat])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = (np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * 
                    np.sin(dlon/2.0)**2)
                c = 2 * np.arcsin(np.sqrt(a))
                dists.append(R * c)
            # Append information 
            geoids.append(geoid)
            frac_nocars.append((tract_vehicleown['AJ2WE003'].values[0]+
                tract_vehicleown['AJ2WE010'].values[0])/
                tract_vehicleown['AJ2WE001'].values[0])
            frac_rent.append(tract_vehicleown['AJ2WE009'].values[0]/
                tract_vehicleown['AJ2WE001'].values[0])
            frac_own.append(tract_vehicleown['AJ2WE002'].values[0]/
                tract_vehicleown['AJ2WE001'].values[0])
            within50.append(np.where(np.array(dists)<50)[0].shape[0])
            within20.append(np.where(np.array(dists)<20)[0].shape[0])
            within10.append(np.where(np.array(dists)<10)[0].shape[0])
            within5.append(np.where(np.array(dists)<5)[0].shape[0])
            within1.append(np.where(np.array(dists)<1)[0].shape[0])
    # Create output DataFrame
    road_density = pd.DataFrame({'GEOID':geoids, 'FracRent':frac_rent,
        'FracOwn':frac_own, 'FracNoCar':frac_nocars, 'Within50':within50,
        'Within20':within20, 'Within10':within10, 'Within5':within5,
        'Within51':within1})
    road_density = road_density.set_index('GEOID')
    road_density.index = road_density.index.map(str)
    road_density = road_density.replace('NaN', '', regex=True)
    road_density.to_csv(
        DIR_OUT+'vehicleownership_roaddensity_us.csv', sep = ',')    
    return 

FIPS = ['01', '04', '05', '06', '08', '09', '10', '11', '12', '13', '16', 
        '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
        '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', 
        '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50',
        '51', '53', '54', '55', '56']
FIPS = ['01']
harmonized_vehicleownership_roaddensity(FIPS)