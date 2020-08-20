#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module calculates a psuedo-road density in each census tract for TIGER 
primary and secondary roads. The fraction of households in each tract without 
vehicles and the method of transportation to work are also recorded in output 
file"""

DIR_NO2 = '/Users/ghkerr/GW/data/'
DIR_CENSUS = '/Users/ghkerr/GW/data/census_no2_harmonzied/'
DIR_SHAPEFILE = '/Users/ghkerr/GW/data/geography/tigerline/'
DIR_OUT = '/Users/ghkerr/GW/data/census_no2_harmonzied/'
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
    import cartopy.feature as cfeature
    import cartopy.crs as ccrs
    # Constants
    R = 6373.0
    searchrad = 0.5
    
    # Lists will be filled with requested information about nearby 
    # road density and 
    geoids = []
    primary10, primary5, primary2, primary1 = [], [], [], []
    secondary10, secondary5, secondary2, secondary1 = [], [], [], []
    frac_nocars, frac_rent, frac_own = [], [], []
    frac_personal, frac_public, frac_bike, frac_walk = [], [], [], []
    frac_other, frac_home, frac_taxi, frac_motorcycle = [], [], [], []

    # # # # Open ACS tract-level information about vehicle ownership
    vehicleown = pd.read_csv(DIR_CENSUS+'nhgis0005_ds239_20185_2018_tract.csv', 
       delimiter=',', header=0, engine='python')
    # Replace GISJOIN identified with GEOID for easy look-up
    gisjoin_to_geoid = [x[1:3]+x[4:7]+x[8:] for x in vehicleown['GISJOIN']]
    vehicleown['GEOID'] = gisjoin_to_geoid
    # Make GEOID a string and index row 
    vehicleown = vehicleown.set_index('GEOID')
    vehicleown.index = vehicleown.index.map(str)
    # Make other columns floats
    col_relevant = ['AJ2WE001', # Total
        'AJ2WE002', # Owner occupied
        'AJ2WE003', # Owner occupied: No vehicle available
        'AJ2WE009', # Renter occupied
        'AJ2WE010', # Renter occupied: No vehicle available
        'AJXCE001', # Total
        'AJXCE002', # Car, truck, or van
        'AJXCE010', # Public transportation (excluding taxicab)
        'AJXCE016', # Taxicab
        'AJXCE017', # Motorcycle
        'AJXCE018', # Bicycle
        'AJXCE019', # Walked
        'AJXCE020', # Other means
        'AJXCE021' # Worked at home
        ]
    vehicleown = vehicleown[col_relevant]
    for col in col_relevant:
        vehicleown[col] = vehicleown[col].astype(float)
    # Replace tracts with 0 houses with NaN to deal with division by zero 
    # (otherwise if there are 0 houses and, for example, 0 rental properties, 
    # there will be a RuntimeWarning)
    vehicleown.loc[vehicleown['AJ2WE001'] == 0.] = np.nan
    vehicleown.loc[vehicleown['AJXCE001'] == 0.] = np.nan    
    print('Census data read!')
    
    # # # Open state primary/secondary road shapefiles and records
    roads_primary, roads_secondary = [], []
    for FIPS_i in FIPS:
        shp = shapereader.Reader(DIR_SHAPEFILE+'roads/'+
            'tl_2019_%s_prisecroads/tl_2019_%s_prisecroads'%(FIPS_i,FIPS_i))
        roads = list(shp.geometries())
        roads_records = list(shp.records())   
        # Segregate primary versus secondary roads. Primary roads have 
        # attribute MTFCC==S1100. They are defined as limited-access highways
        # that connect to other roads only at interchanges and not at at-grade 
        # intersections. This category includes Interstate highways, as well 
        # as all other highways with limited access (some of which are toll
        # roads). Limited-access highways with only one lane in each direction, 
        # as well as those that are undivided, are also included under S1100.        
        # Secondary roads have attribute MTFCC==S1200 and are defined as 
        # main arteries that are not limited access, usually in the U.S. 
        # highway, state highway, or county highway systems. These roads have 
        # one or more lanes of traffic in each direction, may or may not be 
        # divided, and usually have at-grade intersections with many other 
        # roads and driveways. They often have both a local name and a route 
        # number. 
        # Note that the RTTYP attribute specifies whether a primary/secondary
        # road is a county (C), interstate (I), common name (M), other (O), 
        # state recognized (S), or U.S. (U) road. This doesn't seem to have to 
        # do with their size/importance but is a reflection of their name.
        # For example, in D.C. (FIPS 11), both the "Capital Beltway" and 
        # "DC Hwy 295" are primary roads, but the former is RTTYP==M and the
        # latter is RTTYP=I
        roads_mtfcc = [x.attributes['MTFCC'] for x in roads_records]
        where_primary = np.where(np.array(roads_mtfcc)=='S1100')[0]
        where_secondary = np.where(np.array(roads_mtfcc)=='S1200')[0]       
        # Add primary/secondary roads to multi-state list
        roads_primary += [roads[x] for x in where_primary]
        roads_secondary += [roads[x] for x in where_secondary]
    # Find centroid of roads 
    roads_primary_lat, roads_primary_lng = [], []
    roads_secondary_lat, roads_secondary_lng = [], []    
    for r in roads_primary: 
        roads_primary_lat.append(r.centroid.xy[1][0])
        roads_primary_lng.append(r.centroid.xy[0][0])
    for r in roads_secondary: 
        roads_secondary_lat.append(r.centroid.xy[1][0])
        roads_secondary_lng.append(r.centroid.xy[0][0])
    # Create DataFrame from road centroids for easy indexing
    roads_primary_centroids = pd.DataFrame({'RoadLat':roads_primary_lat,
        'RoadLng':roads_primary_lng})
    roads_secondary_centroids = pd.DataFrame({'RoadLat':roads_secondary_lat,
        'RoadLng':roads_secondary_lng})
    print('TIGER road data read!')

    # # # # Loop through census tracts and extract nearby road density 
    # and census information 
    print('Harmonizing census and road data for FIPS...')
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
            # Slice primary and secondary roads ~near tract center
            tract_roads_primary = roads_primary_centroids.loc[
                (roads_primary_centroids['RoadLat']>tract_intptlat-searchrad) &
                (roads_primary_centroids['RoadLat']<tract_intptlat+searchrad) & 
                (roads_primary_centroids['RoadLng']>tract_intptlon-searchrad) &
                (roads_primary_centroids['RoadLng']<tract_intptlon+searchrad)]
            tract_roads_secondary = roads_secondary_centroids.loc[
                (roads_secondary_centroids['RoadLat']>tract_intptlat-searchrad) &
                (roads_secondary_centroids['RoadLat']<tract_intptlat+searchrad) & 
                (roads_secondary_centroids['RoadLng']>tract_intptlon-searchrad) &
                (roads_secondary_centroids['RoadLng']<tract_intptlon+searchrad)]            
            # Loop through roads within search radius and calculate haversine 
            # distance
            dists_primary, dists_secondary = [], []
            for r in np.arange(0,len(tract_roads_primary),1):
                lon1, lat1, lon2, lat2 = map(np.radians, [
                    tract_roads_primary.iloc[r]['RoadLng'],
                    tract_roads_primary.iloc[r]['RoadLat'],
                    tract_intptlon, tract_intptlat])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = (np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * 
                    np.sin(dlon/2.0)**2)
                c = 2 * np.arcsin(np.sqrt(a))
                dists_primary.append(R * c)
            for r in np.arange(0,len(tract_roads_secondary),1):
                lon1, lat1, lon2, lat2 = map(np.radians, [
                    tract_roads_secondary.iloc[r]['RoadLng'],
                    tract_roads_secondary.iloc[r]['RoadLat'],
                    tract_intptlon, tract_intptlat])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = (np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * 
                    np.sin(dlon/2.0)**2)
                c = 2 * np.arcsin(np.sqrt(a))
                dists_secondary.append(R * c)                
            # Append information 
            geoids.append(geoid)
            frac_nocars.append((tract_vehicleown['AJ2WE003'].values[0]+
                tract_vehicleown['AJ2WE010'].values[0])/
                tract_vehicleown['AJ2WE001'].values[0])
            frac_rent.append(tract_vehicleown['AJ2WE009'].values[0]/
                tract_vehicleown['AJ2WE001'].values[0])
            frac_own.append(tract_vehicleown['AJ2WE002'].values[0]/
                tract_vehicleown['AJ2WE001'].values[0])
            frac_personal.append(tract_vehicleown['AJXCE002'].values[0]/
                tract_vehicleown['AJXCE001'].values[0])
            frac_public.append(tract_vehicleown['AJXCE010'].values[0]/
                tract_vehicleown['AJXCE001'].values[0])
            frac_bike.append(tract_vehicleown['AJXCE018'].values[0]/
                tract_vehicleown['AJXCE001'].values[0])
            frac_walk.append(tract_vehicleown['AJXCE019'].values[0]/
                tract_vehicleown['AJXCE001'].values[0])
            frac_other.append(tract_vehicleown['AJXCE020'].values[0]/
                tract_vehicleown['AJXCE001'].values[0])
            frac_home.append(tract_vehicleown['AJXCE021'].values[0]/
                tract_vehicleown['AJXCE001'].values[0])
            frac_taxi.append(tract_vehicleown['AJXCE016'].values[0]/
                tract_vehicleown['AJXCE001'].values[0])
            frac_motorcycle.append(tract_vehicleown['AJXCE017'].values[0]/
                tract_vehicleown['AJXCE001'].values[0])
            primary10.append(np.where(np.array(dists_primary)<10)[0].shape[0])
            primary5.append(np.where(np.array(dists_primary)<5)[0].shape[0])
            primary2.append(np.where(np.array(dists_primary)<2)[0].shape[0])
            primary1.append(np.where(np.array(dists_primary)<1)[0].shape[0])
            secondary10.append(np.where(np.array(dists_secondary)<10)[0].shape[0])
            secondary5.append(np.where(np.array(dists_secondary)<5)[0].shape[0])
            secondary2.append(np.where(np.array(dists_secondary)<2)[0].shape[0])
            secondary1.append(np.where(np.array(dists_secondary)<1)[0].shape[0])
    # Create output DataFrame
    road_density = pd.DataFrame({
        'GEOID':geoids, 
        'FracRent':frac_rent,
        'FracOwn':frac_own, 
        'FracNoCar':frac_nocars, 
        'FracPersonal':frac_personal,
        'FracPublic':frac_public,
        'FracBike':frac_bike,
        'FracWalk':frac_walk,
        'FracOther':frac_other,
        'FracHome':frac_home,
        'FracTaxi':frac_taxi,
        'FracMotorcycle':frac_motorcycle,
        'PrimaryWithin10':primary10,
        'PrimaryWithin5':primary5,
        'PrimaryWithin2':primary2,
        'PrimaryWithin1':primary1,
        'SecondaryWithin10':secondary10,
        'SecondaryWithin5':secondary5,
        'SecondaryWithin2':secondary2,
        'SecondaryWithin1':secondary1})
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
harmonized_vehicleownership_roaddensity(FIPS)



