#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open, parse, and merge harmonzied TROPOMI-U.S. Census Bureau/ACS data. Created 
on Fri Oct 23 20:19:08 2020

@author: ghkerr
"""

# For running locally
DIR = '/Users/ghkerr/GW/'
DIR_TROPOMI = DIR+'data/'
DIR_GEO = DIR+'data/geography/'
DIR_HARM = DIR+'data/census_no2_harmonzied/'
DIR_FIGS = DIR+'tropomi_ej/figs/'

def open_census_no2_harmonzied(FIPS): 
    """Open harmonized TROPOMI NO2 and census csv files for a given state/
    group of states. All columns (besides the column with the FIPS codes) are
    transformed to floats. An additional column with the absolute change of
    NO2 during the lockdown (spring 2020 vs. spring 2019) is added.

    Parameters
    ----------
    FIPS : list
        Contains FIPS code(s) for the state(s) of interest
        
    Returns
    -------
    state_harm : pandas.core.frame.DataFrame
        Harmonized tract-level TROPOMI NO2 and census data for state(s) of 
        interest
    """
    import pandas as pd
    # NHGIS census information version (may need to change if there are 
    # updates to census information)
    nhgis_version = '0003_ds239_20185_2018'
    # DataFrame that will be filled with harmonzied data for multiple states
    state_harm = pd.DataFrame()
    # Loop through states of interest and read in harmonized NO2/census data
    for FIPS_i in FIPS:
        state_harm_i = pd.read_csv(DIR_HARM+'Tropomi_NO2_%s_nhgis%s_tract.csv'
            %(FIPS_i, nhgis_version), delimiter=',', header=0, engine='python')
        # For states with FIPS codes 0-9, there is no leading zero in their 
        # GEOID row, so add one such that all GEOIDs for any particular state
        # are identical in length
        if FIPS_i in ['01','02','03','04','05','06','07','08','09']:
            state_harm_i['GEOID'] = state_harm_i['GEOID'].map(
            lambda x: f'{x:0>11}')
        # Make GEOID a string and index row 
        state_harm_i = state_harm_i.set_index('GEOID')
        state_harm_i.index = state_harm_i.index.map(str)
        # Make other columns floats
        for col in state_harm_i.columns:
            if col != 'FIPS':
                state_harm_i[col] = state_harm_i[col].astype(float)
        # Add column for percentage change in NO2 between pre- and post-COVID
        # periods
        state_harm_i['NO2_ABS'] = (state_harm_i['POSTNO2']-
            state_harm_i['PRENO2']) 
        state_harm = state_harm.append(state_harm_i)
    return state_harm

def merge_harmonized_vehicleownership(harmonized):
    """function opens census data on vehicle ownership and derived road density
    and merges it with harmonized TROPOMI-census data
    
    Parameters
    ----------
    harmonized : pandas.core.frame.DataFrame
        Harmonized tract-level TROPOMI NO2 and census data for state(s) of 
        interest

    Returns
    -------        
    harmonized_vehicle : pandas.core.frame.DataFrame
        Harmonized tract-level TROPOMI NO2 and census data for state(s) of 
        interest merged with vehicle ownership/road density data
    """
    import pandas as pd
    # Read in csv fiel for vehicle ownership/road density
    vehicle = pd.read_csv(DIR_HARM+'vehicleownership_roaddensity_us.csv', 
        delimiter=',', header=0, engine='python')
    # Leading 0 for state FIPS codes < 10 
    vehicle['GEOID'] = vehicle['GEOID'].map(lambda x: f'{x:0>11}')
    # Make GEOID a string and index row 
    vehicle = vehicle.set_index('GEOID')
    vehicle.index = vehicle.index.map(str)
    # Make other columns floats
    for col in vehicle.columns:
        vehicle[col] = vehicle[col].astype(float)
    # Merge with harmonized census data
    harmonized_vehicle = harmonized.merge(vehicle, left_index=True, 
        right_index=True)
    return harmonized_vehicle

def split_harmonized_byruralurban(harmonized):
    """Fuction loads a county-based rural/urban look up (see https://www.
    google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=
    2ahUKEwiqjb_27-TqAhUxgnIEHSLMBGgQFjABegQIAhAB&url=https%3A%2F%2Fwww2.
    census.gov%2Fgeo%2Fdocs%2Freference%2Fua%2FCounty_Rural_Lookup.xlsx&usg=
    AOvVaw2AYNpyKzfdR2BOTcIOAOzp and 
    https://www2.census.gov/geo/docs/reference/ua/) and determines which 
    counties have less than 5% of their population designated as rural (n.b., 
    this threshold should be changed to see the effect it has on results).
    The tract GEOIDS within the urban and rural counties are subsequently 
    sliced from DataFrame. 

    Parameters
    ----------
    harmonized : pandas.core.frame.DataFrame
        Harmonized tract-level TROPOMI NO2 and census data for state(s) of 
        interest

    Returns
    -------        
    harmonized_urban : pandas.core.frame.DataFrame
        Harmonized tract-level TROPOMI NO2 and census data for urban tracts in 
        state(s) of interest
    harmonized_rural : pandas.core.frame.DataFrame
        Harmonized tract-level TROPOMI NO2 and census data for rural tracts in 
        state(s) of interest        
    """
    import numpy as np
    import pandas as pd
    rural_lookup = pd.read_csv(DIR_HARM+'County_Rural_Lookup.csv', 
        delimiter=',', header=0, engine='python')
    # Designate counties that have a rural population < 5% as urbans
    urbancounties = rural_lookup[rural_lookup
        ['2010 Census \nPercent Rural'] < 5.]
    # Transform 2015 GEOID column so the length matches the GEOIDs in 
    # harmonized data
    urbancounties = [(str(x).zfill(5)) for x in urbancounties['2015 GEOID']]
    # Find tracts considered urban
    urbantracts = []
    for x in np.arange(0, len(harmonized), 1):    
        GEOID_statecountyonly = np.str(harmonized.index[x])[:5]
        if GEOID_statecountyonly in urbancounties:
            urbantracts.append(harmonized.index[x])
    # Sample urban and rural tracts 
    harmonized_urban = harmonized.loc[harmonized.index.isin(urbantracts)]
    harmonized_rural = harmonized.loc[~harmonized.index.isin(urbantracts)]
    return harmonized_urban, harmonized_rural

def subset_harmonized_bycountyfips(harmonized, fips_interest):
    """Subset harmonized dataset for counties of interest.

    Parameters
    ----------
    harmonized : pandas.core.frame.DataFrame
        Harmonized tract-level TROPOMI NO2 and census data for state(s) of 
        interest
    fips_interest : list
        FIPS codes (state/county = SSCCC) for counties of interest. Codes 
        must be strings. 

    Returns
    -------
    harmonized_fips_interest : pandas.core.frame.DataFrame
        Harmonized data only for counties of interest
    """
    geoids = harmonized.index.values
    geoids_fips_interest = []
    for prefix in fips_interest: 
        in_fips_interest = [x for x in geoids if x.startswith(prefix)]
        geoids_fips_interest.append(in_fips_interest)
    geoids_fips_interest = sum(geoids_fips_interest, [])
    # Slice harmonized dataset to include only counties/cities of interest
    harmonized_fips_interest = harmonized.loc[harmonized.index.isin(
        geoids_fips_interest)]
    return harmonized_fips_interest