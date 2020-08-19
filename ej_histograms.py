#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 15:21:23 2020

@author: ghkerr
"""
DIR = '/Users/ghkerr/GW/'
DIR_TROPOMI = DIR+'data/'
DIR_GEO = DIR+'data/geography/'
DIR_HARM = DIR+'data/census_no2_harmonzied/'
DIR_FIGS = DIR+'tropomi_ej/figs/'
DIR_CEMS = DIR+'data/emissions/CEMS/'

# Constants
ptile_upper = 90
ptile_lower = 10
missingdata = 'darkgrey'

# import netCDF4 as nc
# # 13 March - 13 June 2019 and 2020 average NO2
# no2_pre_dg = nc.Dataset(DIR_TROPOMI+
#     'Tropomi_NO2_griddedon0.01grid_Mar13-Jun132019_precovid19_QA75.ncf')
# no2_pre_dg = no2_pre_dg['NO2'][:]
# no2_post_dg = nc.Dataset(DIR_TROPOMI+
#     'Tropomi_NO2_griddedon0.01grid_Mar13-Jun132020_postcovid19_QA75.ncf')
# no2_post_dg = no2_post_dg['NO2'][:].data
# # 1 April - 30 June 2019 and 2020 average NO2
# no2_preapr_dg = nc.Dataset(DIR_TROPOMI+
#     'Tropomi_NO2_griddedon0.01grid_Apr01-Jun302019_precovid19_QA75.ncf')
# no2_preapr_dg = no2_preapr_dg['NO2'][:]
# no2_postapr_dg = nc.Dataset(DIR_TROPOMI+
#     'Tropomi_NO2_griddedon0.01grid_Apr01-Jun302020_precovid19_QA75.ncf')
# no2_postapr_dg = no2_postapr_dg['NO2'][:].data
# no2_all_dg = nc.Dataset(DIR_TROPOMI+
#     'Tropomi_NO2_griddedon0.01grid_allyears_QA75.ncf')
# no2_all_dg = no2_all_dg['NO2'][:].data
# lat_dg = nc.Dataset(DIR_TROPOMI+'LatLonGrid.ncf')['LAT'][:].data
# lng_dg = nc.Dataset(DIR_TROPOMI+'LatLonGrid.ncf')['LON'][:].data

def open_census_no2_harmonzied(FIPS): 
    """Open harmonized TROPOMI NO2 and census csv files for a given state/
    group of states. All columns (besides the column with the FIPS codes) are
    transformed to floats. An additional column with the percentage change of
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
    import numpy as np
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
        state_harm_i['NO2_PC'] = ((state_harm_i['POSTNO2']-
            state_harm_i['PRENO2'])/state_harm_i['PRENO2'])*100.
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

def open_cems(FIPS, counties=None):
    """Open facility-level emissions for a state (and counties within those
    states of interest)
    
    Parameters
    ----------
    FIPS : list
        Contains FIPS code(s) for the state(s) of interest
    counties : list, optional
        County names of interest
        
    Returns
    -------
    emissions : pandas.core.frame.DataFrame
        CEMS facility-level emissions for states and counties of interest
    """
    import numpy as np
    import pandas as pd
    # DataFrame that will be filled with emissionsfor multiple states
    emissions = pd.DataFrame()
    # Loop through states of interest and read in emissions/facility data
    for FIPS_i in FIPS:
        state_emission = pd.read_csv(DIR_CEMS+'%s/'%FIPS_i+
            'emission_%s_2019-2020.csv'%FIPS_i, delimiter=',', header=0,
            engine='python', index_col=False)
        state_facility = pd.read_csv(DIR_CEMS+'%s/'%FIPS_i+
            'facility_%s_2019-2020.csv'%FIPS_i, delimiter=',', header=0, 
            engine='python', index_col=False)
        emissions = emissions.append(state_emission)
    if counties:
        # Select counties of interest
        emissions = emissions.loc[emissions[' County'].isin(counties)]
    emissions[' Date'] = emissions[' Date'].apply(pd.to_datetime)        
    return emissions

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

def harmonized_demographics(harmonized, ptile_upper, ptile_lower):
    """Calculate demographics (race, income, educational attainment) for all 
    census tracts and tracts with the lowest (<10th percentile) and highest 
    (>90th percentile) historic (i.e., 2019) NO2 pollution and largest (<10th 
    percentile) and smallest (>90th percentile) gains during lockdowns. 
    
    Parameters
    ----------
    harmonized : pandas.core.frame.DataFrame
        Harmonized tract-level TROPOMI NO2 and census data for state(s) of 
        interest
    ptile_upper : int
        The upper percentile (e.g., 80th or 90th) above which historic NO2 is 
        classified as "most" and COVID-related changes in NO2 are classified 
        as "largest"
    ptile_lower : int
        The upper percentile (e.g., 10th or 20th) below which historic NO2 is 
        classified as "least" and COVID-related changes in NO2 are classified 
        as "smallest"

    Returns
    -------
    demographdict : dict  
        Average demographics (race, income, educational attainment) for 
        highest and lowest historic NO2 and largest and smallest gains during 
        lockdowns
    mostno2 : pandas.core.frame.DataFrame
        Census tracts with the highest historic NO2
    leastno2 : pandas.core.frame.DataFrame
        Census tracts with the lowest historic NO2    
    increaseno2 : pandas.core.frame.DataFrame
        Census tracts with the smallest gains during lockdowns
    decreaseno2 : pandas.core.frame.DataFrame
        Census tracts with the largest gains during lockdowns    
    """
    import numpy as np
    # Tracts with the most and least NO2 pollution pre-lockdown (to check 
    # baseline)
    mostno2 = harmonized.loc[harmonized['PRENO2'] >
        np.nanpercentile(harmonized['PRENO2'], ptile_upper)]
    leastno2 = harmonized.loc[harmonized['PRENO2']<
        np.nanpercentile(harmonized['PRENO2'], ptile_lower)]
    # Largest increases/decreases in NO2 during lockdown 
    increaseno2 = harmonized.loc[harmonized['NO2_ABS']>
        np.nanpercentile(harmonized['NO2_ABS'], ptile_upper)]
    decreaseno2 = harmonized.loc[harmonized['NO2_ABS']<
        np.nanpercentile(harmonized['NO2_ABS'], ptile_lower)]
    # Calculate mean values
    no2_mean = np.nanmean(harmonized['PRENO2'])
    gains_mean = np.nanmean(harmonized['NO2_ABS'])
    income_mean = np.nanmean(harmonized['AJZAE001'])
    white_mean = np.nanmean(harmonized['AJWNE002']/harmonized['AJWBE001'])
    black_mean = np.nanmean(harmonized['AJWNE003']/harmonized['AJWBE001'])     
    otherrace_mean = np.nanmean((harmonized['AJWNE004']+harmonized['AJWNE005']+
        harmonized['AJWNE006']+harmonized['AJWNE007']+harmonized['AJWNE008']
        )/harmonized['AJWBE001'])
    second_mean = np.nanmean(
        harmonized.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        harmonized['AJYPE001'])
    college_mean = np.nanmean(
        harmonized.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        harmonized['AJYPE001'])
    grad_mean = np.nanmean(
        harmonized.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        harmonized['AJYPE001'])
    hispanic_mean = np.nanmean(harmonized['AJWWE003']/
        harmonized['AJWWE001'])
    nonhispanic_mean = np.nanmean(harmonized['AJWWE002']/
        harmonized['AJWWE001'])    
    insure_mean = np.nanmean((
        harmonized['AJ35E003']+harmonized['AJ35E010']+
        harmonized['AJ35E019']+harmonized['AJ35E026']+
        harmonized['AJ35E035']+harmonized['AJ35E042']+
        harmonized['AJ35E052']+harmonized['AJ35E058'])/
        harmonized['AJ35E001'])
    noninsure_mean = np.nanmean((
        harmonized['AJ35E017']+harmonized['AJ35E033']+
        harmonized['AJ35E050']+harmonized['AJ35E066'])/
        harmonized['AJ35E001'])    
    male_mean = np.nanmean(harmonized['AJWBE002']/harmonized['AJWBE001'])
    female_mean = np.nanmean(harmonized['AJWBE026']/harmonized['AJWBE001'])
    age_mean = np.nanmean(harmonized['AJWCE001'])
    publicassistance_mean = np.nanmean(harmonized['AJZVE002']/
        harmonized['AJZVE001'])
    nopublicassistance_mean = np.nanmean(harmonized['AJZVE003']/
        harmonized['AJZVE001'])
    # # # # Median income 
    # Baseline
    income_mostno2 = np.nanmean(mostno2['AJZAE001'])
    income_leastno2 = np.nanmean(leastno2['AJZAE001'])
    # Lockdown
    income_increaseno2 = np.nanmean(increaseno2['AJZAE001'])
    income_decreaseno2 = np.nanmean(decreaseno2['AJZAE001'])
    # # # # # Race
    # Baseline
    white_mostno2 = np.nanmean(mostno2['AJWNE002']/mostno2['AJWBE001'])
    black_mostno2 = np.nanmean(mostno2['AJWNE003']/mostno2['AJWBE001'])
    otherrace_mostno2 = np.nanmean((mostno2['AJWNE004']+mostno2['AJWNE005']+
        mostno2['AJWNE006']+mostno2['AJWNE007']+mostno2['AJWNE008'])/
        mostno2['AJWBE001'])
    white_leastno2 = np.nanmean(leastno2['AJWNE002']/leastno2['AJWBE001'])
    black_leastno2 = np.nanmean(leastno2['AJWNE003']/leastno2['AJWBE001'])
    otherrace_leastno2 = np.nanmean((leastno2['AJWNE004']+leastno2['AJWNE005']+
        leastno2['AJWNE006']+leastno2['AJWNE007']+leastno2['AJWNE008'])/
        leastno2['AJWBE001'])
    # Lockdown
    white_increaseno2 = np.nanmean(increaseno2['AJWNE002']/
        increaseno2['AJWBE001'])
    black_increaseno2 = np.nanmean(increaseno2['AJWNE003']/
        increaseno2['AJWBE001'])
    otherrace_increaseno2 = np.nanmean((increaseno2['AJWNE004']+
        increaseno2['AJWNE005']+increaseno2['AJWNE006']+
        increaseno2['AJWNE007']+increaseno2['AJWNE008'])/
        increaseno2['AJWBE001'])
    white_decreaseno2 = np.nanmean(decreaseno2['AJWNE002']/
        decreaseno2['AJWBE001'])
    black_decreaseno2 = np.nanmean(decreaseno2['AJWNE003']/
        decreaseno2['AJWBE001'])
    otherrace_decreaseno2 = np.nanmean((decreaseno2['AJWNE004']+
        decreaseno2['AJWNE005']+decreaseno2['AJWNE006']+
        decreaseno2['AJWNE007']+decreaseno2['AJWNE008'])/
        decreaseno2['AJWBE001'])
    # # # # Educational attainment; n.b. "second" stands for secondary and 
    # includes high school/GED or less education, "college" stands for 
    # post-secondary and includes some college or college, and "grad" stands 
    # for graduate school/professional school
    # Baseline
    second_mostno2 = np.nanmean(
        mostno2.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        mostno2['AJYPE001'])
    college_mostno2 = np.nanmean(
        mostno2.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        mostno2['AJYPE001'])
    grad_mostno2 = np.nanmean(
        mostno2.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        mostno2['AJYPE001'])
    second_leastno2 = np.nanmean(
        leastno2.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        leastno2['AJYPE001'])
    college_leastno2 = np.nanmean(
        leastno2.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        leastno2['AJYPE001'])
    grad_leastno2 = np.nanmean(
        leastno2.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        leastno2['AJYPE001'])
    second_increaseno2 = np.nanmean(
        increaseno2.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        increaseno2['AJYPE001'])
    college_increaseno2 = np.nanmean(
        increaseno2.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        increaseno2['AJYPE001'])
    grad_increaseno2 = np.nanmean(
        increaseno2.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        increaseno2['AJYPE001'])
    second_decreaseno2 = np.nanmean(
        decreaseno2.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        decreaseno2['AJYPE001'])
    college_decreaseno2 = np.nanmean(
        decreaseno2.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        decreaseno2['AJYPE001'])
    grad_decreaseno2 = np.nanmean(
        decreaseno2.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        decreaseno2['AJYPE001'])
    # # # # Hispanic
    # Baseline
    hispanic_mostno2 = np.nanmean(mostno2['AJWWE003']/
        mostno2['AJWWE001'])
    nonhispanic_mostno2 = np.nanmean(mostno2['AJWWE002']/
        mostno2['AJWWE001'])
    hispanic_leastno2 = np.nanmean(leastno2['AJWWE003']/
        leastno2['AJWWE001'])
    nonhispanic_leastno2 = np.nanmean(leastno2['AJWWE002']/
        leastno2['AJWWE001'])
    # Lockdown
    hispanic_no2increase = np.nanmean(increaseno2['AJWWE003']/
        increaseno2['AJWWE001'])
    nonhispanic_no2increase = np.nanmean(increaseno2['AJWWE002']/
        increaseno2['AJWWE001'])
    hispanic_no2decrease = np.nanmean(decreaseno2['AJWWE003']/
        decreaseno2['AJWWE001'])
    nonhispanic_no2decrease = np.nanmean(decreaseno2['AJWWE002']/
        decreaseno2['AJWWE001'])
    # Insurance status; n.b. that many of the columns that specify that types 
    # of insurance coverage (e.g., employer-based, direct-purchase, Medicare, 
    # etc.) further subdivide the broad "With one type of health insurance 
    # coverage" or "With two or more types of health insurance coverage,"
    # so they're not really needed
    # Baseline
    insure_mostno2 = np.nanmean((mostno2['AJ35E003']+mostno2['AJ35E010']+
        mostno2['AJ35E019']+mostno2['AJ35E026']+mostno2['AJ35E035']+
        mostno2['AJ35E042']+mostno2['AJ35E052']+mostno2['AJ35E058'])/
        mostno2['AJ35E001'])
    noinsure_mostno2 = np.nanmean((mostno2['AJ35E017']+mostno2['AJ35E033']+
        mostno2['AJ35E050']+mostno2['AJ35E066'])/mostno2['AJ35E001'])
    insure_leastno2 = np.nanmean((leastno2['AJ35E003']+leastno2['AJ35E010']+
        leastno2['AJ35E019']+leastno2['AJ35E026']+leastno2['AJ35E035']+
        leastno2['AJ35E042']+leastno2['AJ35E052']+leastno2['AJ35E058'])/
        leastno2['AJ35E001'])
    noinsure_leastno2 = np.nanmean((leastno2['AJ35E017']+leastno2['AJ35E033']+
        leastno2['AJ35E050']+leastno2['AJ35E066'])/leastno2['AJ35E001'])
    # Lockdown
    insure_no2increase = np.nanmean((
        increaseno2['AJ35E003']+increaseno2['AJ35E010']+
        increaseno2['AJ35E019']+increaseno2['AJ35E026']+
        increaseno2['AJ35E035']+increaseno2['AJ35E042']+
        increaseno2['AJ35E052']+increaseno2['AJ35E058'])/
        increaseno2['AJ35E001'])
    noinsure_no2increase = np.nanmean((
        increaseno2['AJ35E017']+increaseno2['AJ35E033']+
        increaseno2['AJ35E050']+increaseno2['AJ35E066'])/
        increaseno2['AJ35E001'])
    insure_no2decrease = np.nanmean((
        decreaseno2['AJ35E003']+decreaseno2['AJ35E010']+
        decreaseno2['AJ35E019']+decreaseno2['AJ35E026']+
        decreaseno2['AJ35E035']+decreaseno2['AJ35E042']+
        decreaseno2['AJ35E052']+decreaseno2['AJ35E058'])/
        decreaseno2['AJ35E001'])
    noinsure_no2decrease = np.nanmean((
        decreaseno2['AJ35E017']+decreaseno2['AJ35E033']+
        decreaseno2['AJ35E050']+decreaseno2['AJ35E066'])/
        decreaseno2['AJ35E001'])
    # # # # Sex
    # Baseline
    male_mostno2 = np.nanmean(mostno2['AJWBE002']/mostno2['AJWBE001'])
    female_mostno2 = np.nanmean(mostno2['AJWBE026']/mostno2['AJWBE001'])
    male_leastno2 = np.nanmean(leastno2['AJWBE002']/leastno2['AJWBE001'])
    female_leastno2 = np.nanmean(leastno2['AJWBE026']/leastno2['AJWBE001'])
    # Lockdown
    male_no2increase = np.nanmean(increaseno2['AJWBE002']/
        increaseno2['AJWBE001'])
    female_no2increase = np.nanmean(increaseno2['AJWBE026']/
        increaseno2['AJWBE001'])
    male_no2decrease = np.nanmean(decreaseno2['AJWBE002']/
        decreaseno2['AJWBE001'])
    female_no2decrease = np.nanmean(decreaseno2['AJWBE026']/
        decreaseno2['AJWBE001'])
    # # # # # Median age
    # Baseline
    age_mostno2 = np.nanmean(mostno2['AJWCE001'])
    age_leastno2 = np.nanmean(leastno2['AJWCE001'])
    # Lockdown
    age_no2increase = np.nanmean(increaseno2['AJWCE001'])
    age_no2decrease = np.nanmean(decreaseno2['AJWCE001'])
    # # # # Public Assistance Income or Food Stamps/SNAP in the Past 12 Months 
    # for Households
    # Baseline 
    pubass_mostno2 = np.nanmean(mostno2['AJZVE002']/mostno2['AJZVE001'])
    nopubass_mostno2 = np.nanmean(mostno2['AJZVE003']/mostno2['AJZVE001'])
    pubass_leastno2 = np.nanmean(leastno2['AJZVE002']/leastno2['AJZVE001'])
    nopubass_leastno2 = np.nanmean(leastno2['AJZVE003']/leastno2['AJZVE001'])
    # Lockdown 
    pubass_no2increase = np.nanmean(increaseno2['AJZVE002']/
        increaseno2['AJZVE001'])
    nopubass_no2increase = np.nanmean(increaseno2['AJZVE003']/
        increaseno2['AJZVE001'])
    pubass_no2decrease = np.nanmean(decreaseno2['AJZVE002']/
        decreaseno2['AJZVE001'])
    nopubass_no2decrease = np.nanmean(decreaseno2['AJZVE003']/
        decreaseno2['AJZVE001'])
    # Create dictionary with output values 
    demographdict = {'2019no2_mean':no2_mean,
        'most2019no2':np.nanmean(mostno2['PRENO2'].values),
        'least2019no2':np.nanmean(leastno2['PRENO2'].values),
        'gains_mean':gains_mean,
        'smallestgains':np.nanmean(increaseno2['NO2_ABS'].values),
        'largestgains':np.nanmean(decreaseno2['NO2_ABS'].values),
        'income_mean':income_mean,
        'white_mean':white_mean*100.,
        'black_mean':black_mean*100.,
        'otherrace_mean':otherrace_mean*100.,
        'secondary_mean':second_mean*100.,
        'college_mean':college_mean*100.,
        'grad_mean':grad_mean*100.,
        'hispanic_mean':hispanic_mean*100.,
        'nonhispanic_mean':nonhispanic_mean*100.,
        'insure_mean':insure_mean*100.,
        'noninsure_mean':noninsure_mean*100.,
        'male_mean':male_mean*100.,
        'female_mean':female_mean*100.,
        'age_mean':age_mean,
        'publicassistance_mean':publicassistance_mean*100.,
        'nopublicassistance_mean':nopublicassistance_mean*100.,
        'income_most2019no2':income_mostno2,
        'income_least2019no2':income_leastno2,
        'income_largestgains':income_decreaseno2,
        'income_smallestgains':income_increaseno2,
        'white_most2019no2':white_mostno2*100.,
        'black_most2019no2':black_mostno2*100.,
        'otherrace_most2019no2':otherrace_mostno2*100.,
        'white_least2019no2':white_leastno2*100.,
        'black_least2019no2':black_leastno2*100.,
        'otherrace_least2019no2':otherrace_leastno2*100.,
        'white_smallestgains':white_increaseno2*100.,
        'black_smallestgains':black_increaseno2*100.,
        'otherrace_smallestgains':otherrace_increaseno2*100.,
        'white_largestgains':white_decreaseno2*100.,
        'black_largestgains':black_decreaseno2*100.,
        'otherrace_largestgains':otherrace_decreaseno2*100.,
        'secondary_most2019no2':second_mostno2*100.,
        'college_most2019no2':college_mostno2*100.,
        'grad_most2019no2':grad_mostno2*100.,
        'secondary_least2019no2':second_leastno2*100.,
        'college_least2019no2':college_leastno2*100.,
        'grad_least2019no2':grad_leastno2*100.,
        'secondary_smallestgains':second_increaseno2*100.,
        'college_smallestgains':college_increaseno2*100.,
        'grad_smallestgains':grad_increaseno2*100.,
        'secondary_largestgains':second_decreaseno2*100.,
        'college_largestgains':college_decreaseno2*100.,
        'grad_largestgains':grad_decreaseno2*100.,
        'hispanic_most2019no2':hispanic_mostno2*100.,
        'nonhispanic_most2019no2':nonhispanic_mostno2*100.,
        'hispanic_least2019no2':hispanic_leastno2*100.,
        'nonhispanic_least2019no2':nonhispanic_leastno2*100.,
        'hispanic_smallestgains':hispanic_no2increase*100.,
        'nonhispanic_smallestgains':nonhispanic_no2increase*100.,
        'hispanic_largestgains':hispanic_no2decrease*100.,
        'nonhispanic_largestgains':nonhispanic_no2decrease*100.,
        'insure_most2019no2':insure_mostno2*100.,
        'noninsure_most2019no2':noinsure_mostno2*100.,
        'insure_least2019no2':insure_leastno2*100.,
        'noninsure_least2019no2':noinsure_leastno2*100.,
        'insure_smallestgains':insure_no2increase*100.,
        'noninsure_smallestgains':noinsure_no2increase*100.,
        'insure_largestgains':insure_no2decrease*100.,
        'noninsure_largestgains':noinsure_no2decrease*100.,
        'male_most2019no2':male_mostno2*100.,
        'female_most2019no2':female_mostno2*100.,
        'male_least2019no2':male_leastno2*100.,
        'female_least2019no2':female_leastno2*100.,
        'male_smallestgains':male_no2increase*100.,
        'female_smallestgains':female_no2increase*100.,
        'male_largestgains':male_no2decrease*100.,
        'female_largestgains':female_no2decrease*100.,
        'age_most2019no2':age_mostno2,
        'age_least2019no2':age_leastno2,
        'age_smallestgains':age_no2increase,
        'age_largestgains':age_no2decrease,
        'publicassistance_most2019no2':pubass_mostno2*100.,
        'nopublicassistance_most2019no2':nopubass_mostno2*100.,
        'publicassistance_least2019no2':pubass_leastno2*100.,
        'nopublicassistance_least2019no2':nopubass_leastno2*100.,
        'publicassistance_smallestgains':pubass_no2increase*100.,
        'nopublicassistance_smallestgains':nopubass_no2increase*100.,        
        'publicassistance_largestgains':pubass_no2decrease*100.,
        'nopublicassistance_largestgains':nopubass_no2decrease*100.}        
    return demographdict, mostno2, leastno2, increaseno2, decreaseno2

def demographic_summarytable(demography, ofilename):
    """For a given state or collection of states (with values contained in 
    "state_harm" DataFrame), plot summary data of race, sex, income, 
    education, etc. for census tracts with the most and least NO2 pre-lockdown
    and largest increases and decreases of NO2 during the lockdown

    Parameters
    ----------
    demography : dict
        Average demographics (race, income, educational attainment) for 
        highest and lowest historic NO2 and largest and smallest gains during 
        lockdowns    
    ofilename : str
        Output file name (should be FIPS code or some description like 
        "conus")
    
    Returns
    -------
    None    
    """
    import numpy as np
    import matplotlib.pyplot as plt
    col_labels = ['Highest historic NO$_{2}$', 'Lowest historic NO$_{2}$', 
      'Smallest gains', 'Largest gains']
    row_labels = ['NO$_{2}$ (%s/%.1f) [molec cm$^{-2}$/%%]'
         %('{:.2e}'.format(demography['2019no2_mean']), demography['gains_mean']),
        'Median income (%.1f) [$]'%demography['income_mean'],
        'Male (%.1f) [%%]'%demography['male_mean'],
        'Female (%.1f) [%%]'%demography['female_mean'],
        'Age (%.1f) [years]'%demography['age_mean'],
        'Race: white (%.1f) [%%]'%demography['white_mean'],
        'Race: black (%.1f) [%%]'%demography['black_mean'],
        'Race: other (%.1f) [%%]'%demography['otherrace_mean'],
        'Hispanic (%.1f) [%%]'%demography['hispanic_mean'],
        'Non-hispanic (%.1f) [%%]'%demography['nonhispanic_mean'],
        'Education: secondary or less (%.1f) [%%]'%demography['secondary_mean'],
        'Education: college (%.1f) [%%]'%demography['college_mean'],
        'Education: graduate or professional (%.1f) [%%]'%demography['grad_mean'],    
        'Health insurance (%.1f) [%%]'%demography['insure_mean'],
        'No health insurance (%.1f) [%%]'%demography['noninsure_mean'],
        'Public assistance (%.1f) [%%]'%demography['publicassistance_mean'],
        'No public assistance (%.1f) [%%]'%demography['nopublicassistance_mean']
        ]
    # Table of values 
    table_vals = [
        # NO2
        ['{:.2e}'.format(demography['most2019no2']),
          '{:.2e}'.format(demography['least2019no2']),
          np.round(demography['smallestgains'], 1),
          np.round(demography['largestgains'], 1)],
        # Income
        [np.round(demography['income_most2019no2'], 2), 
          np.round(demography['income_least2019no2'], 2), 
          np.round(demography['income_smallestgains'], 2), 
          np.round(demography['income_largestgains'], 2)],
        # Male
        [np.round(demography['male_most2019no2'], 1),
          np.round(demography['male_least2019no2'], 1), 
          np.round(demography['male_smallestgains'], 1),
          np.round(demography['male_largestgains'], 1)],
        # Female
        [np.round(demography['female_most2019no2'], 1),
          np.round(demography['female_least2019no2'], 1), 
          np.round(demography['female_smallestgains'], 1),
          np.round(demography['female_largestgains'], 1)],
        # Age
        [np.round(demography['age_most2019no2'], 1), 
          np.round(demography['age_least2019no2'], 1), 
          np.round(demography['age_smallestgains'], 1), 
          np.round(demography['age_largestgains'], 1)],
        # Race: white
        [np.round(demography['white_most2019no2'], 1), 
          np.round(demography['white_least2019no2'], 1), 
          np.round(demography['white_smallestgains'], 1), 
          np.round(demography['white_largestgains'], 1)],
        # Race: black/African American 
        [np.round(demography['black_most2019no2'], 1),
          np.round(demography['black_least2019no2'], 1), 
          np.round(demography['black_smallestgains'], 1), 
          np.round(demography['black_largestgains'], 1)],
        # Race: other
        [np.round(demography['otherrace_most2019no2'], 1), 
          np.round(demography['otherrace_least2019no2'], 1), 
          np.round(demography['otherrace_smallestgains'], 1), 
          np.round(demography['otherrace_largestgains'], 1)],
        # Hispanic
        [np.round(demography['hispanic_most2019no2'], 1),
          np.round(demography['hispanic_least2019no2'], 1),
          np.round(demography['hispanic_smallestgains'], 1),
          np.round(demography['hispanic_largestgains'], 1)], 
        # Non-hispanic
        [np.round(demography['nonhispanic_most2019no2'], 1), 
          np.round(demography['nonhispanic_least2019no2'], 1), 
          np.round(demography['nonhispanic_smallestgains'], 1), 
          np.round(demography['nonhispanic_largestgains'], 1)],
        # Education: secondary or less
        [np.round(demography['secondary_most2019no2'], 1), 
          np.round(demography['secondary_least2019no2'], 1), 
          np.round(demography['secondary_smallestgains'], 1),
          np.round(demography['secondary_largestgains'], 1)],
        # Education: college
        [np.round(demography['college_most2019no2'], 1), 
          np.round(demography['college_least2019no2'], 1),
          np.round(demography['college_smallestgains'], 1), 
          np.round(demography['college_largestgains'], 1)],
        # Education: graduate
        [np.round(demography['grad_most2019no2'], 1), 
          np.round(demography['grad_least2019no2'], 1),
          np.round(demography['grad_smallestgains'], 1), 
          np.round(demography['grad_largestgains'], 1)],    
        # Health insurance
        [np.round(demography['insure_most2019no2'], 1), 
          np.round(demography['insure_least2019no2'], 1), 
          np.round(demography['insure_smallestgains'], 1),
          np.round(demography['insure_largestgains'], 1)],
        # No health insurance
        [np.round(demography['noninsure_most2019no2'], 1), 
          np.round(demography['noninsure_least2019no2'], 1),
          np.round(demography['noninsure_smallestgains'], 1), 
          np.round(demography['noninsure_largestgains'], 1)],
        # Public assistance
        [np.round(demography['publicassistance_most2019no2'], 1), 
          np.round(demography['publicassistance_least2019no2'], 1),
          np.round(demography['publicassistance_smallestgains'], 1), 
          np.round(demography['publicassistance_largestgains'], 1)],
        # No public assistance
        [np.round(demography['nopublicassistance_most2019no2'], 1), 
          np.round(demography['nopublicassistance_least2019no2'], 1),
          np.round(demography['nopublicassistance_smallestgains'], 1), 
          np.round(demography['nopublicassistance_largestgains'], 1)],
        ]
    # Draw table
    hcell, wcell = 0.2, 2.7 #May need tweaking!
    fig = plt.figure(figsize=(len(col_labels)*wcell, 
        len(table_vals)*hcell))    
    ax = plt.subplot2grid((1,1),(0,0))    
    the_table = ax.table(cellText=table_vals, colWidths=[0.25]*4,
        rowLabels=row_labels, colLabels=col_labels, loc='center')
    # Removing ticks and spines enables you to get the figure only with table
    ax.tick_params(axis='x', which='both', bottom=False, top=False, 
        labelbottom=False)
    ax.tick_params(axis='y', which='both', right=False, left=False, 
        labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    the_table.auto_set_font_size(False)
    plt.subplots_adjust(left=0.4, right=0.95)
    plt.savefig(DIR_FIGS+'demographic_summarytable_%s.png'%ofilename, 
        dpi=500)
    plt.show()
    return

def NO2_gridded_tractavg(lng_dg, lat_dg, no2_pre_dg, no2_post_dg, FIPS, 
    harmonized):
    """XXX
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import netCDF4 as nc
    import numpy as np
    import matplotlib as mpl
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import shapereader
    proj = ccrs.PlateCarree(central_longitude=0.0)
    fig = plt.figure(figsize=(8,4))
    ax1 = plt.subplot2grid((2,2),(0,0), projection=ccrs.PlateCarree(
        central_longitude=0.))
    ax2 = plt.subplot2grid((2,2),(0,1), projection=ccrs.PlateCarree(
        central_longitude=0.))
    ax3 = plt.subplot2grid((2,2),(1,0), projection=ccrs.PlateCarree(
        central_longitude=0.))
    ax4 = plt.subplot2grid((2,2),(1,1), projection=ccrs.PlateCarree(
        central_longitude=0.))
    # Create discrete colormaps
    cmapbase = plt.get_cmap('YlGnBu', 12)
    normbase = matplotlib.colors.Normalize(vmin=0e15, vmax=6e15)
    cmaplock = plt.get_cmap('coolwarm', 10)
    normlock = matplotlib.colors.Normalize(vmin=-50, vmax=50)
    # Plot gridded NO2 retrivals 
    mb1 = ax1.pcolormesh(lng_dg, lat_dg, no2_pre_dg, cmap=cmapbase, 
        norm=normbase, transform=proj, rasterized=True)
    mb2 = ax2.pcolormesh(lng_dg, lat_dg, (no2_post_dg-no2_pre_dg)/no2_pre_dg*100., 
        cmap=cmaplock, norm=normlock, transform=proj, rasterized=True)
    for FIPS_i in FIPS: 
        print(FIPS_i)
        # Tigerline shapefile for state
        shp = shapereader.Reader(DIR_GEO+'tigerline/tl_2019_%s_tract/tl_2019_%s_tract'%(
            FIPS_i, FIPS_i))
        records = shp.records()
        tracts = shp.geometries()
        for record, tract in zip(records, tracts):
            # Find GEOID of tract
            gi = record.attributes['GEOID']
            # Look up harmonized NO2-census data for tract
            harmonized_tract = harmonized.loc[harmonized.index.isin([gi])]
            baseline = harmonized_tract.PRENO2.values[0]
            lockdown = harmonized_tract.NO2_PC.values[0]
            if np.isnan(baseline)==True:
                ax3.add_geometries(tract, proj, facecolor='grey', 
                    edgecolor=None)
                ax4.add_geometries(tract, proj, facecolor='grey', 
                    edgecolor=None)
            else:
                ax3.add_geometries(tract, proj, facecolor=cmapbase(
                    normbase(baseline)), edgecolor=None, rasterized=True)
                ax4.add_geometries(tract, proj, facecolor=cmaplock(
                    normlock(lockdown)), edgecolor=None, rasterized=True) 
    # Add borders, set map extent, etc. 
    for ax in [ax1, ax2, ax3, ax4]:
        # ax.set_extent([-85.,-72., 37, 42.5], proj)
        ax.set_extent([-125,-66.5, 24.5, 49.48], proj)        
        # Add coasts 
        ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='50m',
            facecolor='none', name='coastline', lw=0.25, 
            edgecolor='k'))
        # Add national borders
        ax.add_feature(cfeature.NaturalEarthFeature('cultural', 
            'admin_0_countries', '50m', edgecolor='k', lw=0.25, 
            facecolor='none'))
        # Add states
        ax.add_feature(cfeature.NaturalEarthFeature('cultural', 
            'admin_1_states_provinces_lines', '50m', edgecolor='k', lw=0.25, 
            facecolor='none'))
        # Add water
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 
            'lakes', '50m', edgecolor='k', lw=0.25, 
            facecolor='w'))
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 
            'ocean', '50m', edgecolor='k', lw=0.25, 
            facecolor='w'))
        ax.background_patch.set_visible(False)
        ax.outline_patch.set_visible(False)
    ax1.set_title('(a)', x=0.1, fontsize=12)
    ax2.set_title('(b)', x=0.1, fontsize=12)
    ax3.set_title('(c)', x=0.1, fontsize=12)
    ax4.set_title('(d)', x=0.1, fontsize=12)
    # Colorbars
    caxbase = fig.add_axes([ax3.get_position().x0, 
        ax3.get_position().y0-0.02, 
        (ax3.get_position().x1-ax3.get_position().x0), 0.02])
    cb = mpl.colorbar.ColorbarBase(caxbase, cmap=cmapbase, norm=normbase, 
        spacing='proportional', orientation='horizontal', extend='max', 
        label='NO$_{2}$ [molec cm$^{-2}$/10$^{15}$]')
    caxbase.xaxis.offsetText.set_visible(False)
    caxbase = fig.add_axes([ax4.get_position().x0, 
        ax4.get_position().y0-0.02, (ax4.get_position().x1-ax4.get_position().x0), 
        0.02])
    cb = mpl.colorbar.ColorbarBase(caxbase, cmap=cmaplock, norm=normlock, 
        spacing='proportional', orientation='horizontal', extend='both', 
        label='$\Delta$ NO$_{2}$ [%]')
    plt.subplots_adjust(bottom=0.2, top=0.95)
    plt.savefig(DIR_FIGS+'NO2_gridded_tractavgNEW.png', dpi=600)
    return

def map_no2historic_no2gains(mostno2, leastno2, increaseno2, decreaseno2, 
    FIPS): 
    """

    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import netCDF4 as nc
    import numpy as np
    import matplotlib as mpl
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import shapereader
    proj = ccrs.PlateCarree(central_longitude=0.0)
    fig = plt.figure(figsize=(8,2.5))
    ax1 = plt.subplot2grid((1,2),(0,0), projection=ccrs.PlateCarree(
        central_longitude=0.))
    ax2 = plt.subplot2grid((1,2),(0,1), projection=ccrs.PlateCarree(
        central_longitude=0.))
    for FIPS_i in FIPS: 
        print(FIPS_i)
        # Tigerline shapefile for state
        shp = shapereader.Reader(DIR_GEO+'tigerline/tl_2019_%s_tract/tl_2019_%s_tract'%(
            FIPS_i, FIPS_i))
        records = shp.records()
        tracts = shp.geometries()
        for record, tract in zip(records, tracts):
            # Find GEOID of tract
            gi = record.attributes['GEOID']
            # Look up harmonized NO2-census data for tract
            mostno2_tract = mostno2.loc[mostno2.index.isin([gi])]
            leastno2_tract = leastno2.loc[leastno2.index.isin([gi])]
            increaseno2_tract = increaseno2.loc[increaseno2.index.isin([gi])]
            decreaseno2_tract = decreaseno2.loc[decreaseno2.index.isin([gi])]
            if mostno2_tract.empty == False:
                ax1.add_geometries(tract, proj, facecolor='#ef8a62', 
                    edgecolor=None, rasterized=True)
            if leastno2_tract.empty == False: 
                ax1.add_geometries(tract, proj, facecolor='#67a9cf', 
                    edgecolor=None, rasterized=True)
            if increaseno2_tract.empty == False: 
                ax2.add_geometries(tract, proj, facecolor='#ef8a62', 
                    edgecolor=None, rasterized=True) 
            if decreaseno2_tract.empty == False:
                ax2.add_geometries(tract, proj, facecolor='#67a9cf', 
                    edgecolor=None, rasterized=True)
    # # Highest historic NO2
    # print('Handling tracts with high historic NO2...')
    # for geoid in mostno2.index: 
    #     # Tigerline shapefile for state
    #     shp = shapereader.Reader('/Users/ghkerr/GW/data/geography/tigerline/'+
    #         'tl_2019_%s_tract/tl_2019_%s_tract'%(str(geoid)[:2], 
    #         str(geoid)[:2]))
    #     records = shp.records()
    #     tracts = shp.geometries()
    #     geoids = [r.attributes['GEOID'] for r in records]
    #     # Find which record and tract correspond to GEOID
    #     relevant_record = np.where(np.array(geoids)==str(geoid))[0][0]
    #     relevant_tract = list(tracts)[relevant_record]
    #     ax1.add_geometries(relevant_tract, proj, facecolor='#ef8a62', 
    #         edgecolor=None, alpha=1.)
    # print('Done!')
    # # Lowest historic NO2
    # print('Handling tracts with low historic NO2...')
    # for geoid in leastno2.index: 
    #     # Tigerline shapefile for state
    #     shp = shapereader.Reader('/Users/ghkerr/GW/data/geography/tigerline/'+
    #         'tl_2019_%s_tract/tl_2019_%s_tract'%(str(geoid)[:2], 
    #           str(geoid)[:2]))
    #     records = shp.records()
    #     tracts = shp.geometries()
    #     geoids = [r.attributes['GEOID'] for r in records]
    #     # Find which record and tract correspond to GEOID
    #     relevant_record = np.where(np.array(geoids)==str(geoid))[0][0]
    #     relevant_tract = list(tracts)[relevant_record]
    #     ax1.add_geometries(relevant_tract, proj, facecolor='#67a9cf', 
    #         edgecolor=None, alpha=1.)
    # print('Done!')
    # # Largest gains
    # print('Handling tracts with largest gains...')
    # for geoid in decreaseno2.index: 
    #     # Tigerline shapefile for state
    #     shp = shapereader.Reader('/Users/ghkerr/GW/data/geography/tigerline/'+
    #         'tl_2019_%s_tract/tl_2019_%s_tract'%(str(geoid)[:2], 
    #           str(geoid)[:2]))
    #     records = shp.records()
    #     tracts = shp.geometries()
    #     geoids = [r.attributes['GEOID'] for r in records]
    #     # Find which record and tract correspond to GEOID
    #     relevant_record = np.where(np.array(geoids)==str(geoid))[0][0]
    #     relevant_tract = list(tracts)[relevant_record]
    #     ax2.add_geometries(relevant_tract, proj, facecolor='#67a9cf', 
    #         edgecolor=None, alpha=1.)
    # print('Done!')
    # # Smallest gains
    # print('Handling tracts with smallest gains...')
    # for geoid in increaseno2.index: 
    #     # Tigerline shapefile for state
    #     shp = shapereader.Reader('/Users/ghkerr/GW/data/geography/tigerline/'+
    #         'tl_2019_%s_tract/tl_2019_%s_tract'%(str(geoid)[:2], 
    #           str(geoid)[:2]))
    #     records = shp.records()
    #     tracts = shp.geometries()
    #     geoids = [r.attributes['GEOID'] for r in records]
    #     # Find which record and tract correspond to GEOID
    #     relevant_record = np.where(np.array(geoids)==str(geoid))[0][0]
    #     relevant_tract = list(tracts)[relevant_record]
    #     ax2.add_geometries(relevant_tract, proj, facecolor='#ef8a62', 
    #         edgecolor=None, alpha=1.)
    # print('Done!')
    # Add borders, set map extent, etc. 
    for ax in [ax1, ax2]:
        ax.set_extent([-125,-66.5, 24.5, 49.48], proj)        
        # Add coasts 
        ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='50m',
            facecolor='none', name='coastline', lw=0.25, 
            edgecolor='k'))
        # Add national borders
        ax.add_feature(cfeature.NaturalEarthFeature('cultural', 
            'admin_0_countries', '50m', edgecolor='k', lw=0.25, 
            facecolor='none'))
        # Add states
        ax.add_feature(cfeature.NaturalEarthFeature('cultural', 
            'admin_1_states_provinces_lines', '50m', edgecolor='k', lw=0.25, 
            facecolor='none'))
        # Add water
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 
            'lakes', '50m', edgecolor='k', lw=0.25, 
            facecolor='w'))
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 
            'ocean', '50m', edgecolor='k', lw=0.25, 
            facecolor='w'))    
        ax.background_patch.set_visible(False)
        ax.outline_patch.set_visible(False)
    ax1.set_title('(a)', x=0.1, fontsize=12)
    ax2.set_title('(b)', x=0.1, fontsize=12)
    # Custom legend 
    patch1 = mpatches.Patch(color='#67a9cf', 
        label='Lowest historic NO$_{2}$')
    patch2 = mpatches.Patch(color='#ef8a62', 
        label='Highest historic NO$_{2}$')
    all_handles = (patch1, patch2)
    leg = ax1.legend(handles=all_handles, ncol=2, bbox_to_anchor=(1.2,-0.1),
        frameon=False, fontsize=9)
    patch1 = mpatches.Patch(color='#67a9cf', label='Largest gains')
    patch2 = mpatches.Patch(color='#ef8a62', label='Smallest gains')
    all_handles = (patch1, patch2)
    ax2.legend(handles=all_handles, ncol=2, bbox_to_anchor=(1.1, -0.1),
        frameon=False, fontsize=9)
    plt.savefig(DIR_FIGS+'map_no2historic_no2gains.png', dpi=600)
    return

def bar_no2historic_no2gains(demography, fstr): 
    """
    Returns
    -------
    """    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig = plt.figure(figsize=(6,8))
    ax1 = plt.subplot2grid((4,2),(0,0))
    ax2 = plt.subplot2grid((4,2),(1,0))
    ax3 = plt.subplot2grid((4,2),(2,0))
    ax4 = plt.subplot2grid((4,2),(3,0))
    ax5 = plt.subplot2grid((4,2),(0,1))
    ax6 = plt.subplot2grid((4,2),(1,1))
    ax7 = plt.subplot2grid((4,2),(2,1))
    ax8 = plt.subplot2grid((4,2),(3,1))
    pos = [1,2,3]
    # Historic NO2 pollution 
    ax1.bar(pos,[demography['least2019no2'], demography['2019no2_mean'],
        demography['most2019no2']], color='darkgrey')
    #  Income
    ax2.bar(pos,[demography['income_least2019no2'], demography['income_mean'],
        demography['income_most2019no2']], color='darkgrey')
    # Race
    ax3.bar(pos, [demography['white_least2019no2'], demography['white_mean'], 
        demography['white_most2019no2']], color='#003f5c')
    ax3.bar(pos, [demography['black_least2019no2'],demography['black_mean'],
        demography['black_most2019no2']], 
        bottom=[demography['white_least2019no2'], demography['white_mean'], 
        demography['white_most2019no2']], color='#bc5090')
    ax3.bar(pos, [demography['otherrace_least2019no2'], 
        demography['otherrace_mean'], demography['otherrace_most2019no2']], 
        bottom=[demography['white_least2019no2']+demography['black_least2019no2'], 
        demography['white_mean']+demography['black_mean'], 
        demography['white_most2019no2']+demography['black_most2019no2']], 
        color='#ffa600')
    # Education 
    ax4.bar(pos, [demography['secondary_least2019no2'], 
        demography['secondary_mean'], demography['secondary_most2019no2']], 
        color='#003f5c')
    ax4.bar(pos, [demography['college_least2019no2'], 
        demography['college_mean'], demography['college_most2019no2']], 
        bottom=[demography['secondary_least2019no2'], 
        demography['secondary_mean'], demography['secondary_most2019no2']], 
        color='#bc5090')
    ax4.bar(pos, [demography['grad_least2019no2'], demography['grad_mean'], 
        demography['grad_most2019no2']], 
        bottom=[demography['secondary_least2019no2']+demography['college_least2019no2'], 
        demography['secondary_mean']+demography['college_mean'], 
        demography['secondary_most2019no2']+demography['college_most2019no2']], 
        color='#ffa600')
    # COVID-19 NO2 gains
    ax5.bar(pos,[demography['largestgains'], demography['gains_mean'], 
        demography['smallestgains']], color='darkgrey')
    # Income
    ax6.bar(pos,[demography['income_largestgains'], demography['income_mean'], 
        demography['income_smallestgains']], color='darkgrey')
    # Race
    ax7.bar(pos, [demography['white_largestgains'], demography['white_mean'], 
        demography['white_smallestgains']], color='#003f5c')
    ax7.bar(pos, [demography['black_largestgains'], demography['black_mean'], 
        demography['black_smallestgains']], bottom=[
        demography['white_largestgains'], demography['white_mean'], 
        demography['white_smallestgains']], 
        color='#bc5090')
    ax7.bar(pos, [demography['otherrace_largestgains'], 
        demography['otherrace_mean'],  demography['otherrace_smallestgains']], 
        bottom=[demography['white_largestgains']+demography['black_largestgains'], 
        demography['white_mean']+demography['black_mean'], 
        demography['white_smallestgains']+demography['black_smallestgains']],
        color='#ffa600')
    # Education 
    ax8.bar(pos, [demography['secondary_largestgains'], 
        demography['secondary_mean'], demography['secondary_smallestgains']], 
        color='#003f5c')
    ax8.bar(pos, [demography['college_largestgains'], 
        demography['college_mean'], demography['college_smallestgains']], 
        bottom=[demography['secondary_largestgains'], 
        demography['secondary_mean'], demography['secondary_smallestgains']], 
        color='#bc5090')
    ax8.bar(pos, [demography['grad_largestgains'], demography['grad_mean'], 
        demography['grad_smallestgains']], bottom=[
        demography['secondary_largestgains']+demography['college_largestgains'], 
        demography['secondary_mean']+demography['college_mean'], 
        demography['secondary_smallestgains']+demography['college_smallestgains']], 
        color='#ffa600')
    # Axis titles
    ax1.set_title('(a)', x=0.1, fontsize=12)
    ax2.set_title('(c)', x=0.1, fontsize=12)
    ax3.set_title('(e)', x=0.1, fontsize=12)
    ax4.set_title('(g)', x=0.1, fontsize=12)
    ax5.set_title('(b)', x=0.1, fontsize=12)
    ax6.set_title('(d)', x=0.1, fontsize=12)
    ax7.set_title('(f)', x=0.1, fontsize=12)
    ax8.set_title('(h)', x=0.1, fontsize=12)
    ax1.set_ylabel('NO$_{2}$ x 10$^{15}$\n[molec cm$^{-2}$]', fontsize=9)
    ax1.set_title('(a)', x=0.1, fontsize=12)
    ax1.set_yticks([0e15, 2e15, 4e15, 6e15, 8e15, 10e15])
    ax1.set_yticklabels(['0', '2', '4', '6', '8', '10'], fontsize=9)
    ax1.yaxis.offsetText.set_visible(False)
    # Axis labels and ticks
    for ax in [ax2, ax6]:
        ax.set_ylim([0, 100000])
        ax.set_yticks([0, 20000, 40000, 60000, 80000, 100000])
        ax.set_yticklabels([])
    ax2.set_yticklabels(['0', '20,000', '40,000', '60,000', '80,000', '100,000'],
        fontsize=9)
    for ax in [ax3, ax4, ax7, ax8]:
        ax.set_ylim([0, 100.])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels([])
    for ax in [ax3, ax4]:
        ax.set_yticklabels(['0', '25', '50', '75', '100'], fontsize=9)
    ax2.set_ylabel('Income [$]', fontsize=9)
    ax3.set_ylabel('Race [%]', fontsize=9)        
    ax4.set_ylabel('Educational\nAttainment [%]', fontsize=9)
    ax4.set_yticklabels(['0', '25', '50', '75', '100'], fontsize=9)
    ax5.set_yticks([-40, -30, -20, -10, 0, 10])
    ax5.set_yticklabels(['-40', '-30', '-20', '-10', '0', '10'], fontsize=9)
    ax5.set_ylabel(r'$\Delta\:$NO$_{2}$ [%]', fontsize=9)
    for ax in [ax1, ax2, ax3, ax5, ax6, ax7]:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax4.set_xticks([1,2,3])
    ax4.set_xticklabels(['NO$_{2}\:$<$\:P_{10}$', 
        '$\overline{\mathregular{NO_{2}}}$', 'NO$_{2}\:$>$\:P_{90}$'], fontsize=9)
    ax8.set_xticks([1,2,3])
    ax8.set_xticklabels(['$\Delta\:$NO$_{2}\:$<$\:P_{10}$', 
        '$\overline{\Delta\:\mathregular{NO_{2}}}$', 
        '$\Delta\:$NO$_{2}\:$>$\:P_{90}$'], fontsize=9)
    for ax in [ax5, ax6, ax7, ax8]:
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
    # Add arrows to illustrate changes
    ax1.annotate('', xy=(0.1,1.4), 
        xycoords='axes fraction', xytext=(0.9,1.4),
        arrowprops=dict(arrowstyle= '<|-', color='k', lw=2), va='center', 
        transform=fig.transFigure)
    ax1.annotate('Highest NO$_{2}$ pollution', xy=(0.06,1.55), 
        xycoords='axes fraction', va='center', fontsize=12,
        transform=fig.transFigure)
    ax5.annotate('', xy=(0.1,1.4), 
        xycoords='axes fraction', xytext=(0.9,1.4),
        arrowprops=dict(arrowstyle= '-|>', color='k', lw=2), va='center', 
        transform=fig.transFigure)
    ax5.annotate('Largest gains', xy=(0.23,1.55), 
        xycoords='axes fraction', va='center', fontsize=12,
        transform=fig.transFigure)
    # Custom legend
    patch1 = mpatches.Patch(color='#003f5c', 
        label='(e-f) White\n(g-h) High school degree or less')
    patch2 = mpatches.Patch(color='#bc5090', 
        label='(e-f) Black or African American\n(g-h) College degree or some college')
    patch3 = mpatches.Patch(color='#ffa600', 
        label='(e-f) Other\n(g-h) Graduate or professional degree')
    all_handles = (patch1, patch2, patch3)
    leg = ax4.legend(handles=all_handles, ncol=2, bbox_to_anchor=(2.4,-0.2),
        frameon=False, fontsize=9)
    plt.subplots_adjust(bottom=0.16, left=0.15, right=0.9, hspace=0.3,
        wspace=0.15)
    plt.savefig(DIR_FIGS+'bar_no2historic_no2gains_%s.png'%fstr, dpi=500)
    return   

def cdf(harmonized, fstr):
    """
    """
    import numpy as np
    from scipy import stats
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    # Most/least wealthy, white, and educated (educated = some college or higher)
    mostwealthy = harmonized.loc[harmonized['AJZAE001'] > np.nanpercentile(
        harmonized['AJZAE001'], 90)]
    leastwealthy = harmonized.loc[harmonized['AJZAE001'] < np.nanpercentile(
        harmonized['AJZAE001'], 10)]
    frac_white = (harmonized['AJWNE002']/harmonized['AJWBE001'])
    mostwhite = harmonized.iloc[np.where(frac_white > 
        np.nanpercentile(frac_white, ptile_upper))]
    leastwhite = harmonized.iloc[np.where(frac_white < 
        np.nanpercentile(frac_white, ptile_lower))]
    frac_educated = (harmonized.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        harmonized['AJYPE001'])
    mosteducated = harmonized.iloc[np.where(frac_educated > 
        np.nanpercentile(frac_educated, ptile_upper))]
    leasteducated = harmonized.iloc[np.where(frac_educated < 
        np.nanpercentile(frac_educated, ptile_lower))]
    # Plotting
    fig = plt.figure(figsize=(4,6))
    ax1 = plt.subplot2grid((3,1),(0,0))
    ax2 = plt.subplot2grid((3,1),(1,0))
    ax3 = plt.subplot2grid((3,1),(2,0))
    bins = np.hstack([np.linspace(0, 1e16, 1000), np.inf])
    # Most wealthy and least wealthy
    n1, b1, p1 = ax1.hist(mostwealthy['PRENO2'].values, bins=bins, 
        density=True, lw=1.5, histtype='step', cumulative=True, 
        color='#2a6a99')
    n2, b2, p2 = ax1.hist(leastwealthy['PRENO2'].values, bins=bins, 
        density=True, lw=1.5, histtype='step', cumulative=True, 
        color='#d88546')
    n3, b3, p3 = ax1.hist(mostwealthy['POSTNO2'].values, bins=bins,
        density=True, histtype='step', lw=1., ls='--',cumulative=True, 
        color='#2a6a99')
    n4, b4, p4 = ax1.hist(leastwealthy['POSTNO2'].values, bins=bins, 
        density=True, histtype='step', lw=1., ls='--', cumulative=True,
        color='#d88546')
    # Most white and least white
    n1, b1, p1 = ax2.hist(mostwhite['PRENO2'].values, bins=bins, 
        density=True, lw=1.5, histtype='step', cumulative=True, 
        color='#2a6a99')
    n2, b2, p2 = ax2.hist(leastwhite['PRENO2'].values, bins=bins, 
        density=True, lw=1.5, ls='-', histtype='step', cumulative=True, 
        color='#d88546')
    n3, b3, p3 = ax2.hist(mostwhite['POSTNO2'].values, bins=bins, density=True, 
        histtype='step', lw=1., ls='--',cumulative=True, color='#2a6a99')
    n4, b4, p4 = ax2.hist(leastwhite['POSTNO2'].values, bins=bins,
        density=True, histtype='step', lw=1., ls='--', cumulative=True, 
        color='#d88546')
    # Most white and least white
    n1, b1, p1 = ax3.hist(mosteducated['PRENO2'].values, bins=bins,
        density=True, lw=1.5, histtype='step', cumulative=True, 
        color='#2a6a99')
    n2, b2, p2 = ax3.hist(leasteducated['PRENO2'].values, bins=bins, 
        density=True, lw=1.5, ls='-', histtype='step', cumulative=True, 
        color='#d88546')
    n3, b3, p3 = ax3.hist(mosteducated['POSTNO2'].values, bins=bins, 
        density=True, histtype='step', lw=1., ls='--',cumulative=True, 
        color='#2a6a99')
    n4, b4, p4 = ax3.hist(leasteducated['POSTNO2'].values, bins=bins, 
        density=True, histtype='step', lw=1., ls='--',cumulative=True, 
        color='#d88546')
    # Aesthetics
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim([1e15, 7e15])
        ax.set_xticks([1e15, 2e15, 3e15, 4e15, 5e15, 6e15, 7e15])
        ax.set_xticklabels([])
        ax.set_ylim([0, 1.01])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.])
        ax.set_yticklabels(['0', '', '0.5', '', '1.0'])
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
    ax2.set_ylabel('Cumulative distribution', fontsize=12)    
    ax3.set_xticklabels(['1', '2', '3', '4', '5', '6', '7'])
    ax3.set_xlabel('NO$_{2}$ x 10$^{15}$ [molec cm$^{-2}$]', fontsize=12)
    ax1.set_title('(a)', x=0.1, fontsize=12)
    ax1.text(.5e16, 0.65, 'Most wealthy', fontsize=12, color='#2a6a99')
    ax1.text(.5e16, 0.5, 'vs.', fontsize=12, color='k')
    ax1.text(.5e16, 0.35, 'Least wealthy', fontsize=12, color='#d88546')
    ax2.set_title('(b)', x=0.1, fontsize=12)
    ax2.text(.5e16, 0.65, 'Most white', fontsize=12, color='#2a6a99')
    ax2.text(.5e16, 0.5, 'vs.', fontsize=12, color='k')
    ax2.text(.5e16, 0.35, 'Least white', fontsize=12, color='#d88546')
    ax3.set_title('(c)', x=0.1, fontsize=12)
    ax3.text(.5e16, 0.65, 'Most educated', fontsize=12, color='#2a6a99')
    ax3.text(.5e16, 0.5, 'vs.', fontsize=12, color='k')
    ax3.text(.5e16, 0.35, 'Least educated', fontsize=12, color='#d88546')
    plt.subplots_adjust(left=0.2, hspace=0.6, bottom=0.15, top=0.9)
    # Custom legend
    custom_lines = [Line2D([0], [0], color='k', lw=1.5),
        Line2D([0], [0], color='k', ls='--', lw=1)]
    ax3.legend(custom_lines, ['Historic', 'Lockdown'], 
        bbox_to_anchor=(0.95, -0.55), ncol=2, frameon=False)
    plt.savefig(DIR_FIGS+'cdf_income_race_education_%sNEW.png'%fstr, dpi=500)
    return 

def lollipop(harmonized, harmonized_urban, harmonized_rural):
    """
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # Top 15 MSAs (can be found at 
    # https://en.wikipedia.org/wiki/List_of_metropolitan_statistical_areas
    # Wikipedia usually lists the counties included in each MSA (but note that
    # this is different than the CSA, which includes more counties). The 
    # corresponding FIPS codes for each county can be found at 
    # www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697
    # New York-Newark-Jersey City, NY-NJ-PA MSA
    newyork = ['36047','36081','36061','36005','36085','36119','34003',
        '34017','34031','36079','36087','36103','36059','34023','34025',
        '34029','34035','34013','34039','34027','34037'	,'34019','42103']
    # Los Angeles-Long Beach-Anaheim, CA MSA
    losangeles = ['06037','06059']
    # Chicago-Naperville-Elgin, IL-IN-WI MSA
    chicago = ['17031','17037','17043','17063','17091','17089','17093',
        '17111','17197','18073','18089','18111','18127','17097','55059']
    # Dallas-Fort Worth-Arlington, TX MSA
    dallas = ['48085','48113','48121','48139','48231','48257','48397',
        '48251','48367','48439','48497']
    # Houston-The Woodlands-Sugar Land, TX MSA
    houston = ['48201','48157','48339','48039','48167','48291','48473',
        '48071','48015']
    # Washington-Arlington-Alexandria, DC-VA-MD-WV MSA
    washington = ['11001','24009','24017','24021','24031','24033','51510',
        '51013','51043','51047','51059','51600','51610','51061','51630',
        '51107','51683','51685','51153','51157','51177','51179','51187']
    # Miami-Fort Lauderdale-Pompano Beach, FL MSA	
    miami = ['12086','12011','12099']
    # Philadelphia-Camden-Wilmington, PA-NJ-DE-MD MSA
    philadelphia = ['34005','34007','34015','42017','42029','42091','42045',
        '42101','10003','24015','34033']
    # Atlanta-Sandy Springs-Alpharetta, GA MSA
    atlanta = ['13121','13135','13067','13089','13063','13057','13117',
        '13151','13223','13077','13097','13045','13113','13217','13015',
        '13297','13247','13013','13255','13227','13143','13085','13035',
        '13199','13171','13211','13231','13159','13149']
    # Phoenix-Mesa-Chandler, AZ MSA
    phoenix = ['04013','04021','04007']
    # Boston-Cambridge-Newton, MA-NH MSA
    boston = ['25021','25023','25025','25009','25017','33015','33017']
    # San Francisco-Oakland-Berkeley, CA MSA
    sanfrancisco = ['06001','06013','06075','06081','06041']
    # Riverside-San Bernardino-Ontario, CA MSA
    riverside = ['06065','06071']
    # Detroit-Warren-Dearborn, MI MSA
    detroit = ['26163','26125','26099','26093','26147','26087']
    # Seattle-Tacoma-Bellevue, WA MSA
    seattle = ['53033','53061','53053']
    # Colors
    color_white = '#0095A8'
    color_non = '#FF7043'
    os = 1.2
    # Initialize figure
    fig = plt.figure(figsize=(6,6.5))
    ax = plt.subplot2grid((1,1),(0,0))
    i = 0 
    yticks = []
    # For all tracts
    frac_white = (harmonized['AJWNE002']/harmonized['AJWBE001'])
    mostwhite = harmonized.iloc[np.where(frac_white > 
        np.nanpercentile(frac_white, ptile_upper))]
    leastwhite = harmonized.iloc[np.where(frac_white < 
        np.nanpercentile(frac_white, ptile_lower))]
    # Lockdown NO2  or white and non-white populations
    ax.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
    ax.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
    ax.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()]), 
        np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()])+
        np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    # Historic NO2 for white and non-white populations
    ax.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    # Draw connection lines
    ax.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
        np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
        np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7    
    # For rural tracts
    frac_white = (harmonized_rural['AJWNE002']/harmonized_rural['AJWBE001'])
    mostwhite = harmonized_rural.iloc[np.where(frac_white > 
        np.nanpercentile(frac_white, ptile_upper))]
    leastwhite = harmonized_rural.iloc[np.where(frac_white < 
        np.nanpercentile(frac_white, ptile_lower))]
    ax.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, 
        zorder=12)
    ax.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()]), 
        np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()])+
        np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
        np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
        np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7  
    # For rural tracts
    frac_white = (harmonized_urban['AJWNE002']/harmonized_urban['AJWBE001'])
    mostwhite = harmonized_urban.iloc[np.where(frac_white > 
        np.nanpercentile(frac_white, ptile_upper))]
    leastwhite = harmonized_urban.iloc[np.where(frac_white < 
        np.nanpercentile(frac_white, ptile_lower))]
    ax.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, 
        zorder=12)
    ax.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()]), 
        np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()])+
        np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
        np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
        np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7        
    citynames = [r'$\bf{All}$', r'$\bf{Rural}$', r'$\bf{Urban}$',
        'New York', 'Los Angeles', 'Chicago', 'Dallas', 'Houston', 
        'Washington', 'Miami', 'Philadelphia', 'Atlanta', 'Phoenix', 
        'Boston', 'San Francisco', 'Riverside', 'Detroit', 'Seattle']   
    for city in [newyork, losangeles, chicago, dallas, houston, washington,
        miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
        riverside, detroit, seattle]:
        # Subset for city
        harmonized_city = subset_harmonized_bycountyfips(harmonized, city)
        # Find particular demographic for each city
        frac_white = (harmonized_city['AJWNE002']/harmonized_city['AJWBE001'])
        mostwhite = harmonized_city.iloc[np.where(frac_white > 
            np.nanpercentile(frac_white, ptile_upper))]
        leastwhite = harmonized_city.iloc[np.where(frac_white < 
            np.nanpercentile(frac_white, ptile_lower))]
        # Lockdown NO2  or white and non-white populations
        ax.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
        ax.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
        ax.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()]), 
            np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()])+
            np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))), 
            [i-os,i-os], color='k', ls='-', zorder=10)    
        # i = i+2
        # Historic NO2 for white and non-white populations
        ax.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
        ax.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
        # Draw connection lines
        ax.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
            np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
            np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))), 
            [i+os,i+os], color='k', ls='--', zorder=10)
        yticks.append(np.nanmean([i]))
        i = i+7
    # Aesthetics 
    ax.xaxis.tick_top()
    ax.set_xlim([0.5e15,10e15])
    ax.set_xticks(np.arange(1e15,10e15,1e15))
    ax.set_yticks(yticks)
    ax.set_yticklabels(citynames)
    ax.tick_params(axis='y', left=False)
    ax.set_xlabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', x=0.15, labelpad=10,
        color='darkgrey')
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', colors='grey')
    ax.xaxis.offsetText.set_visible(False)
    # Hide spines
    for side in ['right', 'left', 'top', 'bottom']:
        ax.spines[side].set_visible(False)
    ax.grid(axis='x', zorder=0, color='darkgrey')
    # Custom legend
    custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
        Line2D([0], [0], marker='o', color=color_non, lw=0),  
        Line2D([0], [0], color='k', lw=1.0),
        Line2D([0], [0], color='k', ls='--', lw=1)]
    ax.legend(custom_lines, ['Most white', 'Least white', 'Baseline', 'Lockdown'], 
        bbox_to_anchor=(0.4, -0.07), loc=8, ncol=4, frameon=False)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.20, right=0.96, bottom=0.05, top=0.9)
    plt.savefig(DIR_FIGS+'lollipop_onrace.png', dpi=500)
    plt.show()
    
    # Most versus least wealthy
    fig = plt.figure(figsize=(6,6.5))
    ax = plt.subplot2grid((1,1),(0,0))
    i = 0 
    yticks = []
    # For all tracts
    mostwealthy = harmonized.loc[harmonized['AJZAE001'] > 
        np.nanpercentile(harmonized['AJZAE001'], 90)]
    leastwealthy = harmonized.loc[harmonized['AJZAE001'] < 
        np.nanpercentile(harmonized['AJZAE001'], 10)]
    # Lockdown NO2  or white and non-white populations
    ax.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
    ax.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
    ax.plot((np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()]), 
        np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+
        np.abs(np.diff([mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    # Historic NO2 for white and non-white populations
    ax.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    # Draw connection lines
    ax.plot((np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()]), 
        np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
        np.abs(np.diff([mostwealthy['POSTNO2'].mean(), leastwealthy['POSTNO2'].mean()]))), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7    
    # For rural tracts
    mostwealthy = harmonized_rural.loc[harmonized_rural['AJZAE001'] > 
        np.nanpercentile(harmonized_rural['AJZAE001'], 90)]
    leastwealthy = harmonized_rural.loc[harmonized_rural['AJZAE001'] < 
        np.nanpercentile(harmonized_rural['AJZAE001'], 10)]
    ax.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, 
        zorder=12)
    ax.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()]), 
        np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+
        np.abs(np.diff([mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()]), 
        np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
        np.abs(np.diff([mostwealthy['POSTNO2'].mean(), leastwealthy['POSTNO2'].mean()]))), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7  
    # For urban tracts
    mostwealthy = harmonized_urban.loc[harmonized_urban['AJZAE001'] > 
        np.nanpercentile(harmonized_urban['AJZAE001'], 90)]
    leastwealthy = harmonized_urban.loc[harmonized_urban['AJZAE001'] < 
        np.nanpercentile(harmonized_urban['AJZAE001'], 10)]
    ax.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, 
        zorder=12)
    ax.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()]), 
        np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+
        np.abs(np.diff([mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()]), 
        np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
        np.abs(np.diff([mostwealthy['POSTNO2'].mean(), leastwealthy['POSTNO2'].mean()]))), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7      
    for city in [newyork, losangeles, chicago, dallas, houston, washington,
        miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
        riverside, detroit, seattle]:
        # Subset for city
        harmonized_city = subset_harmonized_bycountyfips(harmonized, city)
        # Find particular demographic for each city
        mostwealthy = harmonized_city.loc[harmonized_city['AJZAE001'] > 
            np.nanpercentile(harmonized_city['AJZAE001'], 90)]
        leastwealthy = harmonized_city.loc[harmonized_city['AJZAE001'] < 
            np.nanpercentile(harmonized_city['AJZAE001'], 10)]
        # Lockdown NO2  or white and non-white populations
        ax.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, 
            zorder=12)
        ax.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, 
            zorder=12)
        ax.plot((np.min([mostwealthy['PRENO2'].mean(),
            leastwealthy['PRENO2'].mean()]), np.min([mostwealthy[
            'PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+np.abs(np.diff(
            [mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))), 
            [i-os,i-os], color='k', ls='-', zorder=10)    
        # Historic NO2 for white and non-white populations
        ax.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, 
            zorder=12)
        ax.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, 
            zorder=12)
        # Draw connection lines
        ax.plot((np.min([mostwealthy['POSTNO2'].mean(),
            leastwealthy['POSTNO2'].mean()]), np.min([
            mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
            np.abs(np.diff([mostwealthy['POSTNO2'].mean(), 
            leastwealthy['POSTNO2'].mean()]))), [i+os,i+os], color='k', 
            ls='--', zorder=10)
        yticks.append(np.nanmean([i]))
        i = i+7
    # Aesthetics 
    ax.xaxis.tick_top()
    ax.set_xlim([0.5e15,10e15])
    ax.set_xticks(np.arange(1e15,10e15,1e15))
    ax.set_yticks(yticks)
    ax.set_yticklabels(citynames)
    ax.tick_params(axis='y', left=False)
    ax.set_xlabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', x=0.15, labelpad=10,
        color='darkgrey')
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', colors='grey')
    ax.xaxis.offsetText.set_visible(False)
    # Hide spines
    for side in ['right', 'left', 'top', 'bottom']:
        ax.spines[side].set_visible(False)
    ax.grid(axis='x', zorder=0, color='darkgrey')
    # Custom legend
    custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
        Line2D([0], [0], marker='o', color=color_non, lw=0),  
        Line2D([0], [0], color='k', lw=1.0),
        Line2D([0], [0], color='k', ls='--', lw=1)]
    ax.legend(custom_lines, ['Highest income', 'Lowest income', 'Baseline',
        'Lockdown'], bbox_to_anchor=(0.4, -0.07), loc=8, ncol=4, frameon=False)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.20, right=0.96, bottom=0.05, top=0.9)
    plt.savefig(DIR_FIGS+'lollipop_onincome.png', dpi=500)
    plt.show()
    
    # Most versus least educated
    fig = plt.figure(figsize=(6,6.5))
    ax = plt.subplot2grid((1,1),(0,0))
    i = 0 
    yticks = []
    # For all tracts
    frac_educated = (harmonized.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        harmonized['AJYPE001'])
    mosteducated = harmonized.iloc[np.where(frac_educated > 
        np.nanpercentile(frac_educated, 90))]
    leasteducated = harmonized.iloc[np.where(frac_educated < 
        np.nanpercentile(frac_educated, 10))]
    # Lockdown NO2  or white and non-white populations
    ax.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
    ax.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
    ax.plot((np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()]), 
        np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()])+
        np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    # Historic NO2 for white and non-white populations
    ax.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    # Draw connection lines
    ax.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
        np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
        np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7    
    # For rural tracts
    frac_educated = (harmonized_rural.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        harmonized_rural['AJYPE001'])
    mosteducated = harmonized_rural.iloc[np.where(frac_educated > 
        np.nanpercentile(frac_educated, 90))]
    leasteducated = harmonized_rural.iloc[np.where(frac_educated < 
        np.nanpercentile(frac_educated, 10))]
    # Lockdown NO2  or white and non-white populations
    ax.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
    ax.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
    ax.plot((np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()]), 
        np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()])+
        np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    # Historic NO2 for white and non-white populations
    ax.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    # Draw connection lines
    ax.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
        np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
        np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7    
    # For urban tracts
    frac_educated = (harmonized_urban.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        harmonized_urban['AJYPE001'])
    mosteducated = harmonized_urban.iloc[np.where(frac_educated > 
        np.nanpercentile(frac_educated, 90))]
    leasteducated = harmonized_urban.iloc[np.where(frac_educated < 
        np.nanpercentile(frac_educated, 10))]
    # Lockdown NO2  or white and non-white populations
    ax.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
    ax.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
    ax.plot((np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()]), 
        np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()])+
        np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    # Historic NO2 for white and non-white populations
    ax.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    # Draw connection lines
    ax.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
        np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
        np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7    
    for city in [newyork, losangeles, chicago, dallas, houston, washington,
        miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
        riverside, detroit, seattle]:
        # Subset for city
        harmonized_city = subset_harmonized_bycountyfips(harmonized, city)
        # Find particular demographic for each city; GHK note: check only 
        # summing to # AJYPE021 to weed out people with some college
        frac_educated = (harmonized_city.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
            harmonized_city['AJYPE001'])
        mosteducated = harmonized_city.iloc[np.where(frac_educated > 
            np.nanpercentile(frac_educated, 90))]
        leasteducated = harmonized_city.iloc[np.where(frac_educated < 
            np.nanpercentile(frac_educated, 10))]
        # Lockdown NO2  or white and non-white populations
        ax.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, 
            zorder=12)
        ax.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, 
            zorder=12)
        ax.plot((np.min([mosteducated['PRENO2'].mean(),
            leasteducated['PRENO2'].mean()]), 
            np.min([mosteducated['PRENO2'].mean(),
            leasteducated['PRENO2'].mean()])+np.abs(np.diff(
            [mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))), 
            [i-os,i-os], color='k', ls='-', zorder=10)    
        # Historic NO2 for white and non-white populations
        ax.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, 
            zorder=12)
        ax.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, 
            zorder=12)
        # Draw connection lines
        ax.plot((np.min([mosteducated['POSTNO2'].mean(), leasteducated[
            'POSTNO2'].mean()]), np.min([mosteducated['POSTNO2'].mean(), 
            leasteducated['POSTNO2'].mean()])+np.abs(np.diff(
            [mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))), 
            [i+os,i+os], color='k', ls='--', zorder=10)
        yticks.append(np.nanmean([i]))
        i = i+7
    # Aesthetics 
    ax.xaxis.tick_top()
    ax.set_xlim([0.5e15,10e15])
    ax.set_xticks(np.arange(1e15,10e15,1e15))
    # ax.set_xticklabels(np.arange(1e15,10e15,1e15), color='grey')
    ax.set_yticks(yticks)
    ax.set_yticklabels(citynames)
    ax.tick_params(axis='y', left=False)
    ax.set_xlabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', x=0.15, labelpad=10,
        color='darkgrey')
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', colors='grey')
    ax.xaxis.offsetText.set_visible(False)
    # Hide spines
    for side in ['right', 'left', 'top', 'bottom']:
        ax.spines[side].set_visible(False)
    ax.grid(axis='x', zorder=0, color='darkgrey')
    # Custom legend
    custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
        Line2D([0], [0], marker='o', color=color_non, lw=0),  
        Line2D([0], [0], color='k', lw=1.0),
        Line2D([0], [0], color='k', ls='--', lw=1)]
    ax.legend(custom_lines, ['Most educated', 'Least educated', 'Baseline', 
        'Lockdown'], bbox_to_anchor=(0.4, -0.07), loc=8, ncol=4, frameon=False)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.20, right=0.96, bottom=0.05, top=0.9)
    plt.savefig(DIR_FIGS+'lollipop_oneducation.png', dpi=500)
    return

def hexbin(harmonized_vehicle):
    """
    """
    import matplotlib
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5,9))
    ax1 = plt.subplot2grid((2,1),(0,0))
    ax2 = plt.subplot2grid((2,1),(1,0))
    ax1.set_title('(a)', x=0.9, y=0.9)
    ax2.set_title('(b)', x=0.9, y=0.9)
    # Create discrete colormaps
    cmap = plt.get_cmap('YlGnBu', 10)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=50)
    ax1.hexbin(harmonized_vehicle['Within10'], harmonized_vehicle['PRENO2'],
        C=harmonized_vehicle['FracNoCar']*100, cmap=cmap, gridsize=35, 
        vmin=0, vmax=50) 
    ax2.hexbin(harmonized_vehicle['Within10'], harmonized_vehicle['NO2_ABS'],
        C=harmonized_vehicle['FracNoCar']*100, cmap=cmap, gridsize=35, 
        vmin=0, vmax=50) 
    # Aesthetics
    ax1.set_xlim([0,300])
    ax1.set_xticklabels([])
    ax1.set_ylim([0.0,14e15])
    ax1.set_yticks(np.linspace(0,1.4e16,8))
    ax1.set_yticklabels([str(int(x)) for x in np.linspace(0,14,8)])
    ax1.yaxis.offsetText.set_visible(False)
    ax1.set_ylabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]')
    for pos in ['right','top']:
        ax1.spines[pos].set_visible(False)
        ax1.spines[pos].set_visible(False)
    ax2.set_xlim([0,300])
    ax2.set_ylim([-6e15, 0])
    ax2.yaxis.offsetText.set_visible(False)
    ax2.set_xlabel('[Roads (10 km radius)$^{-1}$]')
    ax2.set_ylabel('$\Delta$ NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]')
    # ax2.xaxis.tick_top()
    for pos in ['right','top']:
        ax2.spines[pos].set_visible(False)
        ax2.spines[pos].set_visible(False)
    # Colorbar
    cax = fig.add_axes([ax1.get_position().x0,
        ax2.get_position().y0-0.07, 
        ax2.get_position().x1-ax2.get_position().x0, 0.02])
    matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
        spacing='proportional', orientation='horizontal', extend='max',
        label='Households without vehicles [%]')
    plt.subplots_adjust(hspace=0.1, top=0.98, bottom=0.15)
    plt.savefig(DIR_FIGS+'hexbin_roaddensity_no2.png', dpi=500)
    return 

def bar_gains(harmonized):
    """
    Parameters
    ----------
    harmonized_vehicle : pandas.core.frame.DataFrame
        Harmonized tract-level TROPOMI NO2 and census data for state(s) of 
        interest merged with vehicle ownership/road density data

    Returns
    -------        
    None    
    """
    import numpy as np
    from decimal import Decimal
    import matplotlib.pyplot as plt
    # Largest, smallest and median gains
    increaseno2 = harmonized.loc[harmonized['NO2_ABS']>
        np.nanpercentile(harmonized['NO2_ABS'], ptile_upper)]
    medianno2 = harmonized.loc[(harmonized['NO2_ABS']>
        np.nanpercentile(harmonized['NO2_ABS'], 45)) & (harmonized['NO2_ABS']<=
        np.nanpercentile(harmonized['NO2_ABS'], 55))]
    decreaseno2 = harmonized.loc[harmonized['NO2_ABS']<
        np.nanpercentile(harmonized['NO2_ABS'], ptile_lower)]
    # Initialize figure, subplot
    fig = plt.figure(figsize=(8,9))
    ax1 = plt.subplot2grid((6,1),(0,0))
    ax2 = plt.subplot2grid((6,1),(1,0))
    ax3 = plt.subplot2grid((6,1),(2,0))
    ax4 = plt.subplot2grid((6,1),(3,0))
    ax5 = plt.subplot2grid((6,1),(4,0))
    ax6 = plt.subplot2grid((6,1),(5,0))
    # Axis titles
    ax1.set_title('(a) $\Delta$ NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', loc='left')
    ax2.set_title('(b) Household income [$]', loc='left')
    ax3.set_title('(c) Racial background [%]', loc='left')
    ax4.set_title('(d) Ethnic background [%]', loc='left')
    ax5.set_title('(e) Educational attainment [%]',loc='left')
    ax6.set_title('(f) Household vehicle ownership [%]',loc='left')
    colors = ['#7fa5c2','darkgrey','#e7ac56']
    
    # # # # Change in NO2
    # Urban 
    ax1.barh([2,1,0], [decreaseno2['NO2_ABS'].mean(), medianno2['NO2_ABS'].mean(), 
        increaseno2['NO2_ABS'].mean()], color=colors[0])
    print('Largest gains for NO2 = %.2E'%Decimal(decreaseno2['NO2_ABS'].mean()))
    ax1.text(decreaseno2['NO2_ABS'].mean(), 2, '-2.9 x 10$^{15}$', color='black', 
        va='center')
    print('Largest gains for NO2 = %.2E'%Decimal(medianno2['NO2_ABS'].mean()))
    ax1.text(medianno2['NO2_ABS'].mean(), 1, '-6.0 x 10$^{14}$', color='black', 
        va='center')
    print('Smallest gains for NO2 = %.2E'%Decimal(increaseno2['NO2_ABS'].mean()))
    ax1.text(increaseno2['NO2_ABS'].mean(), 0, '8.4 x 10$^{13}$', color='black', 
        va='center')
    # # Rural
    # ax1.barh([2,1,0], [decreaseno2['NO2_ABS'].mean(), medianno2['NO2_ABS'].mean(), 
    #     increaseno2['NO2_ABS'].mean()], color=colors[0])
    # print('Largest gains for NO2 = %.2E'%Decimal(decreaseno2['NO2_ABS'].mean()))
    # ax1.text(decreaseno2['NO2_ABS'].mean(), 2, '-2.9 x 10$^{15}$', color='black', 
    #     va='center')
    # print('Largest gains for NO2 = %.2E'%Decimal(medianno2['NO2_ABS'].mean()))
    # ax1.text(medianno2['NO2_ABS'].mean(), 1, '-6.0 x 10$^{14}$', color='black', 
    #     va='center')
    # print('Smallest gains for NO2 = %.2E'%Decimal(increaseno2['NO2_ABS'].mean()))
    # ax1.text(increaseno2['NO2_ABS'].mean(), 0, '8.4 x 10$^{13}$', color='black', 
    #     va='center')    
    
    # # # # Income
    ax2.barh([2,1,0], [decreaseno2['AJZAE001'].mean(), 
        medianno2['AJZAE001'].mean(), increaseno2['AJZAE001'].mean()], 
        color=colors[0])
    ax2.text(60000, 2, ' %d'%(decreaseno2['AJZAE001'].mean()), color='black', 
        va='center')
    ax2.text(60000, 1, ' %d'%(medianno2['AJZAE001'].mean()), color='black', 
        va='center')
    ax2.text(60000, 0, ' %d'%(increaseno2['AJZAE001'].mean()), color='black', 
        va='center')
    
    # # # # Racial background
    left, i = 0, 0
    labels = ['White', 'Black', 'Other']
    # Largest gains
    for data, color in zip([(decreaseno2['AJWNE002']/
        decreaseno2['AJWBE001']).mean(),
        (decreaseno2['AJWNE003']/decreaseno2['AJWBE001']).mean(),      
        ((decreaseno2['AJWNE004']+decreaseno2['AJWNE005']+
         decreaseno2['AJWNE006']+decreaseno2['AJWNE007']+
        decreaseno2['AJWNE008'])/decreaseno2['AJWBE001']).mean()], colors):             
        ax3.barh(2, data, color=color, left=left)
        ax3.text(left+0.01, 2, '%d'%(np.round(data,2)*100), color='black', 
            va='center')
        left += data
        i = i+1
    # Median gains    
    left, i = 0, 0
    for data, color in zip([(medianno2['AJWNE002']/
        medianno2['AJWBE001']).mean(),
        (medianno2['AJWNE003']/medianno2['AJWBE001']).mean(),      
        ((medianno2['AJWNE004']+medianno2['AJWNE005']+
         medianno2['AJWNE006']+medianno2['AJWNE007']+
        medianno2['AJWNE008'])/medianno2['AJWBE001']).mean()], colors):             
        ax3.barh(1, data, color=color, left=left)
        ax3.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='black', 
            va='center')    
        left += data
        i = i+1    
    # Smallest gains
    left, i = 0, 0
    for data, color in zip([(increaseno2['AJWNE002']/
        increaseno2['AJWBE001']).mean(),
        (increaseno2['AJWNE003']/increaseno2['AJWBE001']).mean(),      
        ((increaseno2['AJWNE004']+increaseno2['AJWNE005']+
         increaseno2['AJWNE006']+increaseno2['AJWNE007']+
        increaseno2['AJWNE008'])/increaseno2['AJWBE001']).mean()], colors):             
        ax3.barh(0, data, color=color, left=left)
        ax3.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='black', 
            va='center')    
        ax3.text(left+0.01, -1, labels[i], color=colors[i], va='center',
            fontweight='bold')
        left += data       
        i = i+1    
        
    # # # # Ethnic background
    # Largest gains
    left, i = 0, 0
    labels = ['Hispanic', 'Non-Hispanic']
    for data, color in zip([(decreaseno2['AJWWE003']/
        decreaseno2['AJWWE001']).mean(),
        (decreaseno2['AJWWE002']/decreaseno2['AJWWE001']).mean()], colors):             
        ax4.barh(2, data, color=color, left=left)
        ax4.text(left+0.01, 2, '%d'%(np.round(data,2)*100), color='black', 
            va='center')
        left += data
        i = i+1
    # Median gains    
    left, i = 0, 0
    for data, color in zip([(medianno2['AJWWE003']/
        medianno2['AJWWE001']).mean(),
        (medianno2['AJWWE002']/medianno2['AJWWE001']).mean()], colors):             
        ax4.barh(1, data, color=color, left=left)
        ax4.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='black', 
            va='center')
        left += data
        i = i+1
    # Median gains
    left, i = 0, 0
    for data, color in zip([(increaseno2['AJWWE003']/
        increaseno2['AJWWE001']).mean(),
        (increaseno2['AJWWE002']/increaseno2['AJWWE001']).mean()], colors): 
        ax4.barh(0, data, color=color, left=left)
        ax4.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='black', 
            va='center')    
        ax4.text(left+0.01, -1, labels[i], color=colors[i], va='center',
            fontweight='bold')              
        left += data
        i = i+1
        
    # # # # Educational attainment
    # Largest gains
    left = 0
    i = 0
    for data, color in zip([
        (decreaseno2.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        decreaseno2['AJYPE001']).mean(),
        (decreaseno2.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        decreaseno2['AJYPE001']).mean(),
        (decreaseno2.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        decreaseno2['AJYPE001']).mean()], colors):             
        ax5.barh(2, data, color=color, left=left)
        ax5.text(left+0.01, 2, '%d'%(np.round(data,2)*100), 
            color='black', va='center') 
        left += data
        i = i+1    
    # Median gains
    left = 0
    i = 0
    for data, color in zip([
        (medianno2.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        medianno2['AJYPE001']).mean(),
        (medianno2.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        medianno2['AJYPE001']).mean(),
        (medianno2.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        medianno2['AJYPE001']).mean()], colors):             
        ax5.barh(1, data, color=color, left=left)
        ax5.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='black', va='center') 
        left += data
        i = i+1        
    # Smallest gains
    left = 0
    i = 0
    labels = ['High school/GED or less', 'College or some college', 'Graduate']
    for data, color in zip([
        (increaseno2.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        increaseno2['AJYPE001']).mean(),
        (increaseno2.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        increaseno2['AJYPE001']).mean(),
        (increaseno2.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        increaseno2['AJYPE001']).mean()], colors):             
        ax5.barh(0, data, color=color, left=left)
        ax5.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='black', 
            va='center') 
        ax5.text(left+0.01, -1, labels[i], color=colors[i], va='center',
            fontweight='bold') 
        left += data
        i = i+1  
          
    # # # # Vehicle ownership
    # Largest gains
    left = 0
    i = 0
    labels = ['None', 'One or more']
    for data, color in zip([decreaseno2['FracNoCar'].mean(),
        (1-decreaseno2['FracNoCar'].mean())], colors):             
        ax6.barh(2, data, color=color, left=left)
        ax6.text(left+0.01, 2, '%d'%(np.round(data,2)*100),
            color='black', va='center') 
        left += data
        i = i+1        
    # Median gains
    left = 0
    i = 0
    for data, color in zip([medianno2['FracNoCar'].mean(),
        (1-medianno2['FracNoCar'].mean())], colors):                        
        ax6.barh(1, data, color=color, left=left)
        ax6.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='black', 
            va='center') 
        left += data
        i = i+1          
    # Smallest gains
    left = 0    
    i = 0
    for data, color in zip([increaseno2['FracNoCar'].mean(),
        (1-increaseno2['FracNoCar'].mean())], colors):             
        ax6.barh(0, data, color=color, left=left)
        ax6.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='black', 
            va='center') 
        ax6.text(left+0.01, -1, labels[i], color=colors[i], va='center',
            fontweight='bold')              
        left += data       
        i = i+1      
    # Aesthetics    
    ax1.set_xlim([-3e15, 1e14])
    ax1.set_xticks([])    
    # Limits for ax2 for urban [60000,80000]
    # Limits for ax2 for rural [50000,80000]    
    # ax2.set_xlim([50000,80000])
    ax2.set_xticks([])
    for ax in [ax3,ax4,ax5,ax6]:
        ax.set_xlim([0,1.])
        ax.set_xticks([])
        ax.set_xticklabels([])
    for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
        ax.set_ylim([-0.5,2.5])
        ax.set_yticks([0,1,2])
        ax.set_yticklabels([])
        ax.set_yticklabels(['Smallest gains', 'Median gains', 'Largest gains'], 
            rotation=45)
    for ax in [ax3, ax4, ax5, ax6]:
        for pos in ['right','top','bottom']:
            ax.spines[pos].set_visible(False)
            ax.spines[pos].set_visible(False)    
    plt.subplots_adjust(hspace=0.7, top=0.96, bottom=0.1)
    plt.savefig(DIR_FIGS+'bar_gains_ruraltracts.png', dpi=500)
    return 
    
# FIPS = ['01', '04', '05', '06', '08', '09', '10', '11', '12', '13', '16', 
#         '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
#         '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', 
#         '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50',
#         '51', '53', '54', '55', '56']
# harmonized = open_census_no2_harmonzied(FIPS)
# # Add vehicle ownership/road density data
# harmonized = merge_harmonized_vehicleownership(harmonized)
# # Split into rural and urban tracts
# harmonized_urban, harmonized_rural = split_harmonized_byruralurban(
#     harmonized)
# # Determine demographics in tracts
# demography, mostno2, leastno2, increaseno2, decreaseno2 = \
#     harmonized_demographics(harmonized, ptile_upper, ptile_lower)
# (demography_urban, mostno2_urban, leastno2_urban, increaseno2_urban, 
#     decreaseno2_urban) = harmonized_demographics(harmonized_urban, ptile_upper, 
#     ptile_lower)
# (demography_rural, mostno2_rural, leastno2_rural, increaseno2_rural, 
#     decreaseno2_rural) = harmonized_demographics(harmonized_rural, ptile_upper, 
#     ptile_lower)



# # Visualizations
# demographic_summarytable(demography, 'alltracts')
# demographic_summarytable(demography_rural, 'ruraltracts')
# demographic_summarytable(demography_urban, 'urbantracts')
# map_no2historic_no2gains(mostno2, leastno2, increaseno2, decreaseno2, FIPS)
# NO2_gridded_tractavg(lng_dg, lat_dg, no2_pre_dg, no2_post_dg, FIPS, harmonized)

# bar_no2historic_no2gains(demography, 'alltracts')
# bar_no2historic_no2gains(demography_urban, 'urbantracts')
# bar_no2historic_no2gains(demography_rural, 'ruraltracts')

# cdf(harmonized, 'alltracts')
# cdf(harmonized_urban, 'urbantracts')
# cdf(harmonized_rural, 'ruraltracts')
# lollipop(harmonized, harmonized_urban, harmonized_rural)
# (demography_city, mostno2_city, leastno2_city, increaseno2_city, 
#     decreaseno2_city) = harmonized_demographics(harmonized_city, 
#     ptile_upper, ptile_lower)
# demographic_summarytable(demography_city, 'newyork')

# hexbin(harmonized_urban)
# bar_gains(harmonized_rural)


decreaseno2 = harmonized_urban.loc[harmonized_urban['NO2_ABS']<
    np.nanpercentile(harmonized_urban['NO2_ABS'], ptile_lower)]


plt.hexbin(decreaseno2['AJZAE001'],decreaseno2['NO2_ABS'], 
    C=decreaseno2['FracNoCar'],cmap=plt.get_cmap('cividis'), vmin=0.00,
    vmax=0.75, gridsize=35);plt.colorbar()

plt.hexbin(harmonized_urban['Within51'],harmonized_urban['NO2_ABS'], 
    C=harmonized_urban['FracNoCar'],cmap=plt.get_cmap('cividis'), vmin=0.00,
    vmax=0.75, gridsize=35);plt.colorbar()

""" TABLE VALUES """ 
# from decimal import Decimal
# import numpy as np
# # # # # Number of tracts
# print('Number of tracts = %d' %(harmonized.shape[0]))
# print('Number of urban tracts = %d' %(harmonized_urban.shape[0]))
# print('Number of rural tracts = %d' %(harmonized_rural.shape[0]))
# # Table values 
# # Baseline NO2
# baseno2 = harmonized['PRENO2'].values
# baseno2_urban = harmonized_urban['PRENO2'].values
# baseno2_rural = harmonized_rural['PRENO2'].values
# print('Baseline NO2 (all) = %.2E +/- %.2E' %(Decimal(np.nanmean(baseno2)),
#     Decimal(np.nanstd(baseno2))))
# print('Baseline NO2 (urban) = %.2E +/- %.2E' %(
#     Decimal(np.nanmean(baseno2_urban)), Decimal(np.nanstd(baseno2_urban))))
# print('Baseline NO2 (rural) = %.2E +/- %.2E' %(
#     Decimal(np.nanmean(baseno2_rural)), Decimal(np.nanstd(baseno2_rural))))      
# # Lockdown NO2
# lockno2 = harmonized['POSTNO2'].values
# lockno2_urban = harmonized_urban['POSTNO2'].values
# lockno2_rural = harmonized_rural['POSTNO2'].values
# print('Lockdown NO2 (all) = %.2E +/- %.2E' %(Decimal(np.nanmean(lockno2)),
#     Decimal(np.nanstd(lockno2))))
# print('Lockdown NO2 (urban) = %.2E +/- %.2E' %(
#     Decimal(np.nanmean(lockno2_urban)), Decimal(np.nanstd(lockno2_urban))))
# print('Lockdown NO2 (rural) = %.2E +/- %.2E' %(
#     Decimal(np.nanmean(lockno2_rural)), Decimal(np.nanstd(lockno2_rural))))      
# # # # # Race 
# white = harmonized['AJWNE002']/harmonized['AJWBE001']
# white_urban = harmonized_urban['AJWNE002']/harmonized_urban['AJWBE001']
# white_rural = harmonized_rural['AJWNE002']/harmonized_rural['AJWBE001']
# print('Fraction white (all) = %.3f +/- %.3f' %(np.nanmean(white),
#     np.nanstd(white)))
# print('Fraction white (urban) = %.3f +/- %.3f' %(np.nanmean(white_urban),
#     np.nanstd(white_urban)))
# print('Fraction white (rural) = %.3f +/- %.3f' %(np.nanmean(white_rural),
#     np.nanstd(white_rural)))
# black = harmonized['AJWNE003']/harmonized['AJWBE001']
# black_urban = harmonized_urban['AJWNE003']/harmonized_urban['AJWBE001']
# black_rural = harmonized_rural['AJWNE003']/harmonized_rural['AJWBE001']
# print('Fraction black (all) = %.3f +/- %.3f' %(np.nanmean(black),
#     np.nanstd(black)))
# print('Fraction black (urban) = %.3f +/- %.3f' %(np.nanmean(black_urban),
#     np.nanstd(black_urban)))
# print('Fraction black (rural) = %.3f +/- %.3f' %(np.nanmean(black_rural),
#     np.nanstd(black_rural)))
# # n.b. Other for all tracts is dominated by Asian (4.79%), Some other race
# # (4.66%), and Two or more races (3.07%)
# other = (harmonized['AJWNE004']+harmonized['AJWNE005']+
#     harmonized['AJWNE006']+harmonized['AJWNE007']+harmonized['AJWNE008']
#     )/harmonized['AJWBE001']
# # n.b. Other for all tracts is dominated by Asian (8.12%), Some other race
# # (7.50%), and Two or more races (3.4%)
# other_urban = (harmonized_urban['AJWNE004']+harmonized_urban['AJWNE005']+
#     harmonized_urban['AJWNE006']+harmonized_urban['AJWNE007']+
#     harmonized_urban['AJWNE008'])/harmonized_urban['AJWBE001']
# # n.b. Other for all tracts is dominated by Two or more races (2.79%), 
# # Some other race alone (2.48%), and Asian (2.2%)
# other_rural = (harmonized_rural['AJWNE004']+harmonized_rural['AJWNE005']+
#     harmonized_rural['AJWNE006']+harmonized_rural['AJWNE007']+
#     harmonized_rural['AJWNE008'])/harmonized_rural['AJWBE001']
# print('Fraction other (all) = %.3f +/- %.3f' %(np.nanmean(other),
#     np.nanstd(other)))
# print('Fraction other (urban) = %.3f +/- %.3f' %(np.nanmean(other_urban),
#     np.nanstd(other_urban)))
# print('Fraction other (rural) = %.3f +/- %.3f' %(np.nanmean(other_rural),
#     np.nanstd(other_rural)))
# # # # # Hispanic or Latino origin 
# hispanic = harmonized['AJWWE003']/harmonized['AJWWE001']
# hispanic_urban = harmonized_urban['AJWWE003']/harmonized_urban['AJWWE001']
# hispanic_rural = harmonized_rural['AJWWE003']/harmonized_rural['AJWWE001']
# print('Fraction hispanic or latino origin (all) = %.3f +/- %.3f' %(
#     np.nanmean(hispanic),  np.nanstd(hispanic)))
# print('Fraction hispanic or latino origin (urban) = %.3f +/- %.3f' %(
#     np.nanmean(hispanic_urban), np.nanstd(hispanic_urban)))
# print('Fraction hispanic or latino origin (rural) = %.3f +/- %.3f' %(
#     np.nanmean(hispanic_rural), np.nanstd(hispanic_rural)))
# # # # # Educational Attainment
# # n.b. For all tracts, 13.0% have no high school diploma, 23.8% have a 
# # regular high school diploma, and 4.1% have GED/alternative
# secondary = (harmonized.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
#     harmonized['AJYPE001'])
# secondary_urban = (harmonized_urban.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
#     harmonized_urban['AJYPE001'])
# secondary_rural = (harmonized_rural.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
#     harmonized_rural['AJYPE001'])
# print('Fraction secondary education or less (all) = %.3f +/- %.3f' %(
#     np.nanmean(secondary), np.nanstd(secondary)))
# print('Fraction secondary education or less (urban) = %.3f +/- %.3f' %(
#     np.nanmean(secondary_urban), np.nanstd(secondary_urban)))
# print('Fraction secondary education or less (rural) = %.3f +/- %.3f' %(
#     np.nanmean(secondary_rural), np.nanstd(secondary_rural)))
# # n.b. For all tracts, 18.5% have a bachelor's degree, 8.27% have an 
# # associate's degree; 14.4% have some college (1 or more years, no degree), 
# # and 6.1% have some college (less than 1 year)
# college = (harmonized.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
#     harmonized['AJYPE001'])
# college_urban = (harmonized_urban.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
#     harmonized_urban['AJYPE001'])
# college_rural = (harmonized_rural.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
#     harmonized_rural['AJYPE001'])
# print('Fraction college education (all) = %.3f +/- %.3f' %(
#     np.nanmean(college), np.nanstd(college)))
# print('Fraction college education (urban) = %.3f +/- %.3f' %(
#     np.nanmean(college_urban), np.nanstd(college_urban)))
# print('Fraction college education (rural) = %.3f +/- %.3f' %(
#     np.nanmean(college_rural), np.nanstd(college_rural)))
# # n.b. For all tracts, 1.3% have a doctorate degree, 2.0% have a 
# # professional school degree, and 8.23% have a master's degree
# grad = (harmonized.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
#     harmonized['AJYPE001'])
# grad_urban = (harmonized_urban.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
#     harmonized_urban['AJYPE001'])
# grad_rural = (harmonized_rural.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
#     harmonized_rural['AJYPE001'])
# print('Fraction graduate education (all) = %.3f +/- %.3f' %(
#     np.nanmean(grad), np.nanstd(grad)))
# print('Fraction graduate education (urban) = %.3f +/- %.3f' %(
#     np.nanmean(grad_urban), np.nanstd(grad_urban)))
# print('Fraction graduate education (rural) = %.3f +/- %.3f' %(
#     np.nanmean(grad_rural), np.nanstd(grad_rural)))
# # # # # Median household income
# income = harmonized['AJZAE001']
# income_urban = harmonized_urban['AJZAE001']
# income_rural = harmonized_rural['AJZAE001']
# print('Median household income (all) = %.3f +/- %.3f' %(
#     np.nanmean(income), np.nanstd(income)))
# print('Median household income (urban) = %.3f +/- %.3f' %(
#     np.nanmean(income_urban), np.nanstd(income_urban)))
# print('Median household income (rural) = %.3f +/- %.3f' %(
#     np.nanmean(income_rural), np.nanstd(income_rural)))
# # # # # Household vehicle ownership 
# nocar = harmonized['FracNoCar']
# nocar_urban = harmonized_urban['FracNoCar']
# nocar_rural = harmonized_rural['FracNoCar']
# print('Household vehicle ownership (all) = %.3f +/- %.3f' %(
#     np.nanmean(nocar), np.nanstd(nocar)))
# print('Household vehicle ownership (urban) = %.3f +/- %.3f' %(
#     np.nanmean(nocar_urban), np.nanstd(nocaør_urban)))
# print('Household vehicle ownership (rural) = %.3f +/- %.3f' %(
#     np.nanmean(nocar_rural), np.nanstd(nocar_rural)))


"""CASE STUDY OF LOS ANGELES"""        
# from scipy import stats
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import netCDF4 as nc
# import numpy as np
# import matplotlib as mpl
# from matplotlib.gridspec import GridSpec
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.io import shapereader
# losangeles = ['06037','06059']
# # Find harmonzied data in city
# harmonized_city = subset_harmonized_bycountyfips(harmonized_urban, 
#     losangeles)
# # Find CEMS data in city
# cems_ca = open_cems(['06'], counties=['Los Angeles County', 'Orange County'])
# cems_loc = cems_ca.groupby([' Facility Latitude',' Facility Longitude']
#     ).size().reset_index(name='Freq')
# cems_ca = cems_ca.groupby([' Date']).sum()
# # Find VMT data in city
# streetlights = load_streetlights(countytimeavg=False)
# streetlights_la = streetlights.loc[streetlights['GEOID'].isin(losangeles)]
# streetlights_la_mean = streetlights_la.groupby(['ref_dt']).mean()
# # Initialize figure, axes
# fig = plt.figure(figsize=(8,5.5))
# ax1 = plt.subplot2grid((2,3),(0,0), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# ax2 = plt.subplot2grid((2,3),(0,1), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# ax3 = plt.subplot2grid((2,3),(0,2), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# ax4 = plt.subplot2grid((2,3),(1,0), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# ax5 = plt.subplot2grid((2,3),(1,1), colspan=2)
# proj = ccrs.PlateCarree(central_longitude=0.0)
# # Create discrete colormaps
# cmapincome = plt.get_cmap('YlGnBu', 9)
# normincome = matplotlib.colors.Normalize(vmin=40000, vmax=100000)
# cmapwhite = plt.get_cmap('YlGnBu', 8)
# normwhite = matplotlib.colors.Normalize(vmin=0, vmax=100)
# cmapbase = plt.get_cmap('YlGnBu', 7)
# normbase = matplotlib.colors.Normalize(vmin=2e15, vmax=9e15)
# cmaplock = plt.get_cmap('Blues_r', 8)
# normlock = matplotlib.colors.Normalize(vmin=-4e15, vmax=0e15)
# # Open shapefile
# shp = shapereader.Reader(DIR_GEO+'tigerline/'+
#     'tl_2019_06_tract/tl_2019_06_tract')
# records = shp.records()
# tracts = shp.geometries()
# records = list(records)
# tracts = list(tracts)
# # Find records and tracts in city 
# geoids_records = [x.attributes['GEOID'] for x in records]
# geoids_records = np.where(np.in1d(np.array(geoids_records), 
#     harmonized_city.index)==True)[0]
# # Slice records and tracts for only entries in city
# records = list(np.array(records)[geoids_records])
# tracts = list(np.array(tracts)[geoids_records])
# # Loop through shapefiles in state
# for geoid in harmonized_city.index:
#     print(geoid)
#     where_geoid = np.where(np.array([x.attributes['GEOID'] for x in 
#         records])==geoid)[0][0]
#     tract = tracts[where_geoid]
#     # Find demographic data/TROPOMI data in tract
#     harmonized_tract = harmonized_city.loc[harmonized_city.index.isin(
#         [geoid])]
#     baseline_tract = harmonized_tract['PRENO2'].values[0]
#     lockdown_tract = (harmonized_tract['POSTNO2'].values[0]-
#         harmonized_tract['PRENO2'].values[0])
#     income_tract = harmonized_tract['AJZAE001'].values[0]
#     white_tract = (harmonized_tract['AJWNE002'].values[0]/
#         harmonized_tract['AJWBE001'].values[0])*100.
#     # For NO2
#     if np.isnan(baseline_tract)==True:
#         ax3.add_geometries(tract, proj, facecolor=missingdata, 
#             edgecolor='None', zorder=10, rasterized=True)
#         ax4.add_geometries(tract, proj, facecolor=missingdata, 
#             edgecolor='None', zorder=10, rasterized=True)
#     else: 
#         ax3.add_geometries(tract, proj, facecolor=cmapbase(
#             normbase(baseline_tract)), edgecolor='None', alpha=1.,
#             zorder=10, rasterized=True)
#         ax4.add_geometries(tract, proj, facecolor=cmaplock(
#             normlock(lockdown_tract)), edgecolor='None', alpha=1.,
#             zorder=10, rasterized=True)
#     # For demographics
#     if np.isnan(income_tract)==True:
#         ax1.add_geometries(tract, proj, facecolor=missingdata, 
#             edgecolor='None', zorder=10, rasterized=True)
#         ax2.add_geometries(tract, proj, facecolor=missingdata, 
#             edgecolor='None', zorder=10, rasterized=True)
#     else:     
#         ax1.add_geometries(tract, proj, facecolor=cmapincome(
#             normincome(income_tract)), edgecolor='None', zorder=10, 
#             rasterized=True)
#         ax2.add_geometries(tract, proj, facecolor=cmapwhite(
#             normwhite(white_tract)), edgecolor='None', zorder=10, 
#             rasterized=True)
# # Add CEMS data        
# ax4.plot(cems_loc[' Facility Longitude'].values, 
#     cems_loc[' Facility Latitude'].values, 'ko', zorder=25, 
#     transform=proj, markersize=3)
# # Add primary and secondary roads
# shp = shapereader.Reader('/Users/ghkerr/Downloads/tl_2018_06_prisecroads/'+
#     'tl_2018_06_prisecroads')
# roads_records = list(shp.records())
# roads = list(shp.geometries())
# # Select only interstates
# roads_rttyp = [x.attributes['RTTYP'] for x in roads_records]
# where_interstate = np.where(np.array(roads_rttyp)=='I')[0]
# roads_i = []
# roads_i += [roads[x] for x in where_interstate]
# roads = cfeature.ShapelyFeature(roads_i, proj)
# ax4.add_feature(roads, facecolor='None', edgecolor='r', zorder=11, lw=0.5)
# # Add counties 
# reader = shapereader.Reader(DIR_GEO+'counties/tl_2019_us_county/'+
#     'tl_2019_us_county')
# counties = list(reader.geometries())
# records = list(reader.records())
# # Select only counties inside/outside city for masking
# county_in = [x.attributes['GEOID'] for x in records]
# county_out = np.where(np.in1d(np.array(county_in), np.array(losangeles))==False)
# county_in = np.where(np.in1d(np.array(county_in), np.array(losangeles))==True)
# county_in = cfeature.ShapelyFeature(np.array(counties)[county_in], proj)
# county_out = cfeature.ShapelyFeature(np.array(counties)[county_out], proj)
# for ax in [ax1, ax2, ax3, ax4]:
#     ax.add_feature(county_in, facecolor='None', edgecolor='k', zorder=16)
#     ax.add_feature(county_out, facecolor='w', edgecolor='w', zorder=12)    
#     ax.set_extent([-119, -117.4, 33.3, 34.9], proj)
#     # Cover Santa Catalina Island
#     latsci = 33.3879
#     lngsci = -118.4163
#     lat_corners = np.array([latsci-0.3, latsci+0.2, latsci+0.2, latsci-0.3])
#     lon_corners = np.array([lngsci-0.3, lngsci-0.3, lngsci+0.3, lngsci+0.3])
#     poly_corners = np.zeros((len(lat_corners), 2), np.float64)
#     poly_corners[:,0] = lon_corners
#     poly_corners[:,1] = lat_corners
#     poly = mpatches.Polygon(poly_corners, closed=True, ec='w', fill=True, 
#         lw=1, fc='w', transform=proj, zorder=25)
#     ax.add_patch(poly)
#     ax.background_patch.set_visible(False)
#     ax.outline_patch.set_visible(False)
# ax5.plot(cems_ca['2020-03-01':'2020-05-12'][' NOx (tons)'].values, '-k', 
#     zorder=5)
# ax5t = ax5.twinx()
# ax5t.plot(streetlights_la_mean['2020-03-01':'2020-05-12']['county_vmt'].values, 
#     '-r', zorder=6)
# ax5t.set_xlim([0, streetlights_la_mean.index.shape[0]-1])
# ax5t.set_xticks([0,14,31,45,61,72])
# ax5t.set_xticklabels(['1 March', '15 March', '1 April', '15 April', '1 May', '12 May'])
# # Stay at home order
# ax5.vlines(18, ymin=ax5.get_ylim()[0], ymax=ax5.get_ylim()[1], 
#     zorder=0, color='darkgrey')
# ax5t.text(19, 1.25e8, 'Stay-at-\nhome order', fontsize=12, 
#     color='darkgrey', rotation='vertical')
# # Label VMT
# ax5.text(0, 1.9, 'Traffic', fontsize=12, color='r')
# ax5t.text(0, 0.84e8, 'Industry', fontsize=12, color='k')
# # Aesthetics
# ax5.tick_params(axis='y', which='both', right=False, left=False, 
#     labelleft=False)
# ax5t.tick_params(axis='y', which='both', right=False, left=False, 
#     labelright=False)
# for pos in ['right','top','left']:
#     ax5.spines[pos].set_visible(False)
#     ax5t.spines[pos].set_visible(False)
# ax5t.yaxis.offsetText.set_visible(False)
# plt.subplots_adjust(left=0.05, top=0.93, hspace=0.6)
# # Add colorbars 
# # (a)
# caxincome = fig.add_axes([ax1.get_position().x0,
#     ax1.get_position().y0-0.03, 
#     ax1.get_position().x1-ax1.get_position().x0, 0.02])
# mpl.colorbar.ColorbarBase(caxincome, cmap=cmapincome, norm=normincome, 
#     spacing='proportional', orientation='horizontal', extend='both')
# # (b)
# caxwhite = fig.add_axes([ax2.get_position().x0,
#     ax2.get_position().y0-0.03, 
#     ax2.get_position().x1-ax2.get_position().x0, 0.02])
# mpl.colorbar.ColorbarBase(caxwhite, cmap=cmapwhite, norm=normwhite, 
#     spacing='proportional', orientation='horizontal', extend='neither')
# # (c)
# caxbase = fig.add_axes([ax3.get_position().x0,
#     ax3.get_position().y0-0.03, 
#     (ax3.get_position().x1-ax3.get_position().x0), 0.02])
# mpl.colorbar.ColorbarBase(caxbase, cmap=cmapbase, norm=normbase, 
#     spacing='proportional', orientation='horizontal', extend='both')
# caxbase.xaxis.offsetText.set_visible(False)
# # (d)
# caxlock = fig.add_axes([ax4.get_position().x0, 
#     ax4.get_position().y0-0.03, 
#     (ax4.get_position().x1-ax4.get_position().x0), 0.02])
# mpl.colorbar.ColorbarBase(caxlock, cmap=cmaplock, norm=normlock, 
#     spacing='proportional', orientation='horizontal', extend='both')
# caxlock.xaxis.offsetText.set_visible(False)
# ax1.set_title('(a) Income [$]', loc='left')
# ax2.set_title('(b) Race [%]', loc='left')
# ax3.set_title('(c) NO$_{2}$ [molec cm$^{-2}$]', loc='left')
# ax4.set_title('(d) $\Delta\:$NO$_{2}$ [molec cm$^{-2}$]', loc='left')
# ax5.set_title('(e)',loc='left')
# plt.savefig('/Users/ghkerr/Desktop/losangelescasestudy.png', dpi=500)
# plt.show()







# # Add U.S. counties 
# reader = shapereader.Reader(DIR_GEO+'counties/countyl010g_shp/'+
#     'countyl010g')
# counties = list(reader.geometries())
# counties = cfeature.ShapelyFeature(counties, proj)
#     # Add counties
#     ax.add_feature(counties, facecolor='none', edgecolor='k')
# # Make bivariate colormap
# harmonized_city.insert(harmonized_city.shape[1], 'HEXBIVAR', 
#     '')
# # HIGH INCOME
# # %-tile white < 33.3 and 33.3 < %-tile income > 66.6
# harmonized_city.loc[(harmonized_city['PercentileWhite'].between(0,33.3,inclusive=False)) & 
#     (harmonized_city['PercentileIncome'].between(66.6,100,inclusive=False)), 'HEXBIVAR'] = '#f3b300'
# # 33.3 < %-tile white < 66.6 and %-tile income > 66.6
# harmonized_city.loc[(harmonized_city['PercentileWhite'].between(33.3,66.6)) & 
#     (harmonized_city['PercentileIncome'].between(66.6,100,inclusive=False)), 'HEXBIVAR'] = '#f3e6b3'
# # %-tile white > 66.6 and %-tile income > 66.6
# harmonized_city.loc[(harmonized_city['PercentileWhite'].between(66.6,100,inclusive=False)) & 
#     (harmonized_city['PercentileIncome'].between(66.6,100,inclusive=False)), 'HEXBIVAR'] = '#f3f3f3'
# # MID INCOME
# # %-tile white < 33.3 and 33.3 <= %-tile income <= 66.6
# harmonized_city.loc[(harmonized_city['PercentileWhite'].between(0,33.3,inclusive=False)) & 
#     (harmonized_city['PercentileIncome'].between(33.3,66.6)), 'HEXBIVAR'] = '#b36600'
# # 33.3 < %-tile white < 66.6 and 33.3 < %-tile income < 66.6
# harmonized_city.loc[(harmonized_city['PercentileWhite'].between(33.3,66.6)) & 
#     (harmonized_city['PercentileIncome'].between(33.3,66.6)), 'HEXBIVAR'] = '#b3b3b3'
# # %-tile white > 66.6 and 33.3 < %-tile income > 66.6
# harmonized_city.loc[(harmonized_city['PercentileWhite'].between(66.6,100,inclusive=False)) & 
#     (harmonized_city['PercentileIncome'].between(33.3,66.6)), 'HEXBIVAR'] = '#b4d3e1'
# # LOW INCOME 
# # %-tile white < 33.3 and %-tile income < 33.3
# harmonized_city.loc[(harmonized_city['PercentileWhite'].between(0,33.3,inclusive=False)) & 
#     (harmonized_city['PercentileIncome'].between(0,33.3,inclusive=False)), 'HEXBIVAR'] = '#000000'
# # 33.3 <= %-tile white <= 66.6 and %-tile income < 33.3
# harmonized_city.loc[(harmonized_city['PercentileWhite'].between(33.3,66.6)) & 
#     (harmonized_city['PercentileIncome'].between(0,33.3,inclusive=False)), 'HEXBIVAR'] = '#376387'
# # %-tile white > 66.6 and %-tile income < 33.3
# harmonized_city.loc[(harmonized_city['PercentileWhite'].between(66.6,100,inclusive=False)) & 
#     (harmonized_city['PercentileIncome'].between(0,33.3,inclusive=False)), 'HEXBIVAR'] = '#509dc2'
# # Add new columns to correspond to the percentile of fraction white and 
# # income 
# frac_white = (harmonized_city['AJWNE002']/harmonized_city['AJWBE001'])
# ptile_white = [stats.percentileofscore(frac_white, x) for x in frac_white]
# ptile_income = [stats.percentileofscore(harmonized_city['AJZAE001'], x) 
#     for x in harmonized_city['AJZAE001']]
# harmonized_city.insert(harmonized_city.shape[1], 'PercentileWhite', 
#     ptile_white)
# harmonized_city.insert(harmonized_city.shape[1], 'PercentileIncome', 
#     ptile_income)
# # Arrows above plot
# caxdemo.annotate('', xy=(0.50,1.5), 
#     xycoords='axes fraction', xytext=(1.0,1.5),
#     arrowprops=dict(arrowstyle= '<|-', color='k', lw=1), va='center', 
#     transform=fig.transFigure)
# caxdemo.annotate('', xy=(0,1.5), 
#     xycoords='axes fraction', xytext=(0.5,1.5),
#     arrowprops=dict(arrowstyle= '-|>', color='k', lw=1), va='center', 
#     transform=fig.transFigure)
# # ax1.annotate('Highest NO$_{2}$ pollution', xy=(0.06,1.55), 
#     # xycoords='axes fraction', va='center', fontsize=12,
#     # transform=fig.transFigure)

