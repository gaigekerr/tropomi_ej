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

# Load font
import sys
if 'mpl' not in sys.modules:
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(
            fname='/Users/ghkerr/Library/Fonts/cmunbmr.ttf')
    matplotlib.rcParams['font.family'] = prop.get_name()
    prop = matplotlib.font_manager.FontProperties(
        fname='/Users/ghkerr/Library/Fonts/cmunbbx.ttf')
    matplotlib.rcParams['mathtext.bf'] = prop.get_name()
    prop = matplotlib.font_manager.FontProperties(
        fname='/Users/ghkerr/Library/Fonts/cmunbmr.ttf')
    matplotlib.rcParams['mathtext.it'] = prop.get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False

# import netCDF4 as nc
# # 13 March - 13 June 2019 and 2020 average NO2
# no2_pre_dg = nc.Dataset(DIR_TROPOMI+
#     'Tropomi_NO2_griddedon0.01grid_Mar13-Jun132019_precovid19_QA75.ncf')
# no2_pre_dg = no2_pre_dg['NO2'][:]
# no2_post_dg = nc.Dataset(DIR_TROPOMI+
#     'Tropomi_NO2_griddedon0.01grid_Mar13-Jun132020_postcovid19_QA75.ncf')
# no2_post_dg = no2_post_dg['NO2'][:].data
# lat_dg = nc.Dataset(DIR_TROPOMI+'LatLonGrid.ncf')['LAT'][:].data
# lng_dg = nc.Dataset(DIR_TROPOMI+'LatLonGrid.ncf')['LON'][:].data

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
    import cartopy.io.shapereader as shpreader
    from cartopy.io import shapereader
    from shapely.geometry import Polygon
    from cartopy.io import shapereader
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
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
    cmaplock = plt.get_cmap('coolwarm', 12)
    normlock = matplotlib.colors.Normalize(vmin=-2e15, vmax=2e15)
    # Plot gridded NO2 retrivals 
    mb1 = ax1.pcolormesh(lng_dg, lat_dg, no2_pre_dg, cmap=cmapbase, 
        norm=normbase, transform=proj, rasterized=True)
    mb2 = ax2.pcolormesh(lng_dg, lat_dg, (no2_post_dg-no2_pre_dg), 
        cmap=cmaplock, norm=normlock, transform=proj, rasterized=True)
    for FIPS_i in FIPS: 
        print(FIPS_i)
        # Tigerline shapefile for state
        shp = shapereader.Reader(DIR_GEO+
            'tigerline/tl_2019_%s_tract/tl_2019_%s_tract'%(FIPS_i, FIPS_i))
        records = shp.records()
        tracts = shp.geometries()
        for record, tract in zip(records, tracts):
            # Find GEOID of tract
            gi = record.attributes['GEOID']
            # Look up harmonized NO2-census data for tract
            harmonized_tract = harmonized.loc[harmonized.index.isin([gi])]
            baseline = harmonized_tract.PRENO2.values[0]
            lockdown = harmonized_tract.NO2_ABS.values[0]
            if np.isnan(baseline)==True:
                ax3.add_geometries(tract, proj, facecolor='grey', 
                    edgecolor='grey', alpha=1., linewidth=0.1, rasterized=True)
                ax4.add_geometries(tract, proj, facecolor='grey', 
                    edgecolor='grey', alpha=1., linewidth=0.1, rasterized=True)
            else:
                ax3.add_geometries(tract, proj, facecolor=cmapbase(
                    normbase(baseline)), edgecolor=cmapbase(
                    normbase(baseline)), alpha=1., linewidth=0.1, rasterized=True)
                ax4.add_geometries(tract, proj, facecolor=cmaplock(
                    normlock(lockdown)), edgecolor=cmaplock(
                    normlock(lockdown)), alpha=1., linewidth=0.1, rasterized=True)
    # Add borders, set map extent, etc. 
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_extent([-125,-66.5, 24.5, 49.48], proj)        
        # Add coasts 
        ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='50m',
            facecolor='none', name='coastline', lw=0.25, 
            edgecolor='k'))
        # # Add national borders
        # ax.add_feature(cfeature.NaturalEarthFeature('cultural', 
        #     'admin_0_countries', '50m', edgecolor='k', lw=0.25, 
        #     facecolor='none'))
        # Color all countries besides U.S.
        shpfilename = shpreader.natural_earth(resolution='50m',
            category='cultural', name='admin_0_countries')
        reader = shpreader.Reader(shpfilename)
        countries = reader.records()               
        for country in countries:
            if country.attributes['ADM0_A3'] != 'USA':
                ax.add_geometries(country.geometry, ccrs.PlateCarree(),
                    facecolor='w', edgecolor='w', lw=0., zorder=100)
            else: 
                ax.add_geometries(country.geometry, ccrs.PlateCarree(),
                    facecolor='None', edgecolor='k', lw=0.25, zorder=100)                
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
    ax1.set_title('(a)', loc='left', fontsize=12)
    ax2.set_title('(b)', loc='left', fontsize=12)
    ax3.set_title('(c)', loc='left', fontsize=12)
    ax4.set_title('(d)', loc='left', fontsize=12)
    # Colorbars
    caxbase = fig.add_axes([ax3.get_position().x0, 
        ax3.get_position().y0-0.02, 
        (ax3.get_position().x1-ax3.get_position().x0), 0.02])
    cb = mpl.colorbar.ColorbarBase(caxbase, cmap=cmapbase, norm=normbase, 
        spacing='proportional', orientation='horizontal', extend='max', 
        label='NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]')
    caxbase.xaxis.offsetText.set_visible(False)
    caxbase = fig.add_axes([ax4.get_position().x0, 
        ax4.get_position().y0-0.02, (ax4.get_position().x1-ax4.get_position().x0), 
        0.02])
    cb = mpl.colorbar.ColorbarBase(caxbase, cmap=cmaplock, norm=normlock, 
        spacing='proportional', orientation='horizontal', extend='both', 
        label='$\Delta\:$NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]')
    caxbase.xaxis.offsetText.set_visible(False)
    plt.subplots_adjust(bottom=0.2, top=0.95)
    plt.savefig(DIR_FIGS+'NO2_gridded_tractavg_absolute.pdf', dpi=600)
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
        np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    # Historic NO2 for white and non-white populations
    ax.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    # Draw connection lines
    ax.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
        np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
        np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))[0]), 
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
        np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
        np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
        np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))[0]), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7  
    # For urban tracts
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
        np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
        np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
        np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))[0]), 
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
            np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))[0]), 
            [i-os,i-os], color='k', ls='-', zorder=10)    
        # i = i+2
        # Historic NO2 for white and non-white populations
        ax.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
        ax.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
        # Draw connection lines
        ax.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
            np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
            np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))[0]), 
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
        np.abs(np.diff([mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    # Historic NO2 for white and non-white populations
    ax.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    # Draw connection lines
    ax.plot((np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()]), 
        np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
        np.abs(np.diff([mostwealthy['POSTNO2'].mean(), leastwealthy['POSTNO2'].mean()]))[0]), 
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
        np.abs(np.diff([mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()]), 
        np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
        np.abs(np.diff([mostwealthy['POSTNO2'].mean(), leastwealthy['POSTNO2'].mean()]))[0]), 
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
        np.abs(np.diff([mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()]), 
        np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
        np.abs(np.diff([mostwealthy['POSTNO2'].mean(), leastwealthy['POSTNO2'].mean()]))[0]), 
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
            [mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
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
            leastwealthy['POSTNO2'].mean()]))[0]), [i+os,i+os], color='k', 
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
        np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    # Historic NO2 for white and non-white populations
    ax.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    # Draw connection lines
    ax.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
        np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
        np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
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
        np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    # Historic NO2 for white and non-white populations
    ax.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    # Draw connection lines
    ax.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
        np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
        np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
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
        np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    # Historic NO2 for white and non-white populations
    ax.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    # Draw connection lines
    ax.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
        np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
        np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
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
            [mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
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
            [mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
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
    
    # Initialize figure
    fig = plt.figure(figsize=(6,6.5))
    ax = plt.subplot2grid((1,1),(0,0))
    i = 0 
    yticks = []
    # For all tracts
    frac = 1-(harmonized['AJWWE003']/harmonized['AJWWE001'])
    most = harmonized.iloc[np.where(frac > 
        np.nanpercentile(frac, ptile_upper))]
    least = harmonized.iloc[np.where(frac < 
        np.nanpercentile(frac, ptile_lower))]
    ax.plot(most['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
    ax.plot(least['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
    ax.plot((np.min([most['PRENO2'].mean(),least['PRENO2'].mean()]), 
        np.min([most['PRENO2'].mean(),least['PRENO2'].mean()])+
        np.abs(np.diff([most['PRENO2'].mean(), least['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax.plot(most['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax.plot(least['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    ax.plot((np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()]), 
        np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()])+
        np.abs(np.diff([most['POSTNO2'].mean(), least['POSTNO2'].mean()]))[0]), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7    
    # For rural tracts
    frac = 1-(harmonized_rural['AJWWE003']/harmonized_rural['AJWWE001'])
    most = harmonized_rural.iloc[np.where(frac > 
        np.nanpercentile(frac, ptile_upper))]
    least = harmonized_rural.iloc[np.where(frac < 
        np.nanpercentile(frac, ptile_lower))]
    ax.plot(most['PRENO2'].mean(), i-os, 'o', color=color_white, 
        zorder=12)
    ax.plot(least['PRENO2'].mean(), i-os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([most['PRENO2'].mean(),least['PRENO2'].mean()]), 
        np.min([most['PRENO2'].mean(),least['PRENO2'].mean()])+
        np.abs(np.diff([most['PRENO2'].mean(), least['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax.plot(most['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax.plot(least['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()]), 
        np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()])+
        np.abs(np.diff([most['POSTNO2'].mean(), least['POSTNO2'].mean()]))[0]), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7  
    # For urban tracts
    frac = 1-(harmonized_urban['AJWWE003']/harmonized_urban['AJWWE001'])
    most = harmonized_urban.iloc[np.where(frac > 
        np.nanpercentile(frac, ptile_upper))]
    least = harmonized_urban.iloc[np.where(frac < 
        np.nanpercentile(frac, ptile_lower))]
    ax.plot(most['PRENO2'].mean(), i-os, 'o', color=color_white, 
        zorder=12)
    ax.plot(least['PRENO2'].mean(), i-os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([most['PRENO2'].mean(),least['PRENO2'].mean()]), 
        np.min([most['PRENO2'].mean(),least['PRENO2'].mean()])+
        np.abs(np.diff([most['PRENO2'].mean(), least['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax.plot(most['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax.plot(least['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()]), 
        np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()])+
        np.abs(np.diff([most['POSTNO2'].mean(), least['POSTNO2'].mean()]))[0]), 
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
        harmonized_city = subset_harmonized_bycountyfips(harmonized, city)
        frac = 1-(harmonized_city['AJWWE003']/harmonized_city['AJWWE001'])
        most = harmonized_city.iloc[np.where(frac > 
            np.nanpercentile(frac, ptile_upper))]
        least = harmonized_city.iloc[np.where(frac < 
            np.nanpercentile(frac, ptile_lower))]
        ax.plot(most['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
        ax.plot(least['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
        ax.plot((np.min([most['PRENO2'].mean(),least['PRENO2'].mean()]), 
            np.min([most['PRENO2'].mean(),least['PRENO2'].mean()])+
            np.abs(np.diff([most['PRENO2'].mean(), least['PRENO2'].mean()]))[0]), 
            [i-os,i-os], color='k', ls='-', zorder=10)    
        ax.plot(most['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
        ax.plot(least['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
        ax.plot((np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()]), 
            np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()])+
            np.abs(np.diff([most['POSTNO2'].mean(), least['POSTNO2'].mean()]))[0]), 
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
    for side in ['right', 'left', 'top', 'bottom']:
        ax.spines[side].set_visible(False)
    ax.grid(axis='x', zorder=0, color='darkgrey')
    custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
        Line2D([0], [0], marker='o', color=color_non, lw=0),  
        Line2D([0], [0], color='k', lw=1.0),
        Line2D([0], [0], color='k', ls='--', lw=1)]
    ax.legend(custom_lines, ['Least Hispanic', 'Most Hispanic', 'Baseline', 'Lockdown'], 
        bbox_to_anchor=(0.4, -0.07), loc=8, ncol=4, frameon=False)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.20, right=0.96, bottom=0.05, top=0.9)
    plt.savefig(DIR_FIGS+'lollipop_onethnicity.png', dpi=500)
    plt.show()    
    
    # Initialize figure
    fig = plt.figure(figsize=(6,6.5))
    ax = plt.subplot2grid((1,1),(0,0))
    i = 0 
    yticks = []
    # For all tracts
    frac = 1-harmonized['FracNoCar']
    most = harmonized.iloc[np.where(frac > 
        np.nanpercentile(frac, ptile_upper))]
    least = harmonized.iloc[np.where(frac < 
        np.nanpercentile(frac, ptile_lower))]
    ax.plot(most['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
    ax.plot(least['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
    ax.plot((np.min([most['PRENO2'].mean(),least['PRENO2'].mean()]), 
        np.min([most['PRENO2'].mean(),least['PRENO2'].mean()])+
        np.abs(np.diff([most['PRENO2'].mean(), least['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax.plot(most['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax.plot(least['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    ax.plot((np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()]), 
        np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()])+
        np.abs(np.diff([most['POSTNO2'].mean(), least['POSTNO2'].mean()]))[0]), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7    
    # For rural tracts
    frac = 1-harmonized_rural['FracNoCar']
    most = harmonized_rural.iloc[np.where(frac > 
        np.nanpercentile(frac, ptile_upper))]
    least = harmonized_rural.iloc[np.where(frac < 
        np.nanpercentile(frac, ptile_lower))]
    ax.plot(most['PRENO2'].mean(), i-os, 'o', color=color_white, 
        zorder=12)
    ax.plot(least['PRENO2'].mean(), i-os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([most['PRENO2'].mean(),least['PRENO2'].mean()]), 
        np.min([most['PRENO2'].mean(),least['PRENO2'].mean()])+
        np.abs(np.diff([most['PRENO2'].mean(), least['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax.plot(most['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax.plot(least['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()]), 
        np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()])+
        np.abs(np.diff([most['POSTNO2'].mean(), least['POSTNO2'].mean()]))[0]), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7  
    # For urban tracts
    frac = 1-harmonized_urban['FracNoCar']
    most = harmonized_urban.iloc[np.where(frac > 
        np.nanpercentile(frac, ptile_upper))]
    least = harmonized_urban.iloc[np.where(frac < 
        np.nanpercentile(frac, ptile_lower))]
    ax.plot(most['PRENO2'].mean(), i-os, 'o', color=color_white, 
        zorder=12)
    ax.plot(least['PRENO2'].mean(), i-os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([most['PRENO2'].mean(),least['PRENO2'].mean()]), 
        np.min([most['PRENO2'].mean(),least['PRENO2'].mean()])+
        np.abs(np.diff([most['PRENO2'].mean(), least['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax.plot(most['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax.plot(least['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax.plot((np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()]), 
        np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()])+
        np.abs(np.diff([most['POSTNO2'].mean(), least['POSTNO2'].mean()]))[0]), 
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
        harmonized_city = subset_harmonized_bycountyfips(harmonized, city)
        frac = 1-harmonized_city['FracNoCar']
        most = harmonized_city.iloc[np.where(frac > 
            np.nanpercentile(frac, ptile_upper))]
        least = harmonized_city.iloc[np.where(frac < 
            np.nanpercentile(frac, ptile_lower))]
        ax.plot(most['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
        ax.plot(least['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
        ax.plot((np.min([most['PRENO2'].mean(),least['PRENO2'].mean()]), 
            np.min([most['PRENO2'].mean(),least['PRENO2'].mean()])+
            np.abs(np.diff([most['PRENO2'].mean(), least['PRENO2'].mean()]))[0]), 
            [i-os,i-os], color='k', ls='-', zorder=10)    
        ax.plot(most['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
        ax.plot(least['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
        ax.plot((np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()]), 
            np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()])+
            np.abs(np.diff([most['POSTNO2'].mean(), least['POSTNO2'].mean()]))[0]), 
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
    for side in ['right', 'left', 'top', 'bottom']:
        ax.spines[side].set_visible(False)
    ax.grid(axis='x', zorder=0, color='darkgrey')
    custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
        Line2D([0], [0], marker='o', color=color_non, lw=0),  
        Line2D([0], [0], color='k', lw=1.0),
        Line2D([0], [0], color='k', ls='--', lw=1)]
    ax.legend(custom_lines, ['Highest vehicle ownership', 
        'Lowest vehicle ownership', 'Baseline', 'Lockdown'], 
        bbox_to_anchor=(0.4, -0.07), loc=8, ncol=4, frameon=False)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.20, right=0.96, bottom=0.05, top=0.9)
    plt.savefig(DIR_FIGS+'lollipop_onvehicleownership.png', dpi=500)
    plt.show()        
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

def hist_demographics_urbanmissing(harmonized_urban): 
    """Distribution of demographics for all urban tracts and urban tracts 
    that are missing NO2 data. On account of the unequal sample size of 
    total urban tracts versus those without data (31483 versus 4918 tracts, 
    respectively), distributions have been normalized to display relative 
    occurrences. Horizontal dashed lines denote mean values for each 
    distribution. 

    Parameters
    ----------
    harmonized_urban : pandas.core.frame.DataFrame
        Harmonized tract-level TROPOMI NO2 and census data for urban tracts in 
        state(s) of interest

    Returns
    -------        
    None    
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    # Overall distribution of variables for urban tracts
    income = harmonized_urban['AJZAE001']
    race = (harmonized_urban['AJWNE002']/harmonized_urban['AJWBE001'])
    education = (harmonized_urban.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        harmonized_urban['AJYPE001'])
    ethnicity = (harmonized_urban['AJWWE002']/harmonized_urban['AJWWE001'])
    nocar = harmonized_urban['FracNoCar']
    # Subset DataFrame for urban tracts lacking co-located TROPOMI grid cell
    nogc_harmonized = harmonized_urban.loc[harmonized_urban['PRENO2'].isna()]
    nogc_income = nogc_harmonized['AJZAE001']
    nogc_race = (nogc_harmonized['AJWNE002']/nogc_harmonized['AJWBE001'])
    nogc_education = (nogc_harmonized.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        nogc_harmonized['AJYPE001'])
    nogc_ethnicity = (nogc_harmonized['AJWWE002']/nogc_harmonized['AJWWE001'])
    nogc_nocar = nogc_harmonized['FracNoCar']
    # Colors, number of bins for histograms
    color_nogc = '#0095A8'
    color_urban = '#FF7043'
    nb = 50
    
    
    # CHECK K-S test
    from scipy.stats import ks_2samp
    ks_2samp(x, y)
    # Initialize figure, axes
    fig = plt.figure(figsize=(9,4))
    ax1 = plt.subplot2grid((2,3),(0,0))
    ax2 = plt.subplot2grid((2,3),(0,1))
    ax3 = plt.subplot2grid((2,3),(0,2))
    ax4 = plt.subplot2grid((2,3),(1,0))
    ax5 = plt.subplot2grid((2,3),(1,1))
    # Income
    n1a, bins1a, rect1a = ax1.hist(nogc_income, nb, density=True, 
        color=color_nogc, orientation='horizontal')
    n1b, bins1b, rect1b = ax1.hist(income, nb, density=True, color=color_urban, 
        orientation='horizontal')
    ax1.hlines(y=income.mean(), xmin=0, xmax=ax1.get_xlim()[1], ls='--',
        lw=1, color='k')
    ax1.hlines(y=nogc_income.mean(), xmin=ax1.get_xlim()[1]*-1, xmax=0, 
        ls='--', lw=1, color='k')
    # Race
    n2a, bins2a, rect2a = ax2.hist(nogc_race, nb, density=True, 
        color=color_nogc, orientation='horizontal')
    n2b, bins2b, rect2b = ax2.hist(race, nb, density=True, color=color_urban, 
        orientation='horizontal')
    ax2.hlines(y=race.mean(), xmin=0, xmax=ax2.get_xlim()[1], ls='--',
        lw=1, color='k')
    ax2.hlines(y=nogc_race.mean(), xmin=ax2.get_xlim()[1]*-1, xmax=0, ls='--',
        lw=1, color='k')
    # Education 
    n3a, bins3a, rect3a = ax3.hist(nogc_education, nb, density=True, 
        color=color_nogc, orientation='horizontal')
    n3b, bins3b, rect3b = ax3.hist(education, nb, density=True, 
        color=color_urban, orientation='horizontal')
    ax3.hlines(y=education.mean(), xmin=0, xmax=ax3.get_xlim()[1], 
        ls='--', lw=1, color='k')
    ax3.hlines(y=nogc_education.mean(), xmin=ax3.get_xlim()[1]*-1, xmax=0, 
        lw=1, ls='--', color='k')
    # Ethnicity
    n4a, bins4a, rect4a = ax4.hist(nogc_ethnicity, nb, density=True, 
        color=color_nogc, orientation='horizontal')
    n4b, bins4b, rect4b = ax4.hist(ethnicity, nb, density=True, 
        color=color_urban, orientation='horizontal')
    ax4.hlines(y=ethnicity.mean(), xmin=0, xmax=ax4.get_xlim()[1], ls='--',
        lw=1, color='k')
    ax4.hlines(y=nogc_ethnicity.mean(), xmin=ax4.get_xlim()[1]*-1, xmax=0, 
        ls='--', lw=1, color='k')
    # Vehicle ownership 
    n5a, bins5a, rect5a = ax5.hist(nogc_nocar, nb, density=True, 
        color=color_nogc, orientation='horizontal')
    n5b, bins5b, rect5b = ax5.hist(nocar, nb, density=True, color=color_urban, 
        orientation='horizontal')
    ax5.hlines(y=nocar.mean(), xmin=0, xmax=ax5.get_xlim()[1], ls='--',
        lw=1, color='k')
    ax5.hlines(y=nogc_nocar.mean(), xmin=ax5.get_xlim()[1]*-1, xmax=0, ls='--',
        lw=1, color='k')
    # Reflect missing data distribution (not total urban distribution) over
    # y-axis
    for rect in [rect1a, rect2a, rect3a, rect4a, rect5a]:
        # iterate through rectangles, change the height of each
        for r in rect:
            r.set_width(r.get_width()*-1)
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xlim([ax.get_xlim()[1]*-1,ax.get_xlim()[1]])
    # Aesthetics
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        for pos in ['right','top','bottom']:
            ax.spines[pos].set_visible(False)
            ax.set_xticks([])
    # Axis limits, ticks    
    ax1.set_ylim([0, 200000])
    for ax in [ax2, ax3, ax4, ax5]:
        ax.set_ylim([0,1])
        ax.set_yticks([0,0.25,0.5,0.75,1])
        ax.set_yticklabels(['0','','50','','100'])
    # Label
    ax1.set_ylabel('Income [$]')
    ax2.set_ylabel('White [%]')
    ax3.set_ylabel('Some college\nor greater [%]')
    ax4.set_ylabel('Non-Latinx [%]')
    ax5.set_ylabel('Households without\na vehicle [%]')
    ax1.set_title('(a)', loc='left', fontsize=12)
    ax2.set_title('(b)', loc='left', fontsize=12)
    ax3.set_title('(c)', loc='left', fontsize=12)
    ax4.set_title('(d)', loc='left', fontsize=12)
    ax5.set_title('(e)', loc='left', fontsize=12)
    # Custom legend 
    patch1 = mpatches.Patch(color=color_nogc, label='Missing NO$_{2}$ data')
    patch2 = mpatches.Patch(color=color_urban, label='All urban')
    all_handles = (patch1, patch2)
    ax5.legend(handles=all_handles, ncol=1, bbox_to_anchor=(2.3,0.8),
        frameon=False, fontsize=12)
    plt.subplots_adjust(wspace=0.4, hspace=0.3, right=0.95)
    plt.savefig(DIR_FIGS+'hist_demographics_urbanmissing.png', dpi=600)
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

# # Calculate percentage of tracts without co-located TROPOMI retrievals 
# print('%.1f of all tracts have NO2 retrievals'%(len(np.where(np.isnan(
#     harmonized['PRENO2'])==False)[0])/len(harmonized)*100.))
# print('%.1f of urban tracts have NO2 retrievals'%(len(np.where(np.isnan(
#     harmonized_urban['PRENO2'])==False)[0])/len(harmonized_urban)*100.))
# print('%.1f of rural tracts have NO2 retrievals'%(len(np.where(np.isnan(
#     harmonized_rural['PRENO2'])==False)[0])/len(harmonized_rural)*100.))
# Visualizations
# demographic_summarytable(demography, 'alltracts')
# demographic_summarytable(demography_rural, 'ruraltracts')
# demographic_summarytable(demography_urban, 'urbantracts')
# map_no2historic_no2gains(mostno2, leastno2, increaseno2, decreaseno2, FIPS)
# NO2_gridded_tractavg(lng_dg, lat_dg, no2_pre_dg, no2_post_dg, FIPS, harmonized)

# bar_no2historic_no2gains(demography, 'alltracts')
# bar_no2historic_no2gains(demography_urban, 'urbantracts')
# bar_no2historic_no2gains(demography_rural, 'ruraltracts')

# lollipop(harmonized, harmonized_urban, harmonized_rural)
# (demography_city, mostno2_city, leastno2_city, increaseno2_city, 
#     decreaseno2_city) = harmonized_demographics(harmonized_city, 
#     ptile_upper, ptile_lower)
# demographic_summarytable(demography_city, 'newyork')

# bar_gains(harmonized_rural)
# hist_demographics_urbanmissing(harmonized_urban)
















    

# def merge_harmonized_error(harmonized):
#     """function opens census data on vehicle ownership and derived road density
#     and merges it with harmonized TROPOMI-census data
    
#     Parameters
#     ----------
#     harmonized : pandas.core.frame.DataFrame
#         Harmonized tract-level TROPOMI NO2 and census data for state(s) of 
#         interest
        
#     Returns
#     -------
#     harmonized_error : pandas.core.frame.DataFrame
#         Harmonized tract-level TROPOMI NO2 and census data for state(s) of 
#         interest with appended margin of error columns for race, income, and
#         educational attainment
#     """
#     import pandas as pd
#     # Read in original NHGIS data with Margins of error
#     error = pd.read_csv(DIR_HARM+'nhgis0003_ds239_20185_2018_tract.csv', 
#         delimiter=',', header=0, engine='python')
#     # Replace GISJOIN identified with GEOID for easy look-up
#     gisjoin_to_geoid = [x[1:3]+x[4:7]+x[8:] for x in error['GISJOIN']]
#     error['GEOID'] = gisjoin_to_geoid
#     # Make GEOID a string and index row 
#     error = error.set_index('GEOID')
#     error.index = error.index.map(str)
#     # Make relevant columns floats
#     col_relevant = [
#         'AJWBM001', # Total population
#         'AJWNM002', # White alone
#         'AJWNM003', #Black or African American alone
#         'AJWNM004', # American Indian and Alaska Native alone
#         'AJWNM005', # Asian alone
#         'AJWNM006', # Native Hawaiian and Other Pacific Islander alone
#         'AJWNM007', # Some other race alone
#         'AJWNM008', # Two or more races
#         'AJWNM009', # Two or more races: Two races including Some other race
#         'AJWNM010', # Two or more races: Two races excluding Some other race, 
#         # and three or more races
#         'AJYPM001', # Total
#         'AJYPM002', # No schooling completed
#         'AJYPM003', # Nursery school
#         'AJYPM004', # Kindergarten
#         'AJYPM005', # 1st grade
#         'AJYPM006', # 2nd grade
#         'AJYPM007', # 3rd grade
#         'AJYPM008', # 4th grade
#         'AJYPM009', # 5th grade
#         'AJYPM010', # 6th grade
#         'AJYPM011', # 7th grade
#         'AJYPM012', # 8th grade
#         'AJYPM013', # 9th grade
#         'AJYPM014', # 10th grade
#         'AJYPM015', # 11th grade
#         'AJYPM016', # 12th grade, no diploma
#         'AJYPM017', # Regular high school diploma
#         'AJYPM018', # GED or alternative credential
#         'AJYPM019', # Some college, less than 1 year
#         'AJYPM020', # Some college, 1 or more years, no degree
#         'AJYPM021', # Associate's degree
#         'AJYPM022', # Bachelor's degree
#         'AJYPM023', # Master's degree
#         'AJYPM024', # Professional school degree
#         'AJYPM025', # Doctorate degree
#         'AJZAM001' # Median household income in the past 12 months 
#         # (in 2018 inflation-adjusted dollars)
#         ]
#     error = error[col_relevant]
#     for col in col_relevant:
#         error[col] = error.loc[:,col].astype(float)
#     # Merge with harmonized census data
#     harmonized_error = harmonized.merge(error, left_index=True, 
#         right_index=True)
#     return harmonized_error

# def calculate_moe(data, moe):
#     """
#     """
#     uncertainty = []
#     niter = 100
#     # Generate array of randomly chosen +/-1; the MoE columns will be
#     # multiplied by these random numbers such that they can be added or 
#     # subtracted at random
#     for x in np.arange(0, niter, 1):
#         randaddsub = np.random.choice(a=[-1, 1], size=(len(data)))
#         randaddsub = pd.Series(data + (moe*randaddsub), name=str(x))    
#         uncertainty.append(randaddsub)
#     uncertainty = pd.concat(uncertainty, axis=1)
#     return uncertainty 

# # https://www.psc.isr.umich.edu/dis/acs/handouts/Compass_Appendix.pdf

# X_num = harmonized['AJWNE002']
# X_den = harmonized['AJWBE001']
# moe_num = harmonized['AJWNM002']
# moe_den = harmonized['AJWBM001']
# phat = (harmonized['AJWNE002']/harmonized['AJWBE001'])

# # phat = (4634/31713.)
# # moe_num = 989.
# # moe_den = 601.
# # X_den = 31713.

# moe_p = np.sqrt(moe_num**2 - (phat**2 * moe_den**2))/X_den



# moe_p = moe_p.mask(np.isinf(moe_p))



# frac_white = harmonized['AJWNE002']/harmonized['AJWBE001']
# frac_white_high = pd.DataFrame(frac_white+moe_p, columns = ['Frac'])
# frac_white_low = pd.DataFrame(frac_white-moe_p, columns = ['Frac'])
# frac_white_low = frac_white_low.mask(np.isinf(frac_white_low))
# frac_white_high = frac_white_high.mask(np.isinf(frac_white_high))


# frac_white_low = frac_white_low.merge(harmonized[['PRENO2','POSTNO2']], 
#     left_index=True, right_index=True)
# frac_white_high = frac_white_high.merge(harmonized[['PRENO2','POSTNO2']], 
#     left_index=True, right_index=True)

# mostwhite_low = frac_white_low.iloc[np.where(frac_white_low['Frac'] > 
#     np.nanpercentile(frac_white_low['Frac'], ptile_upper))]
# mostwhite_high = frac_white_high.iloc[np.where(frac_white_high['Frac'] > 
#     np.nanpercentile(frac_white_high['Frac'], ptile_upper))]

# frac_white = (harmonized['AJWNE002']/harmonized['AJWBE001'])
# mostwhite = harmonized.iloc[np.where(frac_white > 
#     np.nanpercentile(frac_white, ptile_upper))]

# leastwhite = harmonized.iloc[np.where(frac_white < 
#     np.nanpercentile(frac_white, ptile_lower))]


# harmonized = merge_harmonized_error(harmonized)


# AJWNE002_uncert = calculate_moe(harmonized['AJWNE002'], harmonized['AJWNM002'])
# AJWBE001_uncert = calculate_moe(harmonized['AJWBE001'], harmonized['AJWBM001'])


# field_low = harmonized['AJWNE002']-harmonized['AJWNM002']
# field_high = harmonized['AJWNE002']+harmonized['AJWNM002']

# frac_white_low = frac_white_low.mask(np.isinf(frac_white_low))
# frac_white_high = frac_white_high.mask(np.isinf(frac_white_high))

# frac_white_high = pd.DataFrame(frac_white_high, columns=['Frac'])
# frac_white_low = pd.DataFrame(frac_white_low, columns=['Frac'])
# # Determine demographics in tracts
# demography, mostno2, leastno2, increaseno2, decreaseno2 = \
#     harmonized_demographics(harmonized, ptile_upper, ptile_lower)
# (demography_urban, mostno2_urban, leastno2_urban, increaseno2_urban, 
#     decreaseno2_urban) = harmonized_demographics(harmonized_urban, ptile_upper, 
#     ptile_lower)
# (demography_rural, mostno2_rural, leastno2_rural, increaseno2_rural, 
#     decreaseno2_rural) = harmonized_demographics(harmonized_rural, ptile_upper, 
#     ptile_lower)





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
# shp = shapereader.Reader(DIR_GEO+'tigerline/roads/tl_2019_06_prisecroads/'+
#     'tl_2019_06_prisecroads')
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
#     ax.set_extent([-119, -117.4, 33.3, 34.9], proj)
#     ax.add_feature(county_out, facecolor='w', edgecolor='w', zorder=11)    
#     ax.add_feature(county_in, facecolor='None', edgecolor='r', linewidth=0.0,
#         zorder=12)
#     ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m'),
#                     linewidth=0.,facecolor='w', edgecolor='w', zorder=13)
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
# ax2.set_title('(b) White [%]', loc='left')
# ax3.set_title('(c) NO$_{2}$ [molec cm$^{-2}$]', loc='left')
# ax4.set_title('(d) $\Delta\:$NO$_{2}$ [molec cm$^{-2}$]', loc='left')
# ax5.set_title('(e)',loc='left')
# plt.savefig('/Users/ghkerr/Desktop/losangelescasestudy.png', dpi=600)
# plt.show()




"""NEW YORK CITY CASE STUDY"""
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
# newyork = ['36047','36081','36061','36005','36085','36119','34003',
#     '34017','34031','36079','36087','36103','36059','34023','34025',
#     '34029','34035','34013','34039','34027','34037'	,'34019','42103']
# # Find harmonzied data in city
# harmonized_city = subset_harmonized_bycountyfips(harmonized, 
#     newyork)
# # Find CEMS data in city
# cems_nyc = open_cems(['36','42','34'], counties=['Kings County', 
#     'Queens County', 'New York County', 'Bronx County', 'Richmond County',
#     'Westchester County', 'Bergen County', 'Hudson County', 'Passaic County',
#     'Putnam County', 'Rockland County', 'Suffolk County', 'Nassau County',
#     'Middlesex County', 'Monmouth County', 'Ocean County', 'Somerset County',
#     'Essex County', 'Union County', 'Morris County', 'Sussex County',
#     'Hunterdon County', 'Pike County'])
# cems_loc = cems_nyc.groupby([' Facility Latitude',' Facility Longitude']
#     ).size().reset_index(name='Freq')
# cems_nyc = cems_nyc.groupby([' Date']).sum()
# # # Find VMT data in city
# streetlights = load_streetlights(countytimeavg=False)
# streetlights_nyc = streetlights.loc[streetlights['GEOID'].isin(newyork)]
# streetlights_nyc_mean = streetlights_nyc.groupby(['ref_dt']).mean()
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
# # Open shapefile for each state in NYC MSA
# records, tracts = [], []
# for state in ['34','36','42']:
#     shp_state = shapereader.Reader(DIR_GEO+'tigerline/'+
#         'tl_2019_%s_tract/tl_2019_%s_tract'%(state,state))
#     records += list(shp_state.records())
#     tracts += list(shp_state.geometries())
# # Find records and tracts in city 
# geoids_records = [x.attributes['GEOID'] for x in records]
# geoids_records = np.where(np.in1d(np.array(geoids_records), 
#     harmonized_city.index)==True)[0]
# # Slice records and tracts for only entries in city
# records = list(np.array(records)[geoids_records])
# tracts = list(np.array(tracts)[geoids_records])
# # Loop through shapefiles in state
# for geoid in harmonized_city.index:
#     # print(geoid)
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
#     transform=proj, markersize=1)
# # Add primary and secondary roads
# roads_records, roads = [], []
# for state in ['34','36','42']:
#     shp = shapereader.Reader(DIR_GEO+
#         'tigerline/roads/tl_2019_%s_prisecroads/'%(state)+
#          'tl_2019_%s_prisecroads'%(state))
#     roads_records += list(shp.records())
#     roads += list(shp.geometries())
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
# county_out = np.where(np.in1d(np.array(county_in), np.array(newyork))==False)
# county_in = np.where(np.in1d(np.array(county_in), np.array(newyork))==True)
# county_in = cfeature.ShapelyFeature(np.array(counties)[county_in], proj)
# county_out = cfeature.ShapelyFeature(np.array(counties)[county_out], proj)
# for ax in [ax1, ax2, ax3, ax4]:
#     ax.set_extent([-75.6768, -71.7163, 39.48, 41.5095], proj)
#     ax.add_feature(county_out, facecolor='w', edgecolor='w', zorder=11)    
#     ax.add_feature(county_in, facecolor='None', edgecolor='r', linewidth=0.0,
#         zorder=12)
#     ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m'),
#                     linewidth=0.,facecolor='w', edgecolor='w', zorder=13)
#     ax.background_patch.set_visible(False)
#     ax.outline_patch.set_visible(False)
# ax5.plot(cems_nyc['2020-03-01':'2020-05-12'][' NOx (tons)'].values, '-k', 
#     zorder=5)
# ax5t = ax5.twinx()
# ax5t.plot(streetlights_nyc_mean['2020-03-01':'2020-05-12']['county_vmt'].values, 
#     '-r', zorder=6)
# ax5t.set_xlim([0, streetlights_nyc_mean.index.shape[0]-1])
# ax5t.set_xticks([0,14,31,45,61,72])
# ax5t.set_xticklabels(['1 March', '15 March', '1 April', '15 April', '1 May', '12 May'])
# # Stay at home order
# ax5.vlines(21, ymin=ax5.get_ylim()[0], ymax=ax5.get_ylim()[1], 
#     zorder=0, color='darkgrey')
# ax5t.text(22, 1.4e7, 'Stay-at-\nhome order', fontsize=12, 
#     color='darkgrey', rotation='vertical')
# # Label VMT
# ax5.text(0, 17, 'Traffic', fontsize=12, color='r')
# ax5t.text(0, 0.25e7, 'Industry', fontsize=12, color='k')
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
# ax2.set_title('(b) White [%]', loc='left')
# ax3.set_title('(c) NO$_{2}$ [molec cm$^{-2}$]', loc='left')
# ax4.set_title('(d) $\Delta\:$NO$_{2}$ [molec cm$^{-2}$]', loc='left')
# ax5.set_title('(e)',loc='left')
# plt.savefig('/Users/ghkerr/Desktop/newyorkcasestudy.png', dpi=600)
# plt.show()



# Add U.S. counties 
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




"""DETROIT"""
# from scipy import stats
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import netCDF4 as nc
# import numpy as np
# import matplotlib as mpl
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.io import shapereader
# detroit = ['26163']
# # Find harmonzied data in city
# harmonized_city = subset_harmonized_bycountyfips(harmonized_urban, 
#     detroit)
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
# # ax5 = plt.subplot2grid((2,3),(1,1), colspan=2)
# proj = ccrs.PlateCarree(central_longitude=0.0)
# # Create discrete colormaps
# cmapincome = plt.get_cmap('YlGnBu', 8)
# normincome = matplotlib.colors.Normalize(vmin=20000, vmax=100000)
# cmapwhite = plt.get_cmap('YlGnBu', 8)
# normwhite = matplotlib.colors.Normalize(vmin=0, vmax=100)
# cmapbase = plt.get_cmap('YlGnBu', 7)
# normbase = matplotlib.colors.Normalize(vmin=2e15, vmax=6e15)
# cmaplock = plt.get_cmap('Blues_r', 8)
# normlock = matplotlib.colors.Normalize(vmin=-2e15, vmax=0e15)
# # Open shapefile for each state in NYC MSA
# records, tracts = [], []
# for state in ['26']:
#     shp_state = shapereader.Reader(DIR_GEO+'tigerline/'+
#         'tl_2019_%s_tract/tl_2019_%s_tract'%(state,state))
#     records += list(shp_state.records())
#     tracts += list(shp_state.geometries())
# # Find records and tracts in city 
# geoids_records = [x.attributes['GEOID'] for x in records]
# geoids_records = np.where(np.in1d(np.array(geoids_records), 
#     harmonized_city.index)==True)[0]
# # Slice records and tracts for only entries in city
# records = list(np.array(records)[geoids_records])
# tracts = list(np.array(tracts)[geoids_records])
# # Loop through shapefiles in state
# for geoid in harmonized_city.index:
#     # print(geoid)
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
# # # Add CEMS data        
# # ax4.plot(cems_loc[' Facility Longitude'].values, 
# #     cems_loc[' Facility Latitude'].values, 'ko', zorder=25, 
# #     transform=proj, markersize=1)
# # Add primary and secondary roads
# roads_records, roads = [], []
# for state in ['26']:
#     shp = shapereader.Reader(DIR_GEO+
#         'tigerline/roads/tl_2019_%s_prisecroads/'%(state)+
#           'tl_2019_%s_prisecroads'%(state))
#     roads_records += list(shp.records())
#     roads += list(shp.geometries())
# roads = cfeature.ShapelyFeature(roads, proj)
# ax4.add_feature(roads, facecolor='None', edgecolor='r', zorder=11, lw=0.5)
# # # Add counties 
# # reader = shapereader.Reader(DIR_GEO+'counties/tl_2019_us_county/'+
# #     'tl_2019_us_county')
# # counties = list(reader.geometries())
# # records = list(reader.records())
# # # Select only counties inside/outside city for masking
# # county_in = [x.attributes['GEOID'] for x in records]
# # county_out = np.where(np.in1d(np.array(county_in), np.array(newyork))==False)
# # county_in = np.where(np.in1d(np.array(county_in), np.array(newyork))==True)
# # county_in = cfeature.ShapelyFeature(np.array(counties)[county_in], proj)
# # county_out = cfeature.ShapelyFeature(np.array(counties)[county_out], proj)
# for ax in [ax1, ax2, ax3, ax4]:#, ax2, ax3, ax4]:
#     ax.set_extent([-83.60, -82.86, 42.025, 42.47], proj)
# #     ax.add_feature(county_out, facecolor='w', edgecolor='w', zorder=11)    
# #     ax.add_feature(county_in, facecolor='None', edgecolor='r', linewidth=0.0,
# #         zorder=12)
#     ax.add_feature(cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m'),
#                     linewidth=0.,facecolor='w', edgecolor='w', zorder=13)
#     ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '10m'),
#                     linewidth=0.,facecolor='w', edgecolor='w', zorder=13)
#     ax.background_patch.set_visible(False)
#     ax.outline_patch.set_visible(False)
# # ax5.plot(cems_nyc['2020-03-01':'2020-05-12'][' NOx (tons)'].values, '-k', 
# #     zorder=5)
# # ax5t = ax5.twinx()
# # ax5t.plot(streetlights_nyc_mean['2020-03-01':'2020-05-12']['county_vmt'].values, 
# #     '-r', zorder=6)
# # ax5t.set_xlim([0, streetlights_nyc_mean.index.shape[0]-1])
# # ax5t.set_xticks([0,14,31,45,61,72])
# # ax5t.set_xticklabels(['1 March', '15 March', '1 April', '15 April', '1 May', '12 May'])
# # # Stay at home order
# # ax5.vlines(21, ymin=ax5.get_ylim()[0], ymax=ax5.get_ylim()[1], 
# #     zorder=0, color='darkgrey')
# # ax5t.text(22, 1.4e7, 'Stay-at-\nhome order', fontsize=12, 
# #     color='darkgrey', rotation='vertical')
# # # Label VMT
# # ax5.text(0, 17, 'Traffic', fontsize=12, color='r')
# # ax5t.text(0, 0.25e7, 'Industry', fontsize=12, color='k')
# # # Aesthetics
# # ax5.tick_params(axis='y', which='both', right=False, left=False, 
# #     labelleft=False)
# # ax5t.tick_params(axis='y', which='both', right=False, left=False, 
# #     labelright=False)
# # for pos in ['right','top','left']:
# #     ax5.spines[pos].set_visible(False)
# #     ax5t.spines[pos].set_visible(False)
# # ax5t.yaxis.offsetText.set_visible(False)
# # plt.subplots_adjust(left=0.05, top=0.93, hspace=0.6)
# # # Add colorbars 
# # # (a)
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
# ax2.set_title('(b) White [%]', loc='left')
# ax3.set_title('(c) NO$_{2}$ [molec cm$^{-2}$]', loc='left')
# ax4.set_title('(d) $\Delta\:$NO$_{2}$ [molec cm$^{-2}$]', loc='left')
# # ax5.set_title('(e)',loc='left')
# plt.savefig('/Users/ghkerr/Desktop/detroitcasestudy.png', dpi=600)
# # plt.show()



""" FIGURE 1 """
# import netCDF4 as nc
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from decimal import Decimal
# import matplotlib as mpl
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import cartopy.io.shapereader as shpreader
# from cartopy.io import shapereader
# from shapely.geometry import Polygon
# import cartopy.io.img_tiles as cimgt
# import cartopy.crs as ccrs
# def rect_from_bound(xmin, xmax, ymin, ymax):
#     """Returns list of (x,y)'s for a rectangle"""
#     xs = [xmax, xmin, xmin, xmax, xmax]
#     ys = [ymax, ymax, ymin, ymin, ymax]
#     return [(x, y) for x, y in zip(xs, ys)]
# proj = ccrs.PlateCarree(central_longitude=0.0)
# fig = plt.figure(figsize=(9,6))
# # Maps
# ax1 = plt.subplot2grid((5,2),(0,0), projection=ccrs.PlateCarree(
#     central_longitude=0.), rowspan=2)
# ax2 = plt.subplot2grid((5,2),(0,1), projection=ccrs.PlateCarree(
#     central_longitude=0.), rowspan=2)
# # Bar charts
# ax3 = plt.subplot2grid((5,2),(2,0))
# ax4 = plt.subplot2grid((5,2),(2,1))
# ax5 = plt.subplot2grid((5,2),(3,0))
# ax6 = plt.subplot2grid((5,2),(3,1))
# ax7 = plt.subplot2grid((5,2),(4,0))
# ax8 = plt.subplot2grid((5,2),(4,1))
# # Create discrete colormaps
# cmapbase = plt.get_cmap('YlGnBu', 12)
# normbase = matplotlib.colors.Normalize(vmin=0e15, vmax=6e15)
# cmaplock = plt.get_cmap('coolwarm', 12)
# normlock = matplotlib.colors.Normalize(vmin=-2e15, vmax=2e15)
# for FIPS_i in FIPS: 
#     print(FIPS_i)
#     # Tigerline shapefile for state
#     shp = shapereader.Reader(DIR_GEO+
#         'tigerline/tl_2019_%s_tract/tl_2019_%s_tract'%(FIPS_i, FIPS_i))
#     records = shp.records()
#     tracts = shp.geometries()
#     for record, tract in zip(records, tracts):
#         # Find GEOID of tract
#         gi = record.attributes['GEOID']
#         # Look up harmonized NO2-census data for tract
#         harmonized_tract = harmonized.loc[harmonized.index.isin([gi])]
#         baseline = harmonized_tract.PRENO2.values[0]
#         lockdown = harmonized_tract.NO2_ABS.values[0]
#         if np.isnan(baseline)==True:
#             ax1.add_geometries([tract], proj, facecolor='darkgrey', 
#                 edgecolor='darkgrey', alpha=1., linewidth=0.1, rasterized=True)
#             ax2.add_geometries([tract], proj, facecolor='darkgrey', 
#                 edgecolor='darkgrey', alpha=1., linewidth=0.1, rasterized=True)
#         else:
#             ax1.add_geometries([tract], proj, facecolor=cmapbase(
#                 normbase(baseline)), edgecolor=cmapbase(
#                 normbase(baseline)), alpha=1., linewidth=0.1, rasterized=True)
#             ax2.add_geometries([tract], proj, facecolor=cmaplock(
#                 normlock(lockdown)), edgecolor=cmaplock(
#                 normlock(lockdown)), alpha=1., linewidth=0.1, rasterized=True)
# shpfilename = shapereader.natural_earth('10m', 'cultural', 
#     'admin_0_countries')
# reader = shpreader.Reader(shpfilename)
# countries = reader.records()   
# # Get location of U.S.
# usa = [x.attributes['ADM0_A3'] for x in countries]
# usa = np.where(np.array(usa)=='USA')[0][0]
# usa = list(reader.geometries())[usa]   
# # Load lakes 
# lakes = shapereader.natural_earth('10m', 'physical', 'lakes')
# lakes_reader = shpreader.Reader(lakes)
# lakes = lakes_reader.records()   
# lake_names = [x.attributes['name'] for x in lakes]
# great_lakes = np.where((np.array(lake_names)=='Lake Superior') |
#     (np.array(lake_names)=='Lake Michigan') | 
#     (np.array(lake_names)=='Lake Huron') |
#     (np.array(lake_names)=='Lake Erie') |
#     (np.array(lake_names)=='Lake Ontario'))[0]
# great_lakes = np.array(list(lakes_reader.geometries()), 
#     dtype=object)[great_lakes]
# # Projection
# st_proj = ccrs.Mercator(central_longitude=0.0)#stamen_terrain.crs  #projection used by Stamen images
# pad1 = 1.  #padding, degrees unit
# exts = [usa[0].bounds[0] - pad1, usa[0].bounds[2] + pad1, 
#         usa[0].bounds[1] - pad1, usa[0].bounds[3] + pad1]
# # Make a mask polygon by polygon's difference operation; base polygon 
# # is a rectangle, another polygon is simplified switzerland
# msk = Polygon(rect_from_bound(*exts)).difference(usa[0].simplify(0.01))
# # Project geometry to the projection used by stamen
# msk_stm  = st_proj.project_geometry(msk, proj)
# # Add borders, set map extent, etc. 
# for ax in [ax1, ax2]:
#     ax.set_extent([-125,-66.5, 24.5, 49.48], proj)       
#     ax.add_geometries(usa, crs=proj, lw=0.25, facecolor='None', 
#         edgecolor='k', zorder=15)
#     # Plot the mask
#     ax.add_geometries(msk_stm, st_proj, zorder=12, facecolor='w', 
#         edgecolor='None', alpha=1.)             
#     # Add states
#     ax.add_feature(cfeature.NaturalEarthFeature('cultural', 
#         'admin_1_states_provinces_lines', '10m', edgecolor='k', lw=0.25, 
#         facecolor='none'))
#     # Add Great Lakes
#     ax.add_geometries(great_lakes, crs=ccrs.PlateCarree(), 
#         facecolor='w', lw=0.25, edgecolor='black', alpha=1., zorder=17)
#     ax.axis('off')
#     # ax.background_patch.set_visible(False)
#     # ax.outline_patch.set_visible(False)
# # # # # Bar charts for demographics
# harmonized = harmonized_urban
# colors = ['#0095A8','darkgrey','#FF7043']
# # Largest, smallest and median gains
# increaseno2 = harmonized.loc[harmonized['NO2_ABS']>
#     np.nanpercentile(harmonized['NO2_ABS'], ptile_upper)]
# # medianno2 = harmonized.loc[(harmonized['NO2_ABS']>
# #     np.nanpercentile(harmonized['NO2_ABS'], 45)) & (harmonized['NO2_ABS']<=
# #     np.nanpercentile(harmonized['NO2_ABS'], 55))]
# decreaseno2 = harmonized.loc[harmonized['NO2_ABS']<
#     np.nanpercentile(harmonized['NO2_ABS'], ptile_lower)]
# # # # # Change in NO2
# # Urban 
# ax3.barh([2,1,0], [decreaseno2['NO2_ABS'].mean(), harmonized['NO2_ABS'].mean(), 
#     increaseno2['NO2_ABS'].mean()], color=colors[0])
# print('Largest gains for NO2 = %.2E'%Decimal(decreaseno2['NO2_ABS'].mean()))
# ax3.text(decreaseno2['NO2_ABS'].mean()+2e13, 2, '-2.9', color='black', 
#     va='center')
# print('Mean gains for NO2 = %.2E'%Decimal(harmonized['NO2_ABS'].mean()))
# ax3.text(harmonized['NO2_ABS'].mean()+2e13, 1, '-0.85', color='black', 
#     va='center')
# print('Smallest gains for NO2 = %.2E'%Decimal(increaseno2['NO2_ABS'].mean()))
# ax3.text(increaseno2['NO2_ABS'].mean()+2e13, 0, '0.08', color='black', 
#     va='center')
# # # # # Income
# ax5.barh([2,1,0], [decreaseno2['AJZAE001'].mean(), 
#     harmonized['AJZAE001'].mean(), increaseno2['AJZAE001'].mean()], 
#     color=colors[0])
# ax5.text(60000, 2, ' %d'%(decreaseno2['AJZAE001'].mean()), color='black', 
#     va='center')
# ax5.text(60000, 1, ' %d'%(harmonized['AJZAE001'].mean()), color='black', 
#     va='center')
# ax5.text(60000, 0, ' %d'%(increaseno2['AJZAE001'].mean()), color='black', 
#     va='center')
    
# # # # # Racial background
# left, i = 0, 0
# labels = ['White', 'Black', 'Other']
# # Largest gains
# for data, color in zip([(decreaseno2['AJWNE002']/
#     decreaseno2['AJWBE001']).mean(),
#     (decreaseno2['AJWNE003']/decreaseno2['AJWBE001']).mean(),      
#     ((decreaseno2['AJWNE004']+decreaseno2['AJWNE005']+
#       decreaseno2['AJWNE006']+decreaseno2['AJWNE007']+
#     decreaseno2['AJWNE008'])/decreaseno2['AJWBE001']).mean()], colors):             
#     ax7.barh(2, data, color=color, left=left)
#     ax7.text(left+0.01, 2, '%d'%(np.round(data,2)*100), color='black', 
#         va='center')
#     left += data
#     i = i+1
# # Mean demographics
# left, i = 0, 0
# for data, color in zip([(harmonized['AJWNE002']/
#     harmonized['AJWBE001']).mean(),
#     (harmonized['AJWNE003']/harmonized['AJWBE001']).mean(),      
#     ((harmonized['AJWNE004']+harmonized['AJWNE005']+
#       harmonized['AJWNE006']+harmonized['AJWNE007']+
#     harmonized['AJWNE008'])/harmonized['AJWBE001']).mean()], colors):             
#     ax7.barh(1, data, color=color, left=left)
#     ax7.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='black', 
#         va='center')    
#     left += data
#     i = i+1    
# # Smallest gains
# left, i = 0, 0
# for data, color in zip([(increaseno2['AJWNE002']/
#     increaseno2['AJWBE001']).mean(),
#     (increaseno2['AJWNE003']/increaseno2['AJWBE001']).mean(),      
#     ((increaseno2['AJWNE004']+increaseno2['AJWNE005']+
#       increaseno2['AJWNE006']+increaseno2['AJWNE007']+
#     increaseno2['AJWNE008'])/increaseno2['AJWBE001']).mean()], colors):             
#     ax7.barh(0, data, color=color, left=left)
#     ax7.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='black', 
#         va='center')    
#     ax7.text(left+0.01, -1, labels[i], color=colors[i], va='center',
#         fontweight='bold')
#     left += data       
#     i = i+1    

# # # # # Ethnic background
# # Largest gains
# left, i = 0, 0
# labels = ['Hispanic', 'Non-Hispanic']
# for data, color in zip([(decreaseno2['AJWWE003']/
#     decreaseno2['AJWWE001']).mean(),
#     (decreaseno2['AJWWE002']/decreaseno2['AJWWE001']).mean()], colors):             
#     ax4.barh(2, data, color=color, left=left)
#     ax4.text(left+0.01, 2, '%d'%(np.round(data,2)*100), color='black', 
#         va='center')
#     left += data
#     i = i+1
# # Mean demographics
# left, i = 0, 0
# for data, color in zip([(harmonized['AJWWE003']/
#     harmonized['AJWWE001']).mean(),
#     (harmonized['AJWWE002']/harmonized['AJWWE001']).mean()], colors):             
#     ax4.barh(1, data, color=color, left=left)
#     ax4.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='black', 
#         va='center')
#     left += data
#     i = i+1
# # Median gains
# left, i = 0, 0
# for data, color in zip([(increaseno2['AJWWE003']/
#     increaseno2['AJWWE001']).mean(),
#     (increaseno2['AJWWE002']/increaseno2['AJWWE001']).mean()], colors): 
#     ax4.barh(0, data, color=color, left=left)
#     ax4.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='black', 
#         va='center')    
#     ax4.text(left+0.01, -1, labels[i], color=colors[i], va='center',
#         fontweight='bold')              
#     left += data
#     i = i+1
    
# # # # # Educational attainment
# # Largest gains
# left = 0
# i = 0
# for data, color in zip([
#     (decreaseno2.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
#     decreaseno2['AJYPE001']).mean(),
#     (decreaseno2.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
#     decreaseno2['AJYPE001']).mean(),
#     (decreaseno2.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
#     decreaseno2['AJYPE001']).mean()], colors):             
#     ax6.barh(2, data, color=color, left=left)
#     ax6.text(left+0.01, 2, '%d'%(np.round(data,2)*100), 
#         color='black', va='center') 
#     left += data
#     i = i+1    
# # Mean demographics
# left = 0
# i = 0
# for data, color in zip([
#     (harmonized.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
#     harmonized['AJYPE001']).mean(),
#     (harmonized.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
#     harmonized['AJYPE001']).mean(),
#     (harmonized.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
#     harmonized['AJYPE001']).mean()], colors):             
#     ax6.barh(1, data, color=color, left=left)
#     ax6.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='black', 
#         va='center') 
#     left += data
#     i = i+1        
# # Smallest gains
# left = 0
# i = 0
# labels = ['High school', 'College', 'Graduate']
# for data, color in zip([
#     (increaseno2.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
#     increaseno2['AJYPE001']).mean(),
#     (increaseno2.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
#     increaseno2['AJYPE001']).mean(),
#     (increaseno2.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
#     increaseno2['AJYPE001']).mean()], colors):             
#     ax6.barh(0, data, color=color, left=left)
#     ax6.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='black', 
#         va='center')
#     if i==2:
#         ax6.text(0.82, -1, labels[i], color=colors[i], va='center',
#             fontweight='bold', fontsize=10)         
#     else:   
#         ax6.text(left, -1, labels[i], color=colors[i], va='center',
#             fontweight='bold', fontsize=10) 
#     left += data
#     i = i+1  
      
# # # # # Vehicle ownership
# # Largest gains
# left = 0
# i = 0
# labels = ['None', 'One or more']
# for data, color in zip([decreaseno2['FracNoCar'].mean(),
#     (1-decreaseno2['FracNoCar'].mean())], colors):             
#     ax8.barh(2, data, color=color, left=left)
#     ax8.text(left+0.01, 2, '%d'%(np.round(data,2)*100),
#         color='black', va='center') 
#     left += data
#     i = i+1        
# # Mean demographics
# left = 0
# i = 0
# for data, color in zip([harmonized['FracNoCar'].mean(),
#     (1-harmonized['FracNoCar'].mean())], colors):                        
#     ax8.barh(1, data, color=color, left=left)
#     ax8.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='black', 
#         va='center') 
#     left += data
#     i = i+1          
# # Smallest gains
# left = 0    
# i = 0
# for data, color in zip([increaseno2['FracNoCar'].mean(),
#     (1-increaseno2['FracNoCar'].mean())], colors):             
#     ax8.barh(0, data, color=color, left=left)
#     ax8.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='black', 
#         va='center') 
#     ax8.text((left*1.5), -1, labels[i], color=colors[i], va='center',
#         fontweight='bold', fontsize=10)              
#     left += data       
#     i = i+1      
# # Aesthetics    
# ax3.set_xlim([-3e15, 1e14])
# ax3.set_xticks([])    
# ax5.set_xlim([60000,80000])
# ax5.set_xticks([])
# for ax in [ax4,ax6,ax7,ax8]:
#     ax.set_xlim([0,1.])
#     ax.set_xticks([])
#     ax.set_xticklabels([])
# for ax in [ax3,ax4,ax5,ax6,ax7,ax8]:
#     ax.set_ylim([-0.5,2.5])
#     ax.set_yticks([0,1,2])
#     ax.set_yticklabels([])
# for ax in [ax3,ax4,ax5,ax6,ax7,ax8]:
#     for pos in ['right','top','bottom']:
#         ax.spines[pos].set_visible(False)
#         ax.spines[pos].set_visible(False)    
# for ax in [ax3,ax5,ax7]:
#     ax.set_yticklabels(['Smallest gains', 'Average', 'Largest gains'], 
#         fontsize=10)
# # Axis titles
# ax1.set_title('(a) NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', loc='left', 
#     fontsize=10)
# ax2.set_title('(b) $\Delta\:$NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', 
#     loc='left', fontsize=10)
# ax3.set_title('(c) $\mathregular{\Delta}$ NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', 
#     loc='left', fontsize=10)
# ax5.set_title('(e) Household income [$]', loc='left', fontsize=10)
# ax7.set_title('(g) Racial background [%]', loc='left', fontsize=10)
# ax4.set_title('(d) Ethnic background [%]', loc='left', fontsize=10)
# ax6.set_title('(f) Educational attainment [%]',loc='left', fontsize=10)
# ax8.set_title('(h) Household vehicle ownership [%]',loc='left', fontsize=10)
# # Adjust subplots
# plt.subplots_adjust(left=0.12, top=0.95, bottom=0.06, right=0.98, hspace=0.75)
# # Add Colorbars
# caxbase = fig.add_axes([ax1.get_position().x0, 
#     ax1.get_position().y0+0.02, 
#     (ax1.get_position().x1-ax1.get_position().x0)/2.7, 0.02])
# cb = mpl.colorbar.ColorbarBase(caxbase, cmap=cmapbase, norm=normbase, 
#     spacing='proportional', orientation='horizontal', extend='max')
# caxbase.xaxis.offsetText.set_visible(False)
# caxbase = fig.add_axes([ax2.get_position().x0, 
#     ax2.get_position().y0+0.02, 
#     (ax2.get_position().x1-ax2.get_position().x0)/2.7, 
#     0.02])
# cb = mpl.colorbar.ColorbarBase(caxbase, cmap=cmaplock, norm=normlock, 
#     spacing='proportional', orientation='horizontal', extend='both',
#     ticks=[-2e15,-1e15,0,1e15,2e15])
# caxbase.xaxis.offsetText.set_visible(False)
# plt.savefig(DIR_FIGS+'fig1.png', dpi=600)


"""FIGURE S2"""
# def figS2(harmonized, harmonized_rural):
#     """
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from decimal import Decimal
#     fig = plt.figure(figsize=(9,9))
#     ax1 = plt.subplot2grid((13,2),(0,0), rowspan=2)
#     ax2 = plt.subplot2grid((13,2),(0,1), rowspan=2)
#     ax3 = plt.subplot2grid((13,2),(2,0), rowspan=2)
#     ax4 = plt.subplot2grid((13,2),(2,1), rowspan=2)
#     ax5 = plt.subplot2grid((13,2),(4,0), rowspan=2)
#     ax6 = plt.subplot2grid((13,2),(4,1), rowspan=2)
#     ax1b = plt.subplot2grid((13,2),(7,0), rowspan=2)
#     ax2b = plt.subplot2grid((13,2),(7,1), rowspan=2)
#     ax3b = plt.subplot2grid((13,2),(9,0), rowspan=2)
#     ax4b = plt.subplot2grid((13,2),(9,1), rowspan=2)
#     ax5b = plt.subplot2grid((13,2),(11,0), rowspan=2)
#     ax6b = plt.subplot2grid((13,2),(11,1), rowspan=2)
#     colors = ['#0095A8','darkgrey','#FF7043']
#     # # # # Bar charts for demographics
#     # Largest, smallest and median gains
#     increaseno2_all = harmonized.loc[harmonized['NO2_ABS']>
#         np.nanpercentile(harmonized['NO2_ABS'], ptile_upper)]
#     decreaseno2_all = harmonized.loc[harmonized['NO2_ABS']<
#         np.nanpercentile(harmonized['NO2_ABS'], ptile_lower)]
#     increaseno2_rural = harmonized_rural.loc[harmonized_rural['NO2_ABS']>
#         np.nanpercentile(harmonized_rural['NO2_ABS'], ptile_upper)]
#     decreaseno2_rural = harmonized_rural.loc[harmonized_rural['NO2_ABS']<
#         np.nanpercentile(harmonized_rural['NO2_ABS'], ptile_lower)]
#     # # # # Change in NO2
#     # All 
#     ax1.barh([2,1,0], [decreaseno2_all['NO2_ABS'].mean(), 
#         harmonized['NO2_ABS'].mean(), increaseno2_all['NO2_ABS'].mean()], 
#         color=colors[0])
#     print('Largest gains for NO2 = %.2E'%Decimal(
#         decreaseno2_all['NO2_ABS'].mean()))
#     ax1.text(decreaseno2_all['NO2_ABS'].mean()+2e13, 2, '-2.07', color='k', 
#         va='center')
#     print('Mean gains for NO2 = %.2E'%Decimal(harmonized['NO2_ABS'].mean()))
#     ax1.text(harmonized['NO2_ABS'].mean()+2e13, 1, '-0.43', color='k', 
#         va='center')
#     print('Smallest gains for NO2 = %.2E'%Decimal(
#         increaseno2_all['NO2_ABS'].mean()))
#     ax1.text(increaseno2_all['NO2_ABS'].mean()+2e13, 0, '0.23', color='k', 
#         va='center')
#     ax1b.barh([2,1,0], [decreaseno2_rural['NO2_ABS'].mean(), harmonized_rural[
#         'NO2_ABS'].mean(), increaseno2_rural['NO2_ABS'].mean()], 
#         color=colors[0])
#     print('Largest gains for NO2 = %.2E'%Decimal(
#         decreaseno2_rural['NO2_ABS'].mean()))
#     ax1b.text(decreaseno2_rural['NO2_ABS'].mean()+10e13, 2, '-0.79', color='k', 
#         va='center')
#     print('Mean gains for NO2 = %.2E'%Decimal(
#         harmonized_rural['NO2_ABS'].mean()))
#     ax1b.text(harmonized_rural['NO2_ABS'].mean()-2.5e14, 1, '-0.15', color='k', 
#         va='center')
#     print('Smallest gains for NO2 = %.2E'%Decimal(
#         increaseno2_rural['NO2_ABS'].mean()))
#     ax1b.text(increaseno2_rural['NO2_ABS'].mean()+2e13, 0, '0.26', color='k', 
#         va='center')
#     # # # # Income
#     ax3.barh([2,1,0], [decreaseno2_all['AJZAE001'].mean(), 
#         harmonized['AJZAE001'].mean(), increaseno2_all['AJZAE001'].mean()], 
#         color=colors[0])
#     ax3.text(53000, 2, ' %d'%(decreaseno2_all['AJZAE001'].mean()), color='k', 
#         va='center')
#     ax3.text(53000, 1, ' %d'%(harmonized['AJZAE001'].mean()), color='k', 
#         va='center')
#     ax3.text(53000, 0, ' %d'%(increaseno2_all['AJZAE001'].mean()), color='k', 
#         va='center')
#     ax3b.barh([2,1,0], [decreaseno2_rural['AJZAE001'].mean(), 
#         harmonized['AJZAE001'].mean(), increaseno2_rural['AJZAE001'].mean()], 
#         color=colors[0])
#     ax3b.text(53000, 2, ' %d'%(decreaseno2_rural['AJZAE001'].mean()), 
#         color='k', va='center')
#     ax3b.text(53000, 1, ' %d'%(harmonized_rural['AJZAE001'].mean()), 
#         color='k', va='center')
#     ax3b.text(53000, 0, ' %d'%(increaseno2_rural['AJZAE001'].mean()), 
#         color='k', va='center')
#     # # # # Racial background
#     left, i = 0, 0
#     labels = ['White', 'Black', 'Other']
#     # Largest gains
#     for data, color in zip([(decreaseno2_all['AJWNE002']/
#         decreaseno2_all['AJWBE001']).mean(),
#         (decreaseno2_all['AJWNE003']/decreaseno2_all['AJWBE001']).mean(),      
#         ((decreaseno2_all['AJWNE004']+decreaseno2_all['AJWNE005']+
#           decreaseno2_all['AJWNE006']+decreaseno2_all['AJWNE007']+
#         decreaseno2_all['AJWNE008'])/decreaseno2_all['AJWBE001']).mean()], 
#             colors):             
#         ax5.barh(2, data, color=color, left=left)
#         ax5.text(left+0.01, 2, '%d'%(np.round(data,2)*100), color='k', 
#             va='center')
#         left += data
#         i = i+1
#     left, i = 0, 0   
#     for data, color in zip([(decreaseno2_rural['AJWNE002']/
#         decreaseno2_rural['AJWBE001']).mean(),
#         (decreaseno2_rural['AJWNE003']/decreaseno2_rural['AJWBE001']).mean(),      
#         ((decreaseno2_rural['AJWNE004']+decreaseno2_rural['AJWNE005']+
#           decreaseno2_rural['AJWNE006']+decreaseno2_rural['AJWNE007']+
#         decreaseno2_rural['AJWNE008'])/decreaseno2_rural['AJWBE001']).mean()], 
#             colors):             
#         ax5b.barh(2, data, color=color, left=left)
#         ax5b.text(left+0.01, 2, '%d'%(np.round(data,2)*100), color='k', 
#             va='center')
#         left += data
#         i = i+1
#     # Mean demographics
#     left, i = 0, 0
#     for data, color in zip([(harmonized['AJWNE002']/
#         harmonized['AJWBE001']).mean(),
#         (harmonized['AJWNE003']/harmonized['AJWBE001']).mean(),      
#         ((harmonized['AJWNE004']+harmonized['AJWNE005']+
#           harmonized['AJWNE006']+harmonized['AJWNE007']+
#         harmonized['AJWNE008'])/harmonized['AJWBE001']).mean()], colors):             
#         ax5.barh(1, data, color=color, left=left)
#         ax5.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='k', 
#             va='center')    
#         left += data
#         i = i+1    
#     left, i = 0, 0
#     for data, color in zip([(harmonized_rural['AJWNE002']/
#         harmonized_rural['AJWBE001']).mean(),
#         (harmonized_rural['AJWNE003']/harmonized_rural['AJWBE001']).mean(),      
#         ((harmonized_rural['AJWNE004']+harmonized_rural['AJWNE005']+
#           harmonized_rural['AJWNE006']+harmonized_rural['AJWNE007']+
#         harmonized_rural['AJWNE008'])/harmonized_rural['AJWBE001']).mean()], 
#             colors):             
#         ax5b.barh(1, data, color=color, left=left)
#         ax5b.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='k', 
#             va='center')    
#         left += data
#         i = i+1        
#     # Smallest gains
#     left, i = 0, 0
#     for data, color in zip([(increaseno2_all['AJWNE002']/
#         increaseno2_all['AJWBE001']).mean(),
#         (increaseno2_all['AJWNE003']/increaseno2_all['AJWBE001']).mean(),      
#         ((increaseno2_all['AJWNE004']+increaseno2_all['AJWNE005']+
#           increaseno2_all['AJWNE006']+increaseno2_all['AJWNE007']+
#         increaseno2_all['AJWNE008'])/increaseno2_all['AJWBE001']).mean()], 
#             colors):             
#         ax5.barh(0, data, color=color, left=left)
#         ax5.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
#             va='center')    
#         if i==2:
#             ax5.text(0.88, -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold')
#         if i==1:
#             ax5.text(0.75, -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold')
#         if i==0:
#             ax5.text(left+0.01, -0.9, labels[i], color=colors[i], va='center',
#                       fontweight='bold')
#         left += data       
#         i = i+1    
#     left, i = 0, 0
#     for data, color in zip([(increaseno2_rural['AJWNE002']/
#         increaseno2_rural['AJWBE001']).mean(),
#         (increaseno2_rural['AJWNE003']/increaseno2_rural['AJWBE001']).mean(),      
#         ((increaseno2_rural['AJWNE004']+increaseno2_rural['AJWNE005']+
#           increaseno2_rural['AJWNE006']+increaseno2_rural['AJWNE007']+
#         increaseno2_rural['AJWNE008'])/increaseno2_rural['AJWBE001']).mean()], 
#             colors):             
#         ax5b.barh(0, data, color=color, left=left)
#         ax5b.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
#             va='center')    
#         if i==2:
#             ax5b.text(0.88, -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold')
#         if i==1:
#             ax5b.text(0.75, -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold')
#         if i==0:
#             ax5b.text(left+0.01, -0.9, labels[i], color=colors[i], va='center',
#                       fontweight='bold')
#         left += data       
#         i = i+1
#     # # # # Ethnic background
#     # Largest gains
#     left, i = 0, 0
#     labels = ['Hispanic', 'Non-Hispanic']
#     for data, color in zip([(decreaseno2_all['AJWWE003']/
#         decreaseno2_all['AJWWE001']).mean(),
#         (decreaseno2_all['AJWWE002']/decreaseno2_all['AJWWE001']).mean()], 
#             colors):             
#         ax2.barh(2, data, color=color, left=left)
#         ax2.text(left+0.01, 2, '%d'%(np.round(data,2)*100), color='k', 
#             va='center')
#         left += data
#         i = i+1
#     left, i = 0, 0
#     for data, color in zip([(decreaseno2_rural['AJWWE003']/
#         decreaseno2_rural['AJWWE001']).mean(),
#         (decreaseno2_rural['AJWWE002']/decreaseno2_rural['AJWWE001']).mean()], 
#             colors):             
#         ax2b.barh(2, data, color=color, left=left)
#         ax2b.text(left+0.01, 2, '%d'%(np.round(data,2)*100), color='k', 
#             va='center')
#         left += data
#         i = i+1    
#     # Mean demographics
#     left, i = 0, 0
#     for data, color in zip([(harmonized['AJWWE003']/
#         harmonized['AJWWE001']).mean(),
#         (harmonized['AJWWE002']/harmonized['AJWWE001']).mean()], colors):             
#         ax2.barh(1, data, color=color, left=left)
#         ax2.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='k', 
#             va='center')
#         left += data
#         i = i+1
#     left, i = 0, 0
#     for data, color in zip([(harmonized_rural['AJWWE003']/
#         harmonized_rural['AJWWE001']).mean(),
#         (harmonized_rural['AJWWE002']/harmonized_rural['AJWWE001']).mean()], 
#             colors):
#         ax2b.barh(1, data, color=color, left=left)
#         ax2b.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='k', 
#             va='center')
#         left += data
#         i = i+1
#     # Smallest gains
#     left, i = 0, 0
#     for data, color in zip([(increaseno2_all['AJWWE003']/
#         increaseno2_all['AJWWE001']).mean(),
#         (increaseno2_all['AJWWE002']/increaseno2_all['AJWWE001']).mean()], 
#             colors): 
#         ax2.barh(0, data, color=color, left=left)
#         ax2.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
#             va='center')    
#         if i==1:
#             ax2.text(0.2, -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold')            
#         else:
#             ax2.text(left+0.01, -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold')    
#         left += data
#         i = i+1
#     left, i = 0, 0
#     for data, color in zip([(increaseno2_rural['AJWWE003']/
#         increaseno2_rural['AJWWE001']).mean(),
#         (increaseno2_rural['AJWWE002']/increaseno2_rural['AJWWE001']).mean()], 
#             colors): 
#         ax2b.barh(0, data, color=color, left=left)
#         ax2b.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
#             va='center')    
#         if i==1:
#             ax2b.text(0.2, -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold')            
#         else:
#             ax2b.text(left+0.01, -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold')             
#         left += data
#         i = i+1
#     # # # # Educational attainment
#     # Largest gains
#     left, i = 0, 0
#     for data, color in zip([
#         (decreaseno2_all.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
#         decreaseno2_all['AJYPE001']).mean(),
#         (decreaseno2_all.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
#         decreaseno2_all['AJYPE001']).mean(),
#         (decreaseno2_all.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
#         decreaseno2_all['AJYPE001']).mean()], colors):             
#         ax4.barh(2, data, color=color, left=left)
#         ax4.text(left+0.01, 2, '%d'%(np.round(data,2)*100), 
#             color='black', va='center') 
#         left += data
#         i = i+1    
#     left, i = 0, 0
#     for data, color in zip([
#         (decreaseno2_rural.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
#         decreaseno2_rural['AJYPE001']).mean(),
#         (decreaseno2_rural.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
#         decreaseno2_rural['AJYPE001']).mean(),
#         (decreaseno2_rural.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
#         decreaseno2_rural['AJYPE001']).mean()], colors):             
#         ax4b.barh(2, data, color=color, left=left)
#         ax4b.text(left+0.01, 2, '%d'%(np.round(data,2)*100), 
#             color='k', va='center') 
#         left += data
#         i = i+1        
#     # Mean demographics
#     left, i = 0, 0
#     for data, color in zip([
#         (harmonized.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
#         harmonized['AJYPE001']).mean(),
#         (harmonized.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
#         harmonized['AJYPE001']).mean(),
#         (harmonized.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
#         harmonized['AJYPE001']).mean()], colors):             
#         ax4.barh(1, data, color=color, left=left)
#         ax4.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='k', 
#             va='center') 
#         left += data
#         i = i+1 
#     left, i = 0, 0
#     for data, color in zip([
#         (harmonized_rural.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
#         harmonized_rural['AJYPE001']).mean(),
#         (harmonized_rural.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
#         harmonized_rural['AJYPE001']).mean(),
#         (harmonized_rural.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
#         harmonized_rural['AJYPE001']).mean()], colors):             
#         ax4b.barh(1, data, color=color, left=left)
#         ax4b.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='k', 
#             va='center') 
#         left += data
#         i = i+1     
#     # Smallest gains
#     left, i = 0, 0
#     labels = ['High school', 'College', 'Graduate']
#     for data, color in zip([
#         (increaseno2_all.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
#         increaseno2_all['AJYPE001']).mean(),
#         (increaseno2_all.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
#         increaseno2_all['AJYPE001']).mean(),
#         (increaseno2_all.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
#         increaseno2_all['AJYPE001']).mean()], colors):             
#         ax4.barh(0, data, color=color, left=left)
#         ax4.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
#             va='center')
#         if i==2:
#             ax4.text(0.82, -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold', fontsize=10)         
#         else:   
#             ax4.text(left, -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold', fontsize=10) 
#         left += data
#         i = i+1  
#     left, i = 0, 0
#     for data, color in zip([
#         (increaseno2_rural.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
#         increaseno2_rural['AJYPE001']).mean(),
#         (increaseno2_rural.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
#         increaseno2_rural['AJYPE001']).mean(),
#         (increaseno2_rural.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
#         increaseno2_rural['AJYPE001']).mean()], colors):             
#         ax4b.barh(0, data, color=color, left=left)
#         ax4b.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
#             va='center')
#         if i==2:
#             ax4b.text(0.82, -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold', fontsize=10)         
#         else:   
#             ax4b.text(left, -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold', fontsize=10) 
#         left += data
#         i = i+1  
#     # # # # Vehicle ownership
#     # Largest gains
#     left, i = 0, 0
#     labels = ['None', 'One or more']
#     for data, color in zip([decreaseno2_all['FracNoCar'].mean(),
#         (1-decreaseno2_all['FracNoCar'].mean())], colors):             
#         ax6.barh(2, data, color=color, left=left)
#         ax6.text(left+0.01, 2, '%d'%(np.round(data,2)*100),
#             color='k', va='center') 
#         left += data
#         i = i+1
#     left, i = 0, 0
#     for data, color in zip([decreaseno2_rural['FracNoCar'].mean(),
#         (1-decreaseno2_rural['FracNoCar'].mean())], colors):             
#         ax6b.barh(2, data, color=color, left=left)
#         ax6b.text(left+0.01, 2, '%d'%(np.round(data,2)*100),
#             color='k', va='center') 
#         left += data
#         i = i+1    
#     # Mean demographics
#     left, i = 0, 0
#     for data, color in zip([harmonized['FracNoCar'].mean(),
#         (1-harmonized['FracNoCar'].mean())], colors):                        
#         ax6.barh(1, data, color=color, left=left)
#         ax6.text(left+0.01, 1, '%d'%(np.round(data,2)*100),
#             color='k', va='center')     
#         left += data
#         i = i+1
#     left, i = 0, 0
#     for data, color in zip([harmonized_rural['FracNoCar'].mean(),
#         (1-harmonized_rural['FracNoCar'].mean())], colors):                        
#         ax6b.barh(1, data, color=color, left=left)
#         ax6b.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='k', 
#             va='center') 
#         left += data
#         i = i+1    
#     # Smallest gains
#     left, i = 0, 0
#     for data, color in zip([increaseno2_all['FracNoCar'].mean(),
#         (1-increaseno2_all['FracNoCar'].mean())], colors):             
#         ax6.barh(0, data, color=color, left=left)
#         ax6.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
#             va='center') 
#         if i==1:
#             ax6.text(0.15, -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold', fontsize=10)
#         else:       
#             ax6.text((left*1.5), -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold', fontsize=10)          
#         left += data       
#         i = i+1
#     left, i = 0, 0
#     for data, color in zip([increaseno2_rural['FracNoCar'].mean(),
#         (1-increaseno2_rural['FracNoCar'].mean())], colors):             
#         ax6b.barh(0, data, color=color, left=left)
#         ax6b.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
#             va='center') 
#         if i==1:
#             ax6b.text(0.15, -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold', fontsize=10)
#         else:       
#             ax6b.text((left*1.5), -0.9, labels[i], color=colors[i], va='center',
#                 fontweight='bold', fontsize=10)
#         left += data       
#         i = i+1    
#     # Aesthetics    
#     for ax in [ax1, ax1b]:
#         ax.set_xlim(-2.1e15,3e14)
#         ax.set_xticks([])    
#     for ax in [ax3,ax3b]:
#         ax.set_xlim([53000,80000])
#         ax.set_xticks([])
#     for ax in [ax5,ax5b,ax2,ax2b,ax4,ax4b,ax6,ax6b]:
#         ax.set_xlim([0,1.])
#         ax.set_xticks([])
#         ax.set_xticklabels([])
#     for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax1b,ax2b,ax3b,ax4b,ax5b,ax6b]:
#         ax.set_ylim([-0.5,2.5])
#         ax.set_yticks([0,1,2])
#         ax.set_yticklabels([])
#         for pos in ['right','top','bottom']:
#             ax.spines[pos].set_visible(False)
#             ax.spines[pos].set_visible(False)    
#     for ax in [ax1,ax3,ax5,ax1b,ax3b,ax5b]:
#         ax.set_yticklabels(['Smallest gains', 'Average', 'Largest gains'], 
#             fontsize=10)
#     # Axis titles
#     ax1.set_title('(a) $\Delta\:$NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', 
#         loc='left', fontsize=10)
#     ax1b.set_title('(g) $\Delta\:$NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', 
#         loc='left', fontsize=10)
#     ax2.set_title('(b) Ethnic background [%]', loc='left', fontsize=10)
#     ax2b.set_title('(h) Ethnic background [%]', loc='left', fontsize=10)
#     ax3.set_title('(c) Household income [$]', loc='left', fontsize=10)
#     ax3b.set_title('(i) Household income [$]', loc='left', fontsize=10)
#     ax4.set_title('(d) Educational attainment [%]',loc='left', fontsize=10)
#     ax4b.set_title('(j) Educational attainment [%]',loc='left', fontsize=10)
#     ax5.set_title('(e) Racial background [%]', loc='left', fontsize=10)
#     ax5b.set_title('(k) Racial background [%]', loc='left', fontsize=10)
#     ax6.set_title('(f) Household vehicle ownership [%]',loc='left', fontsize=10)
#     ax6b.set_title('(l) Household vehicle ownership [%]',loc='left', fontsize=10)
#     fig.text(0.5, 0.98, '$\mathbf{All}$', fontsize=14, ha='center')
#     fig.text(0.5, 0.48, '$\mathbf{Rural}$', fontsize=14, ha='center')
#     plt.subplots_adjust(top=0.95, bottom=0.05, hspace=3)
#     plt.savefig(DIR_FIGS+'figS2.png', dpi=600)
#     return 

"""FIGURE 2"""
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# # Top 15 MSAs (can be found at 
# # https://en.wikipedia.org/wiki/List_of_metropolitan_statistical_areas
# # Wikipedia usually lists the counties included in each MSA (but note that
# # this is different than the CSA, which includes more counties). The 
# # corresponding FIPS codes for each county can be found at 
# # www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697
# # New York-Newark-Jersey City, NY-NJ-PA MSA
# newyork = ['36047','36081','36061','36005','36085','36119','34003',
#     '34017','34031','36079','36087','36103','36059','34023','34025',
#     '34029','34035','34013','34039','34027','34037'	,'34019','42103']
# # Los Angeles-Long Beach-Anaheim, CA MSA
# losangeles = ['06037','06059']
# # Chicago-Naperville-Elgin, IL-IN-WI MSA
# chicago = ['17031','17037','17043','17063','17091','17089','17093',
#     '17111','17197','18073','18089','18111','18127','17097','55059']
# # Dallas-Fort Worth-Arlington, TX MSA
# dallas = ['48085','48113','48121','48139','48231','48257','48397',
#     '48251','48367','48439','48497']
# # Houston-The Woodlands-Sugar Land, TX MSA
# houston = ['48201','48157','48339','48039','48167','48291','48473',
#     '48071','48015']
# # Washington-Arlington-Alexandria, DC-VA-MD-WV MSA
# washington = ['11001','24009','24017','24021','24031','24033','51510',
#     '51013','51043','51047','51059','51600','51610','51061','51630',
#     '51107','51683','51685','51153','51157','51177','51179','51187']
# # Miami-Fort Lauderdale-Pompano Beach, FL MSA	
# miami = ['12086','12011','12099']
# # Philadelphia-Camden-Wilmington, PA-NJ-DE-MD MSA
# philadelphia = ['34005','34007','34015','42017','42029','42091','42045',
#     '42101','10003','24015','34033']
# # Atlanta-Sandy Springs-Alpharetta, GA MSA
# atlanta = ['13121','13135','13067','13089','13063','13057','13117',
#     '13151','13223','13077','13097','13045','13113','13217','13015',
#     '13297','13247','13013','13255','13227','13143','13085','13035',
#     '13199','13171','13211','13231','13159','13149']
# # Phoenix-Mesa-Chandler, AZ MSA
# phoenix = ['04013','04021','04007']
# # Boston-Cambridge-Newton, MA-NH MSA
# boston = ['25021','25023','25025','25009','25017','33015','33017']
# # San Francisco-Oakland-Berkeley, CA MSA
# sanfrancisco = ['06001','06013','06075','06081','06041']
# # Riverside-San Bernardino-Ontario, CA MSA
# riverside = ['06065','06071']
# # Detroit-Warren-Dearborn, MI MSA
# detroit = ['26163','26125','26099','26093','26147','26087']
# # Seattle-Tacoma-Bellevue, WA MSA
# seattle = ['53033','53061','53053']
# # Colors
# color_white = '#0095A8'
# color_non = '#FF7043'
# os = 1.2
# # Initialize figure
# fig = plt.figure(figsize=(12,7))
# ax1 = plt.subplot2grid((2,3),(0,0),rowspan=2)
# ax2 = plt.subplot2grid((2,3),(0,1),rowspan=2)
# ax3 = plt.subplot2grid((2,3),(0,2),rowspan=2)
# i = 0 
# yticks = []
# # For all tracts
# frac_white = (harmonized['AJWNE002']/harmonized['AJWBE001'])
# mostwhite = harmonized.iloc[np.where(frac_white > 
#     np.nanpercentile(frac_white, ptile_upper))]
# leastwhite = harmonized.iloc[np.where(frac_white < 
#     np.nanpercentile(frac_white, ptile_lower))]
# # Lockdown NO2  or white and non-white populations
# ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
# ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
# ax1.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()]), 
#     np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()])+
#     np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))[0]), 
#     [i-os,i-os], color='k', ls='-', zorder=10)    
# # Historic NO2 for white and non-white populations
# ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
# ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
# # Draw connection lines
# ax1.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
#     np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
#     np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))[0]), 
#     [i+os,i+os], color='k', ls='--', zorder=10)
# yticks.append(np.nanmean([i]))
# i = i+7    
# # For rural tracts
# frac_white = (harmonized_rural['AJWNE002']/harmonized_rural['AJWBE001'])
# mostwhite = harmonized_rural.iloc[np.where(frac_white > 
#     np.nanpercentile(frac_white, ptile_upper))]
# leastwhite = harmonized_rural.iloc[np.where(frac_white < 
#     np.nanpercentile(frac_white, ptile_lower))]
# ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, 
#     zorder=12)
# ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, 
#     zorder=12)
# ax1.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()]), 
#     np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()])+
#     np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))[0]), 
#     [i-os,i-os], color='k', ls='-', zorder=10)    
# ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, 
#     zorder=12)
# ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, 
#     zorder=12)
# ax1.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
#     np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
#     np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))[0]), 
#     [i+os,i+os], color='k', ls='--', zorder=10)
# yticks.append(np.nanmean([i]))
# i = i+7  
# # For rural tracts
# frac_white = (harmonized_urban['AJWNE002']/harmonized_urban['AJWBE001'])
# mostwhite = harmonized_urban.iloc[np.where(frac_white > 
#     np.nanpercentile(frac_white, ptile_upper))]
# leastwhite = harmonized_urban.iloc[np.where(frac_white < 
#     np.nanpercentile(frac_white, ptile_lower))]
# ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, 
#     zorder=12)
# ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, 
#     zorder=12)
# ax1.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()]), 
#     np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()])+
#     np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))[0]), 
#     [i-os,i-os], color='k', ls='-', zorder=10)    
# ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, 
#     zorder=12)
# ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, 
#     zorder=12)
# ax1.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
#     np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
#     np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))[0]), 
#     [i+os,i+os], color='k', ls='--', zorder=10)
# yticks.append(np.nanmean([i]))
# i = i+7        
# ratio_pre = []
# ratio_post = []
# citynames = [r'$\bf{All}$', r'$\bf{Rural}$', r'$\bf{Urban}$',
#     'New York', 'Los Angeles', 'Chicago', 'Dallas', 'Houston', 
#     'Washington', 'Miami', 'Philadelphia', 'Atlanta', 'Phoenix', 
#     'Boston', 'San Francisco', 'Riverside', 'Detroit', 'Seattle']   
# for city in [newyork, losangeles, chicago, dallas, houston, washington,
#     miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
#     riverside, detroit, seattle]:
#     # Subset for city
#     harmonized_city = subset_harmonized_bycountyfips(harmonized, city)
#     # Find particular demographic for each city
#     frac_white = (harmonized_city['AJWNE002']/harmonized_city['AJWBE001'])
#     mostwhite = harmonized_city.iloc[np.where(frac_white > 
#         np.nanpercentile(frac_white, ptile_upper))]
#     leastwhite = harmonized_city.iloc[np.where(frac_white < 
#         np.nanpercentile(frac_white, ptile_lower))]
#     # Lockdown NO2  or white and non-white populations
#     ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
#     ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
#     ratio_pre.append(leastwhite['PRENO2'].mean()/mostwhite['PRENO2'].mean())
#     ax1.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()]), 
#         np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()])+
#         np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))[0]), 
#         [i-os,i-os], color='k', ls='-', zorder=10)    
#     # i = i+2
#     # Historic NO2 for white and non-white populations
#     ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
#     ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
#     ratio_post.append(leastwhite['POSTNO2'].mean()/mostwhite['POSTNO2'].mean())
#     # Draw connection lines
#     ax1.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
#         np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
#         np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))[0]), 
#         [i+os,i+os], color='k', ls='--', zorder=10)
#     yticks.append(np.nanmean([i]))
#     i = i+7
# # Aesthetics 
# ax1.set_xlim([0.5e15,10e15])
# ax1.set_xticks(np.arange(1e15,10e15,1e15))
# ax1.set_yticks(yticks)
# ax1.set_yticklabels(citynames)
# ax1.tick_params(axis='y', left=False, length=0)
# ax1.set_xlabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', x=0.26, 
#     color='darkgrey')
# ax1.tick_params(axis='x', colors='darkgrey')
# ax1.xaxis.offsetText.set_visible(False)
# for side in ['right', 'left', 'top', 'bottom']:
#     ax1.spines[side].set_visible(False)
# ax1.grid(axis='x', zorder=0, color='darkgrey')
# ax1.invert_yaxis()

# # # # Most versus least wealthy
# i = 0 
# yticks = []
# mostwealthy = harmonized.loc[harmonized['AJZAE001'] > 
#     np.nanpercentile(harmonized['AJZAE001'], 90)]
# leastwealthy = harmonized.loc[harmonized['AJZAE001'] < 
#     np.nanpercentile(harmonized['AJZAE001'], 10)]
# ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
# ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
# ax2.plot((np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()]), 
#     np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+
#     np.abs(np.diff([mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
#     [i-os,i-os], color='k', ls='-', zorder=10)    
# ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
# ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
# ax2.plot((np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()]), 
#     np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
#     np.abs(np.diff([mostwealthy['POSTNO2'].mean(), leastwealthy['POSTNO2'].mean()]))[0]), 
#     [i+os,i+os], color='k', ls='--', zorder=10)
# yticks.append(np.nanmean([i]))
# i = i+7    
# mostwealthy = harmonized_rural.loc[harmonized_rural['AJZAE001'] > 
#     np.nanpercentile(harmonized_rural['AJZAE001'], 90)]
# leastwealthy = harmonized_rural.loc[harmonized_rural['AJZAE001'] < 
#     np.nanpercentile(harmonized_rural['AJZAE001'], 10)]
# ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, 
#     zorder=12)
# ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, 
#     zorder=12)
# ax2.plot((np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()]), 
#     np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+
#     np.abs(np.diff([mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
#     [i-os,i-os], color='k', ls='-', zorder=10)    
# ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, 
#     zorder=12)
# ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, 
#     zorder=12)
# ax2.plot((np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()]), 
#     np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
#     np.abs(np.diff([mostwealthy['POSTNO2'].mean(), leastwealthy['POSTNO2'].mean()]))[0]), 
#     [i+os,i+os], color='k', ls='--', zorder=10)
# yticks.append(np.nanmean([i]))
# i = i+7  
# mostwealthy = harmonized_urban.loc[harmonized_urban['AJZAE001'] > 
#     np.nanpercentile(harmonized_urban['AJZAE001'], 90)]
# leastwealthy = harmonized_urban.loc[harmonized_urban['AJZAE001'] < 
#     np.nanpercentile(harmonized_urban['AJZAE001'], 10)]
# ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, 
#     zorder=12)
# ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, 
#     zorder=12)
# ax2.plot((np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()]), 
#     np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+
#     np.abs(np.diff([mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
#     [i-os,i-os], color='k', ls='-', zorder=10)    
# ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, 
#     zorder=12)
# ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, 
#     zorder=12)
# ax2.plot((np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()]), 
#     np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
#     np.abs(np.diff([mostwealthy['POSTNO2'].mean(), leastwealthy['POSTNO2'].mean()]))[0]), 
#     [i+os,i+os], color='k', ls='--', zorder=10)
# yticks.append(np.nanmean([i]))
# i = i+7      
# ratio_pre = []
# ratio_post = []
# for city in [newyork, losangeles, chicago, dallas, houston, washington,
#     miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
#     riverside, detroit, seattle]:
#     harmonized_city = subset_harmonized_bycountyfips(harmonized, city)
#     mostwealthy = harmonized_city.loc[harmonized_city['AJZAE001'] > 
#         np.nanpercentile(harmonized_city['AJZAE001'], 90)]
#     leastwealthy = harmonized_city.loc[harmonized_city['AJZAE001'] < 
#         np.nanpercentile(harmonized_city['AJZAE001'], 10)]
#     ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, 
#         zorder=12)
#     ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, 
#         zorder=12)
#     ax2.plot((np.min([mostwealthy['PRENO2'].mean(),
#         leastwealthy['PRENO2'].mean()]), np.min([mostwealthy[
#         'PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+np.abs(np.diff(
#         [mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
#         [i-os,i-os], color='k', ls='-', zorder=10)    
#     ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, 
#         zorder=12)
#     ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, 
#         zorder=12)
#     ax2.plot((np.min([mostwealthy['POSTNO2'].mean(),
#         leastwealthy['POSTNO2'].mean()]), np.min([
#         mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
#         np.abs(np.diff([mostwealthy['POSTNO2'].mean(), 
#         leastwealthy['POSTNO2'].mean()]))[0]), [i+os,i+os], color='k', 
#         ls='--', zorder=10)
#     yticks.append(np.nanmean([i]))
#     ratio_pre.append(leastwealthy['PRENO2'].mean()/mostwealthy['PRENO2'].mean())
#     ratio_post.append(leastwealthy['POSTNO2'].mean()/mostwealthy['POSTNO2'].mean())
#     i = i+7
# # Aesthetics 
# ax2.set_xlim([0.5e15,10e15])
# ax2.set_xticks(np.arange(1e15,10e15,1e15))
# ax2.set_yticks(yticks)
# ax2.set_yticklabels([])
# ax2.tick_params(axis='y', left=False)
# ax2.set_xlabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', x=0.26, 
#     color='darkgrey')
# ax2.tick_params(axis='x', colors='darkgrey')
# ax2.xaxis.offsetText.set_visible(False)
# for side in ['right', 'left', 'top', 'bottom']:
#     ax2.spines[side].set_visible(False)
# ax2.grid(axis='x', zorder=0, color='darkgrey')
# ax2.invert_yaxis()

# # Most versus least educated
# i = 0 
# yticks = []
# frac_educated = (harmonized.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
#     harmonized['AJYPE001'])
# mosteducated = harmonized.iloc[np.where(frac_educated > 
#     np.nanpercentile(frac_educated, 90))]
# leasteducated = harmonized.iloc[np.where(frac_educated < 
#     np.nanpercentile(frac_educated, 10))]
# ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
# ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
# ax3.plot((np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()]), 
#     np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()])+
#     np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
#     [i-os,i-os], color='k', ls='-', zorder=10)    
# ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
# ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
# ax3.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
#     np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
#     np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
#     [i+os,i+os], color='k', ls='--', zorder=10)
# yticks.append(np.nanmean([i]))
# i = i+7    
# frac_educated = (harmonized_rural.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
#     harmonized_rural['AJYPE001'])
# mosteducated = harmonized_rural.iloc[np.where(frac_educated > 
#     np.nanpercentile(frac_educated, 90))]
# leasteducated = harmonized_rural.iloc[np.where(frac_educated < 
#     np.nanpercentile(frac_educated, 10))]
# ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
# ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
# ax3.plot((np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()]), 
#     np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()])+
#     np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
#     [i-os,i-os], color='k', ls='-', zorder=10)    
# ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
# ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
# ax3.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
#     np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
#     np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
#     [i+os,i+os], color='k', ls='--', zorder=10)
# yticks.append(np.nanmean([i]))
# i = i+7    
# frac_educated = (harmonized_urban.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
#     harmonized_urban['AJYPE001'])
# mosteducated = harmonized_urban.iloc[np.where(frac_educated > 
#     np.nanpercentile(frac_educated, 90))]
# leasteducated = harmonized_urban.iloc[np.where(frac_educated < 
#     np.nanpercentile(frac_educated, 10))]
# ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
# ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
# ax3.plot((np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()]), 
#     np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()])+
#     np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
#     [i-os,i-os], color='k', ls='-', zorder=10)    
# ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
# ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
# ax3.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
#     np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
#     np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
#     [i+os,i+os], color='k', ls='--', zorder=10)
# yticks.append(np.nanmean([i]))
# i = i+7    
# for city in [newyork, losangeles, chicago, dallas, houston, washington,
#     miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
#     riverside, detroit, seattle]:
#     harmonized_city = subset_harmonized_bycountyfips(harmonized, city)
#     frac_educated = (harmonized_city.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
#         harmonized_city['AJYPE001'])
#     mosteducated = harmonized_city.iloc[np.where(frac_educated > 
#         np.nanpercentile(frac_educated, 90))]
#     leasteducated = harmonized_city.iloc[np.where(frac_educated < 
#         np.nanpercentile(frac_educated, 10))]
#     ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, 
#         zorder=12)
#     ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, 
#         zorder=12)
#     ax3.plot((np.min([mosteducated['PRENO2'].mean(),
#         leasteducated['PRENO2'].mean()]), 
#         np.min([mosteducated['PRENO2'].mean(),
#         leasteducated['PRENO2'].mean()])+np.abs(np.diff(
#         [mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
#         [i-os,i-os], color='k', ls='-', zorder=10)    
#     ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, 
#         zorder=12)
#     ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, 
#         zorder=12)
#     ax3.plot((np.min([mosteducated['POSTNO2'].mean(), leasteducated[
#         'POSTNO2'].mean()]), np.min([mosteducated['POSTNO2'].mean(), 
#         leasteducated['POSTNO2'].mean()])+np.abs(np.diff(
#         [mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
#         [i+os,i+os], color='k', ls='--', zorder=10)
#     yticks.append(np.nanmean([i]))
#     i = i+7
# ax3.set_xlim([0.5e15,10e15])
# ax3.set_xticks(np.arange(1e15,10e15,1e15))
# ax3.set_yticks(yticks)
# ax3.set_yticklabels([])
# ax3.tick_params(axis='y', left=False)
# ax3.set_xlabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', x=0.26, 
#     color='darkgrey')
# ax3.tick_params(axis='x', colors='darkgrey')
# ax3.xaxis.offsetText.set_visible(False)
# for side in ['right', 'left', 'top', 'bottom']:
#     ax3.spines[side].set_visible(False)
# ax3.grid(axis='x', zorder=0, color='darkgrey')
# ax3.invert_yaxis()
# ax1.set_title('(a) Racial background', loc='left', fontsize=10)
# ax2.set_title('(b) Household income', loc='left', fontsize=10)
# ax3.set_title('(c) Educational attainment', loc='left', fontsize=10)
# plt.subplots_adjust(wspace=0.05, left=0.09, top=0.95, bottom=0.17, 
#     right=0.98)
# # Custom legends for different colored scatterpoints
# custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
#     Line2D([0], [0], marker='o', color=color_non, lw=0)]
# ax1.legend(custom_lines, ['Most white', 'Least white'], 
#     bbox_to_anchor=(0.48, -0.15), loc=8, ncol=2, fontsize=10, 
#     frameon=False)
# custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
#     Line2D([0], [0], marker='o', color=color_non, lw=0)]
# ax2.legend(custom_lines, ['Highest income', 'Lowest income'], 
#     bbox_to_anchor=(0.48, -0.15), loc=8, ncol=2, fontsize=10, 
#     frameon=False)
# custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
#     Line2D([0], [0], marker='o', color=color_non, lw=0)]
# ax3.legend(custom_lines, ['Most educated', 'Least educated'], 
#     bbox_to_anchor=(0.48, -0.15), loc=8, ncol=2, fontsize=10, 
#     frameon=False)
# # Custom legend for baseline vs. lockdown bars
# ax1t = ax1.twinx()
# ax1t.axis('off')
# custom_lines = [Line2D([0], [0], color='k', marker='o', lw=2),
#     Line2D([0], [0], color='k', marker='o', ls='--', lw=2)]
# ax1t.legend(custom_lines, ['Baseline', 'Lockdown'], ncol=2, loc=8, 
#     bbox_to_anchor=(0.58, -0.2), numpoints=2, frameon=False, 
#     handlelength=5)
# plt.savefig(DIR_FIGS+'lollipop.png', dpi=500)
# plt.show()

"""FIGURE SX"""
# import pandas as pd
# # NHGIS census information version (may need to change if there are 
# # updates to census information)
# nhgis_version = '0003_ds239_20185_2018'
# # DataFrame that will be filled with harmonzied data for multiple states
# state_harm = pd.DataFrame()
# # Loop through states of interest and read in harmonized NO2/census data
# for FIPS_i in FIPS:
#     state_harm_i = pd.read_csv(DIR_HARM+'Tropomi_NO2_updated/'+
#         'Tropomi_NO2_updated_%s_nhgis%s_tract.csv'%(FIPS_i, nhgis_version), 
#         delimiter=',', header=0, engine='python')
#     # For states with FIPS codes 0-9, there is no leading zero in their 
#     # GEOID row, so add one such that all GEOIDs for any particular state
#     # are identical in length
#     if FIPS_i in ['01','02','03','04','05','06','07','08','09']:
#         state_harm_i['GEOID'] = state_harm_i['GEOID'].map(
#         lambda x: f'{x:0>11}')
#     # Make GEOID a string and index row 
#     state_harm_i = state_harm_i.set_index('GEOID')
#     state_harm_i.index = state_harm_i.index.map(str)
#     # Make relevant columns floats
#     col_relevant = ['PRENO2', 'POSTNO2', 'PRENO2APR', 'POSTNO2APR', 'ALLNO2']
#     state_harm_i = state_harm_i[col_relevant]
#     for col in col_relevant:
#         state_harm_i[col] = state_harm_i.loc[:,col].astype(float)    
#     state_harm = state_harm.append(state_harm_i)    
# # ONLY FOR URBAN 
# harmonized_at = harmonized_urban.merge(state_harm, left_index=True, 
#     right_index=True)
# # Fraction of White population in tract
# white = (harmonized_at['AJWNE002']/harmonized_at['AJWBE001'])
# mostwhite = harmonized_at.iloc[np.where(white > 
#     np.nanpercentile(white, ptile_upper))]
# leastwhite = harmonized_at.iloc[np.where(white < 
#     np.nanpercentile(white, ptile_lower))]
# # Fraction of population in tract with some college to a Doctorate degree
# education = (harmonized_at.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
#     harmonized_at['AJYPE001'])
# mosteducated = harmonized_at.iloc[np.where(education > 
#     np.nanpercentile(education, ptile_upper))]
# leasteducated = harmonized_at.iloc[np.where(education < 
#     np.nanpercentile(education, ptile_lower))]
# # Fraction of non-Hispanic  or Latino population in tract
# ethnicity = (harmonized_at['AJWWE002']/harmonized_at['AJWWE001'])
# leastethnic = harmonized_at.iloc[np.where(ethnicity > 
#     np.nanpercentile(ethnicity, ptile_upper))]
# mostethnic = harmonized_at.iloc[np.where(ethnicity < 
#     np.nanpercentile(ethnicity, ptile_lower))]
# # Fraction of population without a vehicle
# vehicle = 1-harmonized_at['FracNoCar']
# mostvehicle = harmonized_at.iloc[np.where(vehicle > 
#     np.nanpercentile(vehicle, ptile_upper))]
# leastvehicle = harmonized_at.iloc[np.where(vehicle < 
#     np.nanpercentile(vehicle, ptile_lower))]
# # Highest vs. lowest income tracts
# income = harmonized_at['AJZAE001']
# highincome = harmonized_at.iloc[np.where(income > 
#     np.nanpercentile(income, ptile_upper))]
# lowincome = harmonized_at.iloc[np.where(income < 
#     np.nanpercentile(income, ptile_lower))]
# fig = plt.figure(figsize=(7,7))
# ax1 = plt.subplot2grid((3,2),(0,0))
# ax2 = plt.subplot2grid((3,2),(1,0))
# ax3 = plt.subplot2grid((3,2),(2,0))
# ax4 = plt.subplot2grid((3,2),(0,1))
# ax5 = plt.subplot2grid((3,2),(1,1))
# ax6 = plt.subplot2grid((3,2),(2,1))
# # 
# lefts = [highincome, mostwhite, leastethnic, mosteducated, mostvehicle]
# rights = [lowincome, leastwhite, mostethnic, leasteducated, leastvehicle]
# axes = [ax1, ax2, ax3, ax4, ax5]
# for ax, left, right in zip(axes, lefts, rights): 
#     bpl = ax.boxplot([left['ALLNO2'].values[~np.isnan(left['ALLNO2'].values)], 
#         left['PRENO2APR'].values[~np.isnan(left['PRENO2APR'].values)],
#         left['PRENO2_x'][~np.isnan(left['PRENO2_x'])]], positions=[1,2,3],
#         whis=[10,90], showfliers=False, patch_artist=True, showcaps=False)
#     bpr = ax.boxplot([right['ALLNO2'].values[~np.isnan(right['ALLNO2'].values)], 
#         right['PRENO2APR'].values[~np.isnan(right['PRENO2APR'].values)],
#         right['PRENO2_x'][~np.isnan(right['PRENO2_x'])]], positions=[5,6,7],
#         whis=[10,90], showfliers=False, patch_artist=True, showcaps=False)    
#     for bp in [bpl, bpr]:
#         for element in ['boxes']:
#             plt.setp(bp[element][0], color='#0095A8')
#             plt.setp(bp[element][1], color='darkgrey')    
#             plt.setp(bp[element][2], color='#FF7043')
#         for element in ['whiskers']:
#             plt.setp(bp[element][0], color='#0095A8', linewidth=2)
#             plt.setp(bp[element][1], color='#0095A8', linewidth=2)
#             plt.setp(bp[element][2], color='darkgrey', linewidth=2)
#             plt.setp(bp[element][3], color='darkgrey', linewidth=2)
#             plt.setp(bp[element][4], color='#FF7043', linewidth=2)
#             plt.setp(bp[element][5], color='#FF7043', linewidth=2)
#         for element in ['medians']:
#             plt.setp(bp[element], color='w', linewidth=2) 
#     ax.set_xlim([0,8])
#     ax.set_xticks([2,6])
#     ax.set_ylim([0.1e16,1.2e16])
# # Labels 
# ax1.set_title('(a) Household income', loc='left', fontsize=10)
# ax2.set_title('(b) Racial background', loc='left', fontsize=10)
# ax3.set_title('(c) Ethnic background', loc='left', fontsize=10)
# ax4.set_title('(d) Educational attainment',loc='left', fontsize=10)
# ax5.set_title('(e) Household vehicle ownership',loc='left', fontsize=10)
# ax1.set_xticklabels(['Highest income', 'Lowest income'], y=0.025)
# ax2.set_xticklabels(['Most White', 'Least White'], y=0.025)
# ax3.set_xticklabels(['Most Hispanic','Least Hispanic'], y=0.025)
# ax4.set_xticklabels(['Most educated','Least educated'], y=0.025)
# ax5.set_xticklabels(['Highest vehicle\nownership','Lowest vehicle\nownership'], y=0.025)
# for ax in [ax1, ax2, ax3, ax4, ax5]:
#     ax.yaxis.offsetText.set_visible(False)
#     ax.set_yticks([2.5e15, 5e15, 7.5e15, 10e15])
#     ax.set_yticklabels(['2.5','5.0','7.5','10'])    
#     ax.tick_params(axis='x', which='both', bottom=False)
# ax4.set_yticklabels([])
# ax5.set_yticklabels([])    
# ax1.set_ylabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', fontsize=10)
# ax2.set_ylabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', fontsize=10)
# ax3.set_ylabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', fontsize=10)
# # Create fake legend
# np.random.seed(10)
# collectn_1 = np.random.normal(100, 10, 200)    
# leg = ax6.boxplot([collectn_1, collectn_1, collectn_1], 
#     positions=[0.5,2,3.5], whis=[10,90], vert=0, showfliers=False, patch_artist=True, 
#     showcaps=False)
# for element in ['boxes']:
#     plt.setp(leg[element][0], color='#0095A8')
#     plt.setp(leg[element][1], color='darkgrey')    
#     plt.setp(leg[element][2], color='#FF7043')
# for element in ['whiskers']:
#     plt.setp(leg[element][0], color='#0095A8', linewidth=2)
#     plt.setp(leg[element][1], color='#0095A8', linewidth=2)
#     plt.setp(leg[element][2], color='darkgrey', linewidth=2)
#     plt.setp(leg[element][3], color='darkgrey', linewidth=2)
#     plt.setp(leg[element][4], color='#FF7043', linewidth=2)
#     plt.setp(leg[element][5], color='#FF7043', linewidth=2)
# for element in ['medians']:
#     plt.setp(leg[element], color='w', linewidth=2)
# ax6.set_xlim([65,115])
# ax6.set_ylim([-1,5])
# ax6.annotate('April 1 -\nJune 30, 2019', ha='right', xy=(86,3))
# ax6.annotate('March 13 -\nJune 13, 2019', ha='right', xy=(86,1.5))
# ax6.annotate('May 1, 2018 -\nDecember 31, 2019', ha='right', xy=(86,0.))
# ax6.axis('off')
# plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05)
# plt.savefig(DIR_FIGS+'figS1.png', dpi=300)

"""FIGURE 3"""
# harmonized = harmonized_urban 
# import numpy as np
# import scipy.stats as st
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from scipy import stats
# priroad_byincome = []
# byincome_cilow, byincome_cihigh = [], []
# priroad_byrace = []
# byrace_cilow, byrace_cihigh = [], []
# priroad_byeducation = []
# byeducation_cilow, byeducation_cihigh = [], []
# priroad_byethnicity = []
# byethnicity_cilow, byethnicity_cihigh = [], []
# priroad_byvehicle = []
# byvehicle_cilow, byvehicle_cihigh = [], []
# pri_density_mean = []
# dno2_mean = []
# ci_low, ci_high = [], []
# # Find tracts with a change in NO2 in the given range
# for ptilel, ptileu in zip(np.arange(0,100,10), np.arange(10,110,10)):    
#     dno2 = harmonized.loc[(harmonized['NO2_ABS']>
#         np.nanpercentile(harmonized['NO2_ABS'], ptilel)) & (
#         harmonized['NO2_ABS']<=np.nanpercentile(harmonized['NO2_ABS'], ptileu))]
#     # Find mean primary/secondary road density within 1 km and change in 
#     # NO2 for every percentile range
#     pri_density_mean.append(dno2['PrimaryWithin1'].mean())
#     dno2_mean.append(dno2['NO2_ABS'].mean())
#     ci = st.t.interval(0.95, len(dno2)-1, loc=np.mean(dno2['PrimaryWithin1']), 
#         scale=st.sem(dno2['PrimaryWithin1']))    
#     ci_low.append(ci[0])
#     ci_high.append(ci[1])    
# income = harmonized['AJZAE001']
# race = (harmonized['AJWNE002']/harmonized['AJWBE001'])
# education = (harmonized.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/harmonized['AJYPE001'])
# ethnicity = (harmonized['AJWWE002']/harmonized['AJWWE001'])
# vehicle = 1-harmonized['FracNoCar']
# for ptilel, ptileu in zip(np.arange(0,100,10), np.arange(10,110,10)):    
#     # Primary road density and confidence intervals by income
#     decile_income = harmonized.loc[(income>
#         np.nanpercentile(income, ptilel)) & (
#         income<=np.nanpercentile(income, ptileu))]
#     ci = st.t.interval(0.95, len(decile_income)-1, 
#         loc=np.mean(decile_income['PrimaryWithin1']), scale=st.sem(
#         decile_income['PrimaryWithin1']))
#     priroad_byincome.append(decile_income['PrimaryWithin1'].mean())        
#     byincome_cilow.append(ci[0])
#     byincome_cihigh.append(ci[1])
#     # By race            
#     decile_race = harmonized.loc[(race>
#         np.nanpercentile(race, ptilel)) & (
#         race<=np.nanpercentile(race, ptileu))] 
#     ci = st.t.interval(0.95, len(decile_race)-1, 
#         loc=np.mean(decile_race['PrimaryWithin1']), scale=st.sem(
#         decile_race['PrimaryWithin1']))
#     priroad_byrace.append(decile_race['PrimaryWithin1'].mean())    
#     byrace_cilow.append(ci[0])
#     byrace_cihigh.append(ci[1])            
#     # By education
#     decile_education = harmonized.loc[(education>
#         np.nanpercentile(education, ptilel)) & (
#         education<=np.nanpercentile(education, ptileu))]      
#     ci = st.t.interval(0.95, len(decile_education)-1, 
#         loc=np.mean(decile_education['PrimaryWithin1']), scale=st.sem(
#         decile_education['PrimaryWithin1']))
#     priroad_byeducation.append(decile_education['PrimaryWithin1'].mean())    
#     byeducation_cilow.append(ci[0])
#     byeducation_cihigh.append(ci[1])
#     # By ethnicity
#     decile_ethnicity = harmonized.loc[(ethnicity>
#         np.nanpercentile(ethnicity, ptilel)) & (
#         ethnicity<=np.nanpercentile(ethnicity, ptileu))]
#     ci = st.t.interval(0.95, len(decile_ethnicity)-1, 
#         loc=np.mean(decile_ethnicity['PrimaryWithin1']), scale=st.sem(
#         decile_ethnicity['PrimaryWithin1']))
#     priroad_byethnicity.append(decile_ethnicity['PrimaryWithin1'].mean())    
#     byethnicity_cilow.append(ci[0])
#     byethnicity_cihigh.append(ci[1])
#     # By vehicle ownership
#     decile_vehicle = harmonized.loc[(vehicle>
#         np.nanpercentile(vehicle, ptilel)) & (
#         vehicle<=np.nanpercentile(vehicle, ptileu))]    
#     ci = st.t.interval(0.95, len(decile_vehicle)-1, 
#         loc=np.mean(decile_vehicle['PrimaryWithin1']), scale=st.sem(
#         decile_vehicle['PrimaryWithin1']))
#     priroad_byvehicle.append(decile_vehicle['PrimaryWithin1'].mean())    
#     byvehicle_cilow.append(ci[0])
#     byvehicle_cihigh.append(ci[1])    
# # Initialize figure, axis
# fig = plt.figure(figsize=(8,5))
# ax1 = plt.subplot2grid((1,1),(0,0))
# color_density = 'k'
# color1 = '#0095A8'
# color2 = '#FF7043'
# color3 = '#5D69B1'
# color4 = '#CC3A8E'
# color5 = '#4daf4a'
# # Plotting
# ax1.plot(pri_density_mean, lw=2, marker='o', 
#     markeredgecolor=color_density, markerfacecolor='w', zorder=10,
#     color=color_density, clip_on=False)
# ax1.fill_between(np.arange(0,10,1),ci_low, ci_high, facecolor=color_density,
#     alpha=0.2)
# ax1.plot(priroad_byincome, ls='-', lw=2, color=color1, zorder=11)
# ax1.plot(priroad_byeducation, ls='-', lw=2, color=color2, zorder=11)
# ax1.plot(priroad_byrace, ls='-', lw=2, color=color3, zorder=11)
# ax1.plot(priroad_byethnicity, ls='-', lw=2, color=color4, zorder=11)
# ax1.plot(priroad_byvehicle, ls='-', lw=2, color=color5, zorder=11)
# # Legend
# ax1.text(4.5, 0.62, '$\mathregular{\Delta}$ NO$_{2}$', fontsize=12, 
#     color=color_density, ha='center', va='center')
# ax1.annotate('Smaller',xy=(4.9,0.62),xytext=(6,0.62), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color_density), 
#     color=color_density, fontsize=12)
# ax1.annotate('Larger',xy=(4.1,0.62),xytext=(2.6,0.62), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color_density), 
#     color=color_density, fontsize=12)
# ax1.text(4.5, 0.585, 'Income', fontsize=12, va='center',
#     color=color1, ha='center')
# ax1.annotate('Higher',xy=(4.9,0.585),xytext=(6,0.585), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color1), 
#     fontsize=12, color=color1)
# ax1.annotate('Lower',xy=(4.1,0.585),xytext=(2.6,0.585), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color1), 
#     fontsize=12, color=color1)
# ax1.text(4.5, 0.55, 'Education', fontsize=12, 
#     color=color2, va='center', ha='center')
# ax1.annotate('More',xy=(5.1,0.55),xytext=(6,0.55), va='center', 
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color2), fontsize=12,
#     color=color2)
# ax1.annotate('Less',xy=(3.9,0.55),xytext=(2.6,0.55), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color2), 
#     fontsize=12, color=color2)
# ax1.text(4.5, 0.515, 'White', fontsize=12, va='center',
#     color=color3, ha='center')
# ax1.annotate('More',xy=(4.85,0.515),xytext=(6,0.515), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color3), color=color3,
#     fontsize=12)
# ax1.annotate('Less',xy=(4.15,0.515),xytext=(2.6,0.515), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color3), fontsize=12, 
#     color=color3)
# ax1.text(4.5, 0.48, 'Hispanic', fontsize=12, 
#     color=color4, ha='center', va='center')
# ax1.annotate('Less',xy=(4.95,0.48),xytext=(6,0.48), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color4), fontsize=12, 
#     color=color4)
# ax1.annotate('More',xy=(4.05,0.48),xytext=(2.6,0.48), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color4), fontsize=12,
#     color=color4)
# ax1.text(4.5, 0.445, 'Vehicle ownership', fontsize=12, ha='center',
#     va='center', color=color5)
# ax1.annotate('More',xy=(5.5,0.445),xytext=(6,0.445), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color5), fontsize=12, 
#     color=color5)
# ax1.annotate('Less',xy=(3.55,0.445),xytext=(2.6,0.445), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color5), fontsize=12,
#     color=color5)
# # Aesthetics 
# ax1.set_xlim([0,9])
# ax1.set_xticks(np.arange(0,10,1))
# # ax1.set_xticklabels(['(0-10]','(10-20]','(20-30]','(30-40]','(40-50]',
# #     '(50-60]','(60-70]','(70-80]','(80-90]','(90-100]'])
# ax1.set_xticklabels(['First', 'Second', 'Third', 'Fourth', 'Fifth', 
#     'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth'], fontsize=10)
# ax1.set_ylim([0.05,0.65])
# ax1.set_xlabel('Decile', fontsize=12)
# ax1.set_ylabel('Primary road density [roads (1 km radius)$^{-1}$]', 
#     fontsize=12)
# plt.savefig(DIR_FIGS+'line_decile_primaryroads.png',dpi=600)
# ax1.text(1, 0.62, 'Largest NO$_{2}$ decrease', fontsize=12, 
#     color=color_density)
# ax1.text(1, 0.585, 'Lowest income', fontsize=12, 
#     color=color1)
# ax1.text(1, 0.55, 'Least educated', fontsize=12, 
#     color=color2)
# ax1.text(1, 0.515, 'Least White', fontsize=12, 
#     color=color3)
# ax1.text(1, 0.48, 'Most Hispanic', fontsize=12, 
#     color=color4)
# ax1.text(1, 0.445, 'Lowest vehicle ownership', fontsize=12, color=color5)
# ax1.text(8.8, 0.62, 'Smallest NO$_{2}$ decrease', fontsize=12, 
#     color=color_density, ha='right')
# ax1.text(8.8, 0.585, 'Largest income', fontsize=12, 
#     color=color1, ha='right')
# ax1.text(8.8, 0.55, 'Most educated', fontsize=12, 
#     color=color2, ha='right')
# ax1.text(8.8, 0.515, 'Most White', fontsize=12, 
#     color=color3, ha='right')
# ax1.text(8.8, 0.48, 'Least Hispanic', fontsize=12, 
#     color=color4, ha='right')
# ax1.text(8.8, 0.445, 'Highest vehicle ownership', fontsize=12, color=color5,
#     ha='right')

"""FIGURE 4"""
# from scipy import stats
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.rcParams['hatch.linewidth'] = 0.3     
# import matplotlib.patches as mpatches
# import netCDF4 as nc
# import numpy as np
# import matplotlib as mpl
# from matplotlib.gridspec import GridSpec
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.io import shapereader
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# # Initialize figure, axes
# fig = plt.figure(figsize=(12,9))
# proj = ccrs.PlateCarree(central_longitude=0.0)
# ax1 = plt.subplot2grid((3,3),(0,0), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# ax2 = plt.subplot2grid((3,3),(0,1), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# ax3 = plt.subplot2grid((3,3),(0,2), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# ax4 = plt.subplot2grid((3,3),(1,0), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# ax5 = plt.subplot2grid((3,3),(1,1), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# ax6 = plt.subplot2grid((3,3),(1,2), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# ax7 = plt.subplot2grid((3,3),(2,0), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# ax8 = plt.subplot2grid((3,3),(2,1), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# ax9 = plt.subplot2grid((3,3),(2,2), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# axes = [ax1, ax4 , ax7, ax2, ax5, ax8, ax3, ax6, ax9]
# fips = [['36','34'], ['13'], ['26']] # NY, GA, MI
# extents = [[-74.05, -73.8, 40.68, 40.9], # New York City
#     [-84.54, -84.2, 33.6, 33.95], # Atlanta
#     [-83.35,-82.87,42.2,42.5]] # Detroit
# citycodes = [['36047','36081','36061','36005','36085','36119', '34003', 
#     '34017','34031','36079','36087','36103','36059','34023','34025',
#     '34029','34035','34013','34039','34027','34037'	,'34019'],
#     # New York-Newark-Jersey City, NY-NJ-PA MSA
#     ['13121','13135','13067','13089','13063','13057','13117', '13151'],
#     # Atlanta-Sandy Springs-Alpharetta, GA MSA             
#     ['26163','26125','26099','26093','26147','26087']]    
#     # Detroit-Warren-Dearborn, MI MSA
# ticks = [np.linspace(-2e15,2e15,5),
#     np.linspace(-0.8e15,0.8e15,5),
#     np.linspace(-0.5e15,0.5e15,5)]
# ticklabels = [['%.2f'%x for x in np.linspace(-2,2,5)],
#     ['%.2f'%x for x in np.linspace(-0.8,0.8,5)],
#     ['%.2f'%x for x in np.linspace(-0.5,0.5,5)]]
# # Load U.S. counties
# reader = shapereader.Reader(DIR_GEO+'counties/tl_2019_us_county/'+
#     'tl_2019_us_county')
# counties = list(reader.geometries())
# counties = cfeature.ShapelyFeature(np.array(counties, dtype=object), proj)
# # Colormaps    
# cmapno2 = plt.get_cmap('coolwarm', 16)
# normno2 = [matplotlib.colors.Normalize(vmin=-2e15, vmax=2e15),
#     matplotlib.colors.Normalize(vmin=-0.8e15, vmax=0.8e15),
#     matplotlib.colors.Normalize(vmin=-0.5e15, vmax=0.5e15)]
# cmapincome = plt.get_cmap('Blues_r', 9)
# normincome = matplotlib.colors.Normalize(vmin=15000, vmax=60000)
# cmaprace = plt.get_cmap('Blues_r', 11)
# normrace = matplotlib.colors.Normalize(vmin=0, vmax=100)
# # Loop through cities of interest
# for i in np.arange(0,3,1):
#     axa = axes[i*3]
#     axb = axes[i*3+1]
#     axc = axes[i*3+2]
#     # Load demographics for each city 
#     harmonized_city = subset_harmonized_bycountyfips(harmonized, 
#         citycodes[i])    
#     # Find indicies of ~city
#     down = (np.abs(lat_dg-extents[i][2])).argmin()
#     up = (np.abs(lat_dg-extents[i][3])).argmin()
#     left = (np.abs(lng_dg-extents[i][0])).argmin()
#     right = (np.abs(lng_dg-extents[i][1])).argmin()
#     # Calculate ~city average change in NO2 during lockdowns
#     diff_cityavg = (np.nanmean(no2_post_dg[down:up+1,left:right+1])-
#         np.nanmean(no2_pre_dg[down:up+1,left:right+1]))
#     # Open shapefile for state; if more than one state, loop through 
#     # through 
#     sf = fips[i]
#     records, tracts = [], []
#     for sfi in sf:
#         shp = shapereader.Reader(DIR_GEO+'tigerline/'+
#             'tl_2019_%s_tract/tl_2019_%s_tract'%(sfi,sfi))
#         recordsi = shp.records()
#         tractsi = shp.geometries()
#         recordsi = list(recordsi)
#         tractsi = list(tractsi)
#         tracts.append(tractsi)
#         records.append(recordsi)
#     # Find records and tracts in city 
#     geoids_records = [x.attributes['GEOID'] for x in np.hstack(records)]
#     geoids_records = np.where(np.in1d(np.array(geoids_records), 
#         harmonized_city.index)==True)[0]
#     # Slice records and tracts for only entries in city
#     records = list(np.hstack(records)[geoids_records])
#     tracts = list(np.hstack(tracts)[geoids_records])
#     # Loop through shapefiles in city
#     for geoid in harmonized_city.index:
#         where_geoid = np.where(np.array([x.attributes['GEOID'] for x in 
#             records])==geoid)[0][0]
#         tract = tracts[where_geoid]
#         # Find demographic data/TROPOMI data in tract
#         harmonized_tract = harmonized_city.loc[harmonized_city.index.isin(
#             [geoid])]
#         # Income in tract
#         income_tract = harmonized_tract['AJZAE001'].values[0]
#         if np.isnan(income_tract)==True:
#             axb.add_geometries([tract], proj, hatch='\\\\\\\\\\\\\\', 
#                 edgecolor='k', linewidth=0, facecolor='None', alpha=1.,
#                 zorder=10)            
#         else:
#             axb.add_geometries([tract], proj, facecolor=cmapincome(normincome(
#                 income_tract)), edgecolor='None', alpha=1., zorder=10)         
#         # Race (percent white) in tract
#         white_tract = (harmonized_tract['AJWNE002'].values[0]/
#             harmonized_tract['AJWBE001'].values[0])*100.
#         if np.isnan(white_tract)==True:
#             axc.add_geometries([tract], proj, hatch='\\\\\\\\\\\\\\', 
#                 edgecolor='k', linewidth=0, facecolor='None', alpha=1., 
#                 zorder=10)          
#         else:            
#             axc.add_geometries([tract], proj, facecolor=cmaprace(normrace(
#                 white_tract)), edgecolor='None', alpha=1., zorder=10)      
#     # Plot oversampled NO2
#     mba = axa.pcolormesh(lng_dg[left:right+1], lat_dg[down:up+1], 
#         (no2_post_dg[down:up+1,left:right+1]-
#         no2_pre_dg[down:up+1,left:right+1])-diff_cityavg, 
#         cmap=cmapno2, norm=normno2[i], transform=proj)        
#     # Add colorbars
#     # Delta NO2
#     divider = make_axes_locatable(axa)
#     cax = divider.append_axes('right', size='5%', pad=0.1, 
#         axes_class=plt.Axes)
#     cbar = fig.colorbar(mba, cax=cax, orientation='vertical', ticks=ticks[i],
#         extend='both')    
#     cax.yaxis.offsetText.set_visible(False)
#     cbar.ax.set_yticklabels(ticklabels[i])
#     # Income
#     if i == 2: 
#         divider = make_axes_locatable(axb)
#         cax = divider.append_axes('right', size='5%', pad=0.1, 
#             axes_class=plt.Axes)
#         mpl.colorbar.ColorbarBase(cax, cmap=cmapincome, norm=normincome, 
#             spacing='proportional', orientation='vertical', extend='both')      
#         # Percent white
#         divider = make_axes_locatable(axc)
#         cax = divider.append_axes('right', size='5%', pad=0.1, 
#             axes_class=plt.Axes)
#         mpl.colorbar.ColorbarBase(cax, cmap=cmaprace, norm=normrace, 
#             spacing='proportional', orientation='vertical', extend='neither')  
#     # Add roads
#     for sfi in sf:
#         shp = shapereader.Reader(DIR_GEO+
#             'tigerline/roads/tl_2019_%s_prisecroads/'%sfi+
#             'tl_2019_%s_prisecroads'%sfi)   
#         roads_records = list(shp.records())
#         roads = list(shp.geometries())
#         # Select only interstates
#         roads_rttyp = [x.attributes['RTTYP'] for x in roads_records]
#         where_interstate = np.where((np.array(roads_rttyp)=='I') |
#                                     (np.array(roads_rttyp)=='U'))[0]
#         roads_i = []
#         roads_i += [roads[x] for x in where_interstate]
#         roads = cfeature.ShapelyFeature(roads_i, proj)
#         axa.add_feature(roads, facecolor='None', edgecolor='k', 
#             zorder=16, lw=0.75)    
#     for ax in [axa, axb, axc]:
#         ax.set_extent(extents[i], proj)
#         ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='10m',
#             facecolor='none', name='coastline', lw=0.5, 
#             edgecolor='k'), zorder=15)
#         ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='10m',
#             facecolor='grey', name='lakes', lw=0.5, 
#             edgecolor='k'), zorder=15)
#         ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='10m',
#             facecolor='grey', name='ocean', lw=0.5, 
#             edgecolor='k'), zorder=15)        
#         ax.add_feature(counties, facecolor='None', lw=0.5, edgecolor='k', 
#             zorder=11)    
# # Add axis titles 
# ax1.set_title('(a) New York', loc='left')
# ax2.set_title('(b) Atlanta', loc='left')
# ax3.set_title('(c) Detroit', loc='left')
# ax4.set_title('(d)', loc='left')
# ax5.set_title('(e)', loc='left')
# ax6.set_title('(f)', loc='left')
# ax7.set_title('(g)', loc='left')
# ax8.set_title('(h)', loc='left')
# ax9.set_title('(i)', loc='left')
# # Add axis labels
# ax1.text(-0.1, 0.5, '($\Delta$ NO$_{2}$ - $\Delta$ NO$_{2\mathregular{,'+
#     ' city}}$)/10$^\mathregular{15}$\n[molec cm$^{\mathregular{-2}}$]', 
#     ha='center', rotation='vertical', rotation_mode='anchor',
#     transform=ax1.transAxes, fontsize=12)
# ax4.text(-0.1, 0.5, 'Income [$]', ha='center', rotation='vertical', 
#     rotation_mode='anchor', transform=ax4.transAxes, fontsize=12)
# ax7.text(-0.1, 0.5, 'White [%]', ha='center', rotation='vertical', 
#     rotation_mode='anchor', transform=ax7.transAxes, fontsize=12)
# for ax in axes: 
#     ax.set_aspect('auto')
#     ax.outline_patch.set_zorder(20)
# plt.subplots_adjust(left=0.07, top=0.95, bottom=0.05, wspace=0.3)
# plt.savefig('/Users/ghkerr/Desktop/fig4.pdf', dpi=600)

""" FIGURE S3"""
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# # Colors
# color_white = '#0095A8'
# color_non = '#FF7043'
# os = 1.2
# # Initialize figure
# fig = plt.figure(figsize=(11,9))
# # All
# ax1 = plt.subplot2grid((5,3),(0,0))
# ax2 = plt.subplot2grid((5,3),(1,0))
# ax3 = plt.subplot2grid((5,3),(2,0))
# ax4 = plt.subplot2grid((5,3),(3,0))
# ax5 = plt.subplot2grid((5,3),(4,0))
# # Rural
# ax6 = plt.subplot2grid((5,3),(0,1))
# ax7 = plt.subplot2grid((5,3),(1,1))
# ax8 = plt.subplot2grid((5,3),(2,1))
# ax9 = plt.subplot2grid((5,3),(3,1))
# ax10 = plt.subplot2grid((5,3),(4,1))
# # Urban
# ax11 = plt.subplot2grid((5,3),(0,2))
# ax12 = plt.subplot2grid((5,3),(1,2))
# ax13 = plt.subplot2grid((5,3),(2,2))
# ax14 = plt.subplot2grid((5,3),(3,2))
# ax15 = plt.subplot2grid((5,3),(4,2))
# # Loop through sets of axies (sets correspond to all, rural, and urban)
# for harm, axes in zip([harmonized, harmonized_rural, harmonized_urban],
#     [[ax1,ax2,ax3,ax4,ax5],[ax6,ax7,ax8,ax9,ax10],[ax11,ax12,ax13,ax14,ax15]]):
#     axa = axes[0]
#     axb = axes[1] 
#     axc = axes[2]
#     axd = axes[3]
#     axe = axes[4]
#     i, yticks = 0, [] 
#     for ptilepair in [(5,95),(10,90),(20,80),(25,75)]:
#         up = ptilepair[-1]
#         down = ptilepair[0]
    
#         # Median household income
#         mostwealthy = harm.loc[harm['AJZAE001'] > 
#             np.nanpercentile(harm['AJZAE001'], up)]
#         leastwealthy = harm.loc[harm['AJZAE001'] < 
#             np.nanpercentile(harm['AJZAE001'], down)]
#         axa.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, 
#               zorder=12, clip_on=False)
#         axa.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, 
#               zorder=12, clip_on=False)
#         axa.plot((np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()]), 
#             np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+
#             np.abs(np.diff([mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
#             [i-os,i-os], color='k', ls='-', zorder=10)    
#         axa.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, 
#             zorder=12, clip_on=False)
#         axa.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, 
#               zorder=12, clip_on=False)
#         axa.plot((np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()]), 
#             np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
#             np.abs(np.diff([mostwealthy['POSTNO2'].mean(), leastwealthy['POSTNO2'].mean()]))[0]), 
#             [i+os,i+os], color='k', ls='--', zorder=10)
#         # Racial background
#         frac_white = (harm['AJWNE002']/harm['AJWBE001'])
#         mostwhite = harm.iloc[np.where(frac_white > 
#             np.nanpercentile(frac_white, up))]
#         leastwhite = harm.iloc[np.where(frac_white < 
#             np.nanpercentile(frac_white, down))]
#         axb.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, 
#               zorder=12, clip_on=False)
#         axb.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, 
#               zorder=12, clip_on=False)
#         axb.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()]), 
#             np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()])+
#             np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))[0]), 
#             [i-os,i-os], color='k', ls='-', zorder=10)    
#         axb.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, 
#             zorder=12, clip_on=False)
#         axb.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, 
#             zorder=12, clip_on=False)
#         axb.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
#             np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
#             np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))[0]), 
#             [i+os,i+os], color='k', ls='--', zorder=10)
#         # Ethnic background
#         frac_hispanic = 1-(harm['AJWWE003']/harm['AJWWE001'])
#         most = harm.iloc[np.where(frac_hispanic > 
#             np.nanpercentile(frac_hispanic, up))]
#         least = harm.iloc[np.where(frac_hispanic < 
#             np.nanpercentile(frac_hispanic, down))]
#         axc.plot(most['PRENO2'].mean(), i-os, 'o', color=color_white, 
#             zorder=12, clip_on=False)
#         axc.plot(least['PRENO2'].mean(), i-os, 'o', color=color_non, 
#             zorder=12, clip_on=False)
#         axc.plot((np.min([most['PRENO2'].mean(),least['PRENO2'].mean()]), 
#             np.min([most['PRENO2'].mean(),least['PRENO2'].mean()])+
#             np.abs(np.diff([most['PRENO2'].mean(), least['PRENO2'].mean()]))[0]), 
#             [i-os,i-os], color='k', ls='-', zorder=10)    
#         axc.plot(most['POSTNO2'].mean(), i+os, 'o', color=color_white, 
#             zorder=12, clip_on=False)
#         axc.plot(least['POSTNO2'].mean(), i+os, 'o', color=color_non, 
#             zorder=12, clip_on=False)
#         axc.plot((np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()]), 
#             np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()])+
#             np.abs(np.diff([most['POSTNO2'].mean(), least['POSTNO2'].mean()]))[0]), 
#             [i+os,i+os], color='k', ls='--', zorder=10)
#         # Educational attainment 
#         frac_educated = (harm.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
#             harm['AJYPE001'])
#         mosteducated = harm.iloc[np.where(frac_educated > 
#             np.nanpercentile(frac_educated, up))]
#         leasteducated = harm.iloc[np.where(frac_educated < 
#             np.nanpercentile(frac_educated, down))]
#         axd.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, 
#             zorder=12, clip_on=False)
#         axd.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, 
#             zorder=12, clip_on=False)
#         axd.plot((np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()]), 
#             np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()])+
#             np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
#             [i-os,i-os], color='k', ls='-', zorder=10)    
#         axd.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, 
#             zorder=12, clip_on=False)
#         axd.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, 
#             zorder=12, clip_on=False)
#         axd.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
#             np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
#             np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
#             [i+os,i+os], color='k', ls='--', zorder=10)
#         # Vehicle ownership
#         frac = 1-harm['FracNoCar']
#         most = harm.iloc[np.where(frac > 
#             np.nanpercentile(frac, up))]
#         least = harm.iloc[np.where(frac < 
#             np.nanpercentile(frac, down))]
#         axe.plot(most['PRENO2'].mean(), i-os, 'o', color=color_white, 
#             zorder=12, clip_on=False)
#         axe.plot(least['PRENO2'].mean(), i-os, 'o', color=color_non, 
#             zorder=12, clip_on=False)
#         axe.plot((np.min([most['PRENO2'].mean(),least['PRENO2'].mean()]), 
#             np.min([most['PRENO2'].mean(),least['PRENO2'].mean()])+
#             np.abs(np.diff([most['PRENO2'].mean(), least['PRENO2'].mean()]))[0]), 
#             [i-os,i-os], color='k', ls='-', clip_on=False, zorder=10)    
#         axe.plot(most['POSTNO2'].mean(), i+os, 'o', color=color_white, 
#             zorder=12, clip_on=False)
#         axe.plot(least['POSTNO2'].mean(), i+os, 'o', color=color_non, 
#             zorder=12, clip_on=False)
#         axe.plot((np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()]), 
#             np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()])+
#             np.abs(np.diff([most['POSTNO2'].mean(), least['POSTNO2'].mean()]))[0]), 
#             [i+os,i+os], color='k', ls='--', clip_on=False, zorder=10)
#         yticks.append(np.nanmean([i]))   
#         i = i+7
# # Aesthetics
# ax1.set_title('(a) All', loc='left', fontsize=12)
# ax2.set_title('(d)', loc='left', fontsize=12)
# ax3.set_title('(g)', loc='left', fontsize=12)
# ax4.set_title('(j)', loc='left', fontsize=12)
# ax5.set_title('(m)', loc='left', fontsize=12)        
# ax6.set_title('(b) Rural', loc='left', fontsize=12)
# ax7.set_title('(e)', loc='left', fontsize=12)
# ax8.set_title('(h)', loc='left', fontsize=12)
# ax9.set_title('(k)', loc='left', fontsize=12)
# ax10.set_title('(n)', loc='left', fontsize=12)
# ax11.set_title('(c) Urban', loc='left', fontsize=12)
# ax12.set_title('(f)', loc='left', fontsize=12)
# ax13.set_title('(i)', loc='left', fontsize=12)
# ax14.set_title('(l)', loc='left', fontsize=12)
# ax15.set_title('(o)', loc='left', fontsize=12)

# ax1.set_ylabel('Income', fontsize=12)
# ax2.set_ylabel('Racial background', fontsize=12)
# ax3.set_ylabel('Ethnic background', fontsize=12)
# ax4.set_ylabel('Educational\nattainment', fontsize=12)
# ax5.set_ylabel('Vehicle\nownership', fontsize=12)
# # Axes ticks        
# for ax in [ax1,ax2,ax3,ax4,ax5]:
#     ax.set_xlim([0.95e15,4.1e15])
#     ax.set_xticks([1e15,2e15,3e15,4e15])
#     ax.set_xticklabels([])
# ax5.set_xticklabels(['1','2','3','4'])
# for ax in [ax6,ax7,ax8,ax9,ax10]:
#     ax.set_xlim([0.95e15,2.55e15])
#     ax.set_xticks([1e15,1.5e15,2e15,2.5e15])
#     ax.set_xticklabels([])
# ax10.set_xticklabels(['1','1.5','2','2.5'])
# for ax in [ax11,ax12,ax13,ax14,ax15]:
#     ax.set_xlim([1.95e15,8.05e15])    
#     ax.set_xticks([2e15,4e15,6e15,8e15])
#     ax.set_xticklabels([])    
# ax15.set_xticklabels(['2','4','6','8'])
# for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15]:    
#     ax.set_yticks(yticks)
#     ax.set_yticklabels([])
#     # Hide spines
#     for side in ['right', 'left', 'top', 'bottom']:
#         ax.spines[side].set_visible(False)
#     ax.tick_params(axis='x', colors='darkgrey')
#     ax.xaxis.offsetText.set_visible(False)
#     ax.grid(axis='x', zorder=0, which='both', color='darkgrey')
#     ax.invert_yaxis()
#     ax.tick_params(axis='y', left=False)
# for ax in [ax5,ax10,ax15]:
#     ax.set_xlabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', x=0.35, labelpad=4,
#         color='darkgrey')
# ticks = ['5/95', r'$\bf{{10/90}}$', '20/80', '25/75']   
# for ax in [ax1,ax2,ax3,ax4,ax5]:    
#     ax.set_yticklabels(ticks)
# plt.subplots_adjust(left=0.1, hspace=0.4, wspace=0.4, right=0.8)
# # Custom legend
# custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
#     Line2D([0], [0], marker='o', color=color_non, lw=0)]
# ax11.legend(custom_lines, ['Highest income', 'Lowest income'], 
#     bbox_to_anchor=(1.5, 0.25), loc=8, ncol=1, frameon=False)
# custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
#     Line2D([0], [0], marker='o', color=color_non, lw=0)]
# ax12.legend(custom_lines, ['Most white', 'Least white'], 
#     bbox_to_anchor=(1.46, 0.25), loc=8, ncol=1, frameon=False)
# custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
#     Line2D([0], [0], marker='o', color=color_non, lw=0)]
# ax13.legend(custom_lines, ['Least Hispanic', 'Most Hispanic'], 
#     bbox_to_anchor=(1.49, 0.25), loc=8, ncol=1, frameon=False)
# custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
#     Line2D([0], [0], marker='o', color=color_non, lw=0)]
# ax14.legend(custom_lines, ['Most educated', 'Least educated'], 
#     bbox_to_anchor=(1.5, 0.25), loc=8, ncol=1, frameon=False)
# custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
#     Line2D([0], [0], marker='o', color=color_non, lw=0),  
#     Line2D([0], [0], color='k', lw=1.0),
#     Line2D([0], [0], color='k', ls='--', lw=1)]
# ax15.legend(custom_lines, ['Highest ownership', 'Lowest ownership', 
#     'Baseline', 'Lockdown'], bbox_to_anchor=(1.54, -0.07), loc=8, ncol=1, 
#     frameon=False)
# plt.savefig('/Users/ghkerr/Desktop/figS3.png', dpi=500)
# plt.show()
