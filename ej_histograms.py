#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate plots for Kerr, G.H, Goldberg, D.L., and Anenberg, S.C. (2020). 
COVID-19 lockdowns reveal pronounced disparities in nitrogen dioxide pollution
levels. Created on Sun Jul 12 15:21:23 2020

@author: ghkerr
"""
# For running locally
DIR = '/Users/ghkerr/GW/'
DIR_TROPOMI = DIR+'data/'
DIR_CENSUS = DIR+'data/census_no2_harmonzied/'
DIR_GEO = DIR+'data/geography/'
DIR_HARM = DIR+'data/census_no2_harmonzied/'
DIR_FIGS = DIR+'tropomi_ej/figs/'
DIR_CEMS = DIR+'data/emissions/CEMS/'
DIR_TYPEFACE = '/Users/ghkerr/Library/Fonts/'
# # For running on EPS-Curiosity
# DIR = '/mnt/scratch1/gaige/data/tropomi_ej/'
# DIR_TROPOMI = DIR
# DIR_GEO = DIR
# DIR_HARM = DIR
# DIR_FIGS = DIR
# DIR_TYPEFACE = DIR

# Constants
ptile_upper = 90
ptile_lower = 10
missingdata = 'darkgrey'

# Load custom font
import sys
if 'mpl' not in sys.modules:
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(
            fname=DIR_TYPEFACE+'cmunbmr.ttf')
    matplotlib.rcParams['font.family'] = prop.get_name()
    prop = matplotlib.font_manager.FontProperties(
        fname=DIR_TYPEFACE+'cmunbbx.ttf')
    matplotlib.rcParams['mathtext.bf'] = prop.get_name()
    prop = matplotlib.font_manager.FontProperties(
        fname=DIR_TYPEFACE+'cmunbmr.ttf')
    matplotlib.rcParams['mathtext.it'] = prop.get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False

def ratio_significance(xpre, ypre, xpost, ypost):
    """Calculate the significance of changes in ratios of NO2 in the least 
    white/least income/least educated tracts to the most white/most income/
    most educated tracts. This is a two-step process, where the first step 
    calculates the standard error of the ratio and the second step calculates 
    a Z-score from the change in the ratio. Note that for the Z-score to be 
    significance at 95% confidence level, Z > 1.96.

    
    Parameters
    ----------
    xpre : float
        Baseline NO2 for the least white/lower income/least educated
    ypre : float 
        Baseline NO2 for the most white/highest income/most educated
    xpost : float
        Lockdown NO2 for the least white/lower income/least educated
    ypost : float
        Lockdown NO2 for the most white/highest income/most educated    
    
    Returns
    -------
    Z : float
        Z-score statistic for the two ratios and their respective standard
        errors
    """
    import numpy as np
    xpre = np.log1p(xpre)
    ypre = np.log1p(ypre)
    xpost = np.log1p(xpost)
    ypost = np.log1p(ypost)
    # Mean of the baseline NO2 distributions
    xmeanpre = np.nanmean(xpre)
    ymeanpre = np.nanmean(ypre)
    # Standard error for the baseline NO2 distributions; standard error is
    # the sample standard deviation divided by the root of the number of 
    # samples
    xsepre = np.nanstd(xpre)/np.sqrt(len(xpre))
    ysepre = np.nanstd(ypre)/np.sqrt(len(ypre))
    # Same as above but for the lockdown NO2 distributions 
    xmeanpost = np.nanmean(xpost)
    ymeanpost = np.nanmean(ypost)
    xsepost = np.nanstd(xpost)/np.sqrt(len(xpost))
    ysepost = np.nanstd(ypost)/np.sqrt(len(ypost))
    # Ratios
    ratio_pre = xmeanpre/ymeanpre
    ratio_post = xmeanpost/ymeanpost
    # Calculate the standard error of the ratio: the standard error of a 
    # ratio (where the numerator is not a subset of the denominator) is 
    # approximated as
    # SE(hX/hY) = (1/hY)*sqrt(SE(X)^2 + (hX^2/hY^2*(SE(Y)^2)))
    # where hY and hX stand for "hat Y" and "hat X" - the mean values 
    # from https://www2.census.gov/programs-surveys/acs/tech_docs/accuracy/
    # 2018_ACS_Accuracy_Document_Worked_Examples.pdf?
    se_pre = (1/xmeanpre)*np.sqrt((xsepre**2)+
        ((ymeanpre**2/xmeanpre**2)*(ysepre**2)))
    se_post = (1/xmeanpost)*np.sqrt((xsepost**2)+
        ((ymeanpost**2/xmeanpost**2)*(ysepost**2)))
    # The Z statistic from the two estimates of ratios (ratioPre and 
    # ratioPost) is given by 
    # Z = (ratioPost - ratioPre)/sqrt(sePost^2 + sePre^2), 
    # where sePost and sePre are the standard errors of ratios (calculated 
    # above); from https://www2.census.gov/programs-surveys/acs/tech_docs/
    # accuracy/2018_ACS_Accuracy_Document_Worked_Examples.pdf?    
    Z = (ratio_post-ratio_pre)/np.sqrt(se_post**2+se_pre**2)
    return Z

def fig1(harmonized, harmonized_urban):
    """
    """
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from decimal import Decimal
    import matplotlib as mpl
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    from cartopy.io import shapereader
    from shapely.geometry import Polygon
    from scipy.stats import ks_2samp
    def rect_from_bound(xmin, xmax, ymin, ymax):
        """Returns list of (x,y)'s for a rectangle"""
        xs = [xmax, xmin, xmin, xmax, xmax]
        ys = [ymax, ymax, ymin, ymin, ymax]
        return [(x, y) for x, y in zip(xs, ys)]
    proj = ccrs.PlateCarree(central_longitude=0.0)
    fig = plt.figure(figsize=(9,6))
    # Maps
    ax1 = plt.subplot2grid((5,2),(0,0), projection=ccrs.PlateCarree(
        central_longitude=0.), rowspan=2)
    ax2 = plt.subplot2grid((5,2),(0,1), projection=ccrs.PlateCarree(
        central_longitude=0.), rowspan=2)
    # Bar charts
    ax3 = plt.subplot2grid((5,2),(2,0))
    ax4 = plt.subplot2grid((5,2),(2,1))
    ax5 = plt.subplot2grid((5,2),(3,0))
    ax6 = plt.subplot2grid((5,2),(3,1))
    ax7 = plt.subplot2grid((5,2),(4,0))
    ax8 = plt.subplot2grid((5,2),(4,1))
    # Create discrete colormaps
    cmapbase = plt.get_cmap('YlGnBu', 12)
    normbase = matplotlib.colors.Normalize(vmin=0e15, vmax=6e15)
    cmaplock = plt.get_cmap('coolwarm', 12)
    normlock = matplotlib.colors.Normalize(vmin=-2e15, vmax=2e15)
    for FIPS_i in FIPS: 
        print(FIPS_i)
        # Tigerline shapefile for state
        shp = shapereader.Reader(DIR_GEO+
            'tigerline/tl_2019_%s_tract/tl_2019_%s_tract.shp'%(FIPS_i, FIPS_i))
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
                ax1.add_geometries([tract], proj, facecolor='darkgrey', 
                    edgecolor='darkgrey', alpha=1., linewidth=0.1, 
                    rasterized=False)
                ax2.add_geometries([tract], proj, facecolor='darkgrey', 
                    edgecolor='darkgrey', alpha=1., linewidth=0.1, 
                    rasterized=False)
            else:
                ax1.add_geometries([tract], proj, facecolor=cmapbase(
                    normbase(baseline)), edgecolor=cmapbase(
                    normbase(baseline)), alpha=1., linewidth=0.1, 
                    rasterized=False)
                ax2.add_geometries([tract], proj, facecolor=cmaplock(
                    normlock(lockdown)), edgecolor=cmaplock(
                    normlock(lockdown)), alpha=1., linewidth=0.1, 
                    rasterized=False)
    shpfilename = shapereader.natural_earth('10m', 'cultural', 
        'admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries = reader.records()   
    # Get location of U.S.
    usa = [x.attributes['ADM0_A3'] for x in countries]
    usa = np.where(np.array(usa)=='USA')[0][0]
    usa = list(reader.geometries())[usa]   
    # Load lakes 
    lakes = shapereader.natural_earth('10m', 'physical', 'lakes')
    lakes_reader = shpreader.Reader(lakes)
    lakes = lakes_reader.records()   
    lake_names = [x.attributes['name'] for x in lakes]
    great_lakes = np.where((np.array(lake_names)=='Lake Superior') |
        (np.array(lake_names)=='Lake Michigan') | 
        (np.array(lake_names)=='Lake Huron') |
        (np.array(lake_names)=='Lake Erie') |
        (np.array(lake_names)=='Lake Ontario'))[0]
    great_lakes = np.array(list(lakes_reader.geometries()), 
        dtype=object)[great_lakes]
    # Projection
    st_proj = ccrs.Mercator(central_longitude=0.0)
    pad1 = 1.  #padding, degrees unit
    exts = [usa[0].bounds[0] - pad1, usa[0].bounds[2] + pad1, 
            usa[0].bounds[1] - pad1, usa[0].bounds[3] + pad1]
    # Make a mask polygon by polygon's difference operation; base polygon 
    # is a rectangle, another polygon is simplified switzerland
    msk = Polygon(rect_from_bound(*exts)).difference(usa[0].simplify(0.01))
    # Project geometry to the projection used by stamen
    msk_stm  = st_proj.project_geometry(msk, proj)
    # Add borders, set map extent, etc. 
    for ax in [ax1, ax2]:
        ax.set_extent([-125,-66.5, 24.5, 49.48], proj)       
        ax.add_geometries(usa, crs=proj, lw=0.25, facecolor='None', 
            edgecolor='k', zorder=15)
        # Plot the mask
        ax.add_geometries(msk_stm, st_proj, zorder=12, facecolor='w', 
            edgecolor='None', alpha=1.)             
        # Add states
        ax.add_feature(cfeature.NaturalEarthFeature('cultural', 
            'admin_1_states_provinces_lines', '10m', edgecolor='k', lw=0.25, 
            facecolor='none'))
        # Add Great Lakes
        ax.add_geometries(great_lakes, crs=ccrs.PlateCarree(), 
            facecolor='w', lw=0.25, edgecolor='black', alpha=1., zorder=17)
        ax.background_patch.set_visible(False)
        ax.outline_patch.set_visible(False)
    # # # # Bar charts for demographics
    harmonized = harmonized_urban
    colors = ['#0095A8','darkgrey','#FF7043']
    # Largest, smallest and median gains
    increaseno2 = harmonized.loc[harmonized['NO2_ABS']>
        np.nanpercentile(harmonized['NO2_ABS'], ptile_upper)]
    decreaseno2 = harmonized.loc[harmonized['NO2_ABS']<
        np.nanpercentile(harmonized['NO2_ABS'], ptile_lower)]
    # # # # Change in NO2
    # Urban 
    ax3.barh([2,1,0], [decreaseno2['NO2_ABS'].mean(), harmonized['NO2_ABS'].mean(), 
        increaseno2['NO2_ABS'].mean()], color=colors[0])
    print('Largest gains for NO2 = %.2E'%Decimal(decreaseno2['NO2_ABS'].mean()))
    ax3.text(decreaseno2['NO2_ABS'].mean()+2e13, 2, '-3.5', color='black', 
        va='center')
    print('Mean gains for NO2 = %.2E'%Decimal(harmonized['NO2_ABS'].mean()))
    ax3.text(harmonized['NO2_ABS'].mean()+2e13, 1, '-1.0', color='black', 
        va='center')
    print('Smallest gains for NO2 = %.2E'%Decimal(increaseno2['NO2_ABS'].mean()))
    ax3.text(increaseno2['NO2_ABS'].mean()+2e13, 0, '0.07', color='black', 
        va='center')
    ks_no2 = ks_2samp(increaseno2['NO2_ABS'], decreaseno2['NO2_ABS'])
    print('K-S results for NO2', ks_no2)
    
    # # # # Income
    ax5.barh([2,1,0], [decreaseno2['AJZAE001'].mean(), 
        harmonized['AJZAE001'].mean(), increaseno2['AJZAE001'].mean()], 
        color=colors[0])
    ax5.text(60000, 2, ' %d'%(decreaseno2['AJZAE001'].mean()), color='black', 
        va='center')
    ax5.text(60000, 1, ' %d'%(harmonized['AJZAE001'].mean()), color='black', 
        va='center')
    ax5.text(60000, 0, ' %d'%(increaseno2['AJZAE001'].mean()), color='black', 
        va='center')
    ks_income = ks_2samp(increaseno2['AJZAE001'], decreaseno2['AJZAE001'])
    print('K-S results for income', ks_income)

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
        ax7.barh(2, data, color=color, left=left)
        ax7.text(left+0.01, 2, '%d'%(np.round(data,2)*100), color='black', 
            va='center')
        left += data
        i = i+1
    # Mean demographics
    left, i = 0, 0
    for data, color in zip([(harmonized['AJWNE002']/
        harmonized['AJWBE001']).mean(),
        (harmonized['AJWNE003']/harmonized['AJWBE001']).mean(),      
        ((harmonized['AJWNE004']+harmonized['AJWNE005']+
          harmonized['AJWNE006']+harmonized['AJWNE007']+
        harmonized['AJWNE008'])/harmonized['AJWBE001']).mean()], colors):             
        ax7.barh(1, data, color=color, left=left)
        ax7.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='black', 
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
        ax7.barh(0, data, color=color, left=left)
        ax7.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='black', 
            va='center')    
        ax7.text(left+0.01, -1, labels[i], color=colors[i], va='center',
            fontweight='bold')
        left += data       
        i = i+1
    # Check which "Other" group changed the most
    # American Indian and Alaska Native alone
    print('American Indian and Alaska Native alone: increase/decrease')
    print((increaseno2['AJWNE004']/increaseno2['AJWBE001']).mean())
    print((decreaseno2['AJWNE004']/decreaseno2['AJWBE001']).mean())
    # Asian alone
    print('Asian alone: increase/decrease')
    print((increaseno2['AJWNE005']/increaseno2['AJWBE001']).mean())
    print((decreaseno2['AJWNE005']/decreaseno2['AJWBE001']).mean())
    # Native Hawaiian and Other Pacific Islander alone
    print('Native Hawaiian and Other Pacific Islander alone: increase/decrease')
    print((increaseno2['AJWNE006']/increaseno2['AJWBE001']).mean())
    print((decreaseno2['AJWNE006']/decreaseno2['AJWBE001']).mean())
    # Some other race alone
    print('Some other race alone: increase/decrease')
    print((increaseno2['AJWNE007']/increaseno2['AJWBE001']).mean())
    print((decreaseno2['AJWNE007']/decreaseno2['AJWBE001']).mean())
    # Two or more races
    print('Two or more races: increase/decrease')    
    print((increaseno2['AJWNE008']/increaseno2['AJWBE001']).mean())
    print((decreaseno2['AJWNE008']/decreaseno2['AJWBE001']).mean())
    ks_white = ks_2samp((increaseno2['AJWNE002']/increaseno2['AJWBE001']).values, 
        (decreaseno2['AJWNE002']/decreaseno2['AJWBE001']).values)   
    print('K-S results for (d) race/white', ks_white)
    ks_black = ks_2samp((increaseno2['AJWNE003']/increaseno2['AJWBE001']).values, 
        (decreaseno2['AJWNE003']/decreaseno2['AJWBE001']).values)   
    print('K-S results for race/black', ks_black)
    ks_other = ks_2samp(
        ((increaseno2['AJWNE004']+increaseno2['AJWNE005']+
          increaseno2['AJWNE006']+increaseno2['AJWNE007']+
        increaseno2['AJWNE008'])/increaseno2['AJWBE001']).values,        
        ((decreaseno2['AJWNE004']+decreaseno2['AJWNE005']+
          decreaseno2['AJWNE006']+decreaseno2['AJWNE007']+
        decreaseno2['AJWNE008'])/decreaseno2['AJWBE001']).values)   
    print('K-S results for race/other', ks_other)
        
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
    # Mean demographics
    left, i = 0, 0
    for data, color in zip([(harmonized['AJWWE003']/
        harmonized['AJWWE001']).mean(),
        (harmonized['AJWWE002']/harmonized['AJWWE001']).mean()], colors):             
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
    ks_nonhis = ks_2samp((increaseno2['AJWWE002']/increaseno2['AJWWE001']).values, 
        (decreaseno2['AJWWE002']/decreaseno2['AJWWE001']).values)   
    print('K-S results for ethnicity/non-Hispanic', ks_nonhis)
    ks_his = ks_2samp((increaseno2['AJWWE003']/increaseno2['AJWWE001']).values, 
        (decreaseno2['AJWWE003']/decreaseno2['AJWWE001']).values)   
    print('K-S results for ethnicity/Hispanic', ks_his)     
        
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
        ax6.barh(2, data, color=color, left=left)
        ax6.text(left+0.01, 2, '%d'%(np.round(data,2)*100), 
            color='black', va='center') 
        left += data
        i = i+1    
    # Mean demographics
    left = 0
    i = 0
    for data, color in zip([
        (harmonized.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        harmonized['AJYPE001']).mean(),
        (harmonized.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        harmonized['AJYPE001']).mean(),
        (harmonized.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        harmonized['AJYPE001']).mean()], colors):             
        ax6.barh(1, data, color=color, left=left)
        ax6.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='black', 
            va='center') 
        left += data
        i = i+1        
    # Smallest gains
    left = 0
    i = 0
    labels = ['High school', 'College', 'Graduate']
    for data, color in zip([
        (increaseno2.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        increaseno2['AJYPE001']).mean(),
        (increaseno2.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        increaseno2['AJYPE001']).mean(),
        (increaseno2.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        increaseno2['AJYPE001']).mean()], colors):             
        ax6.barh(0, data, color=color, left=left)
        ax6.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='black', 
            va='center')
        if i==2:
            ax6.text(0.82, -1, labels[i], color=colors[i], va='center',
                fontweight='bold', fontsize=10)         
        else:   
            ax6.text(left, -1, labels[i], color=colors[i], va='center',
                fontweight='bold', fontsize=10) 
        left += data
        i = i+1  
    ks_high = ks_2samp(
        (increaseno2.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        increaseno2['AJYPE001']).values, 
        (decreaseno2.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        decreaseno2['AJYPE001']).values) 
    print('K-S results for education/high school', ks_high)
    ks_college = ks_2samp(
        (increaseno2.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        increaseno2['AJYPE001']).values, 
        (decreaseno2.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        decreaseno2['AJYPE001']).values)
    print('K-S results for education/college', ks_college)
    ks_grad = ks_2samp(
        (increaseno2.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        increaseno2['AJYPE001']).values, 
        (decreaseno2.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        decreaseno2['AJYPE001']).values) 
    print('K-S results for education/graduate', ks_grad)    
    
    # # # # Vehicle ownership
    # Largest gains
    left = 0
    i = 0
    labels = ['None', 'One or more']
    for data, color in zip([decreaseno2['FracNoCar'].mean(),
        (1-decreaseno2['FracNoCar'].mean())], colors):             
        ax8.barh(2, data, color=color, left=left)
        ax8.text(left+0.01, 2, '%d'%(np.round(data,2)*100),
            color='black', va='center') 
        left += data
        i = i+1        
    # Mean demographics
    left = 0
    i = 0
    for data, color in zip([harmonized['FracNoCar'].mean(),
        (1-harmonized['FracNoCar'].mean())], colors):                        
        ax8.barh(1, data, color=color, left=left)
        ax8.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='black', 
            va='center') 
        left += data
        i = i+1          
    # Smallest gains
    left = 0    
    i = 0
    for data, color in zip([increaseno2['FracNoCar'].mean(),
        (1-increaseno2['FracNoCar'].mean())], colors):             
        ax8.barh(0, data, color=color, left=left)
        ax8.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='black', 
            va='center') 
        ax8.text((left*1.5), -1, labels[i], color=colors[i], va='center',
            fontweight='bold', fontsize=10)              
        left += data       
        i = i+1      
    ks_nocar = ks_2samp(increaseno2['FracNoCar'].values, 
        decreaseno2['FracNoCar'].values)
    print('K-S results for vehicle ownership/no car', ks_nocar)
    ks_car = ks_2samp(1-increaseno2['FracNoCar'].values, 
        1-decreaseno2['FracNoCar'].values)
    print('K-S results for vehicle ownership/car', ks_car)
        
    # Aesthetics    
    ax3.set_xlim([-3.5e15, 1e14])
    ax3.set_xticks([])    
    ax5.set_xlim([60000,74000])
    ax5.set_xticks([])
    for ax in [ax4,ax6,ax7,ax8]:
        ax.set_xlim([0,1.])
        ax.set_xticks([])
        ax.set_xticklabels([])
    for ax in [ax3,ax4,ax5,ax6,ax7,ax8]:
        ax.set_ylim([-0.5,2.5])
        ax.set_yticks([0,1,2])
        ax.set_yticklabels([])
    for ax in [ax3,ax4,ax5,ax6,ax7,ax8]:
        for pos in ['right','top','bottom']:
            ax.spines[pos].set_visible(False)
            ax.spines[pos].set_visible(False)    
    for ax in [ax3,ax5,ax7]:
        ax.set_yticklabels(['Smallest drops', 'Average', 'Largest drops'], 
            fontsize=10)
    # Axis titles
    ax1.set_title('(a) NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', loc='left', 
        fontsize=10)
    ax2.set_title('(b) $\Delta\:$NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', 
        loc='left', fontsize=10)
    ax3.set_title('(c) $\mathregular{\Delta}$ NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', 
        loc='left', fontsize=10)
    ax5.set_title('(e) Median household income [$]', loc='left', fontsize=10)
    ax7.set_title('(g) Racial background [%]', loc='left', fontsize=10)
    ax4.set_title('(d) Ethnic background [%]', loc='left', fontsize=10)
    ax6.set_title('(f) Educational attainment [%]',loc='left', fontsize=10)
    ax8.set_title('(h) Household vehicle ownership [%]',loc='left', fontsize=10)
    # Adjust subplots
    plt.subplots_adjust(left=0.12, top=0.95, bottom=0.06, right=0.98, 
        hspace=0.75)
    # Add Colorbars
    caxbase = fig.add_axes([ax1.get_position().x0, 
        ax1.get_position().y0+0.02, 
        (ax1.get_position().x1-ax1.get_position().x0)/2.7, 0.02])
    cb = mpl.colorbar.ColorbarBase(caxbase, cmap=cmapbase, norm=normbase, 
        spacing='proportional', orientation='horizontal', extend='max')
    caxbase.xaxis.offsetText.set_visible(False)
    caxbase = fig.add_axes([ax2.get_position().x0, 
        ax2.get_position().y0+0.02, 
        (ax2.get_position().x1-ax2.get_position().x0)/2.7, 
        0.02])
    cb = mpl.colorbar.ColorbarBase(caxbase, cmap=cmaplock, norm=normlock, 
        spacing='proportional', orientation='horizontal', extend='both',
        ticks=[-2e15,-1e15,0,1e15,2e15])
    caxbase.xaxis.offsetText.set_visible(False)
    plt.savefig(DIR_FIGS+'fig1_revised.png', dpi=1000)
    return 

def fig2(harmonized, harmonized_rural, harmonized_urban):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import ks_2samp
    # # # # 
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
    fig = plt.figure(figsize=(12,7))
    ax1 = plt.subplot2grid((2,3),(0,0),rowspan=2)
    ax2 = plt.subplot2grid((2,3),(0,1),rowspan=2)
    ax3 = plt.subplot2grid((2,3),(0,2),rowspan=2)
    i = 0 
    yticks = []
    # For all tracts
    frac_white = ((harmonized['AJWNE002'])/harmonized['AJWBE001'])
    mostwhite = harmonized.iloc[np.where(frac_white > 
        np.nanpercentile(frac_white, ptile_upper))]
    leastwhite = harmonized.iloc[np.where(frac_white < 
        np.nanpercentile(frac_white, ptile_lower))]
    ratio_pre = (leastwhite['PRENO2'].mean()/mostwhite['PRENO2'].mean())    
    ratio_post = (leastwhite['POSTNO2'].mean()/mostwhite['POSTNO2'].mean())
    print('Disparities in all tracts for race:')
    print('Baseline %.3f'%(ratio_pre))
    print('Lockdown %.3f'%(ratio_post))
    print('\n')
    # Compute the Kolmogorov-Smirnov statistic on 2 samples (in our case, the 
    # NO2 distributions from the most X versus least X tracts). The 2 sample 
    # K-S test returns a D statistic and a p-value corresponding to the D 
    # statistic. 
    # The D statistic is the absolute max distance (supremum) between the 
    # CDFs of the two samples. The closer this number is to 0 the more likely 
    # it is that the two samples were drawn from the same distribution. 
    # The p value is the evidence against a null hypothesis. The smaller the
    # p-value, the stronger the evidence that you should reject the null
    # hypothesis. For example, a p value of 0.0254 is 2.54%. This means there
    # is a 2.54% chance the results could be random (i.e. happened by chance).
    # Denote points with X if not statistically significant at alpha = 0.05
    pval_preno2 = ks_2samp(mostwhite['PRENO2'], leastwhite['PRENO2']).pvalue
    pval_postno2 = ks_2samp(mostwhite['POSTNO2'], leastwhite['POSTNO2']).pvalue
    # 
    Z = ratio_significance(leastwhite['PRENO2'], mostwhite['PRENO2'], 
        leastwhite['POSTNO2'], mostwhite['POSTNO2'])
    if pval_preno2 >= 0.05:
        ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'x', color=color_white, 
            zorder=12)
        ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'x', color=color_non, 
            zorder=12)    
    else:
        ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, 
            zorder=12)
        ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, 
            zorder=12)
        ax1.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'
            ].mean()]), np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'
            ].mean()])+np.abs(np.diff([mostwhite['PRENO2'].mean(), 
            leastwhite['PRENO2'].mean()]))[0]), [i-os,i-os], color='k', ls='-', 
            zorder=10)
    if pval_postno2 >= 0.05:
        ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'x', color=color_white, 
            zorder=12)
        ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'x', color=color_non, 
            zorder=12)
    else:
        ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, 
            zorder=12)
        ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, 
            zorder=12)        
        ax1.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'
            ].mean()]), np.min([mostwhite['POSTNO2'].mean(),leastwhite[
            'POSTNO2'].mean()])+np.abs(np.diff([mostwhite['POSTNO2'].mean(), 
            leastwhite['POSTNO2'].mean()]))[0]), [i+os,i+os], color='k', 
            ls='--', zorder=10)   
    if np.abs(Z) < 1.96:
        ax1.fill([0,9e15,9e15,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
            alpha=0.15, facecolor='grey', edgecolor='grey')                                                
    yticks.append(np.nanmean([i]))
    i = i+7    
    # For rural tracts
    frac_white = ((harmonized_rural['AJWNE002'])/harmonized_rural['AJWBE001'])
    mostwhite = harmonized_rural.iloc[np.where(frac_white > 
        np.nanpercentile(frac_white, ptile_upper))]
    leastwhite = harmonized_rural.iloc[np.where(frac_white < 
        np.nanpercentile(frac_white, ptile_lower))]
    pval_preno2 = ks_2samp(mostwhite['PRENO2'], leastwhite['PRENO2']).pvalue
    pval_postno2 = ks_2samp(mostwhite['POSTNO2'], leastwhite['POSTNO2']).pvalue
    Z = ratio_significance(leastwhite['PRENO2'], mostwhite['PRENO2'], 
        leastwhite['POSTNO2'], mostwhite['POSTNO2'])    
    if pval_preno2 >= 0.05:
        ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'x', color=color_white, 
            zorder=12)
        ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'x', color=color_non, 
            zorder=12)    
    else:
        ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, 
            zorder=12)
        ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, 
            zorder=12)
        ax1.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'
            ].mean()]), np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'
            ].mean()])+np.abs(np.diff([mostwhite['PRENO2'].mean(), 
            leastwhite['PRENO2'].mean()]))[0]), [i-os,i-os], color='k', ls='-', 
            zorder=10)
    if pval_postno2 >= 0.05:
        ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'x', color=color_white, 
            zorder=12)
        ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'x', color=color_non, 
            zorder=12)
    else:
        ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, 
            zorder=12)
        ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, 
            zorder=12)        
        ax1.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'
            ].mean()]), np.min([mostwhite['POSTNO2'].mean(),leastwhite[
            'POSTNO2'].mean()])+np.abs(np.diff([mostwhite['POSTNO2'].mean(), 
            leastwhite['POSTNO2'].mean()]))[0]), [i+os,i+os], color='k', 
            ls='--', zorder=10)   
    if np.abs(Z) < 1.96:
        ax1.fill([0,9e15,9e15,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
            alpha=0.15, facecolor='grey', edgecolor='grey')                                                
    yticks.append(np.nanmean([i]))
    i = i+7  
    # For urban tracts
    frac_white = ((harmonized_urban['AJWNE002'])/harmonized_urban['AJWBE001'])
    mostwhite = harmonized_urban.iloc[np.where(frac_white > 
        np.nanpercentile(frac_white, ptile_upper))]
    leastwhite = harmonized_urban.iloc[np.where(frac_white < 
        np.nanpercentile(frac_white, ptile_lower))]
    pval_preno2 = ks_2samp(mostwhite['PRENO2'], leastwhite['PRENO2']).pvalue
    pval_postno2 = ks_2samp(mostwhite['POSTNO2'], leastwhite['POSTNO2']).pvalue
    Z = ratio_significance(leastwhite['PRENO2'], mostwhite['PRENO2'], 
        leastwhite['POSTNO2'], mostwhite['POSTNO2'])
    if pval_preno2 >= 0.05:
        ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'x', color=color_white, 
            zorder=12)
        ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'x', color=color_non, 
            zorder=12)    
    else:
        ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, 
            zorder=12)
        ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, 
            zorder=12)
        ax1.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'
            ].mean()]), np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'
            ].mean()])+np.abs(np.diff([mostwhite['PRENO2'].mean(), 
            leastwhite['PRENO2'].mean()]))[0]), [i-os,i-os], color='k', ls='-', 
            zorder=10)
    if pval_postno2 >= 0.05:
        ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'x', color=color_white, 
            zorder=12)
        ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'x', color=color_non, 
            zorder=12)
    else:
        ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, 
            zorder=12)
        ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, 
            zorder=12)        
        ax1.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'
            ].mean()]), np.min([mostwhite['POSTNO2'].mean(),leastwhite[
            'POSTNO2'].mean()])+np.abs(np.diff([mostwhite['POSTNO2'].mean(), 
            leastwhite['POSTNO2'].mean()]))[0]), [i+os,i+os], color='k', 
            ls='--', zorder=10)
    if np.abs(Z) < 1.96:
        ax1.fill([0,9e15,9e15,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
            alpha=0.15, facecolor='grey', edgecolor='grey')                                                
    yticks.append(np.nanmean([i]))
    i = i+7        
    ratio_pre = []
    ratio_post = []
    citynames = [r'$\bf{All}$', r'$\bf{Rural}$', r'$\bf{Urban}$',
        'New York', 'Los Angeles', 'Chicago', 'Dallas', 'Houston', 
        'Washington', 'Miami', 'Philadelphia', 'Atlanta', 'Phoenix', 
        'Boston', 'San Francisco', 'Riverside', 'Detroit', 'Seattle']   
    for city in [newyork, losangeles, chicago, dallas, houston, washington,
        miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
        riverside, detroit, seattle]:
        # Subset for city
        harmonized_city = tropomi_census_utils.subset_harmonized_bycountyfips(
            harmonized, city)
        # Find particular demographic for each city
        frac_white = (harmonized_city['AJWNE002']/harmonized_city['AJWBE001'])            
        mostwhite = harmonized_city.iloc[np.where(frac_white > 
            np.nanpercentile(frac_white, ptile_upper))]
        leastwhite = harmonized_city.iloc[np.where(frac_white < 
            np.nanpercentile(frac_white, ptile_lower))]
        # Calculate significance
        pval_preno2 = ks_2samp(mostwhite['PRENO2'], 
            leastwhite['PRENO2']).pvalue
        pval_postno2 = ks_2samp(mostwhite['POSTNO2'], 
            leastwhite['POSTNO2']).pvalue
        Z = ratio_significance(leastwhite['PRENO2'], mostwhite['PRENO2'], 
            leastwhite['POSTNO2'], mostwhite['POSTNO2'])
        if pval_preno2 >= 0.05:
            ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'x', color=color_white, 
                zorder=12)
            ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'x', color=color_non, 
                zorder=12)    
        else:
            ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, 
                zorder=12)
            ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, 
                zorder=12)
            ax1.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'
                ].mean()]), np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'
                ].mean()])+np.abs(np.diff([mostwhite['PRENO2'].mean(), 
                leastwhite['PRENO2'].mean()]))[0]), [i-os,i-os], color='k', ls='-', 
                zorder=10)
        if pval_postno2 >= 0.05:
            ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'x', color=color_white, 
                zorder=12)
            ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'x', color=color_non, 
                zorder=12)
        else:
            ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, 
                zorder=12)
            ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, 
                zorder=12)        
            ax1.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'
                ].mean()]), np.min([mostwhite['POSTNO2'].mean(),leastwhite[
                'POSTNO2'].mean()])+np.abs(np.diff([mostwhite['POSTNO2'].mean(), 
                leastwhite['POSTNO2'].mean()]))[0]), [i+os,i+os], color='k', 
                ls='--', zorder=10)
        if np.abs(Z) < 1.96:
            ax1.fill([0,9e15,9e15,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
                alpha=0.15, facecolor='grey', edgecolor='grey')
        yticks.append(np.nanmean([i]))
        ratio_pre.append(leastwhite['PRENO2'].mean()/mostwhite['PRENO2'].mean())    
        ratio_post.append(leastwhite['POSTNO2'].mean()/mostwhite['POSTNO2'].mean())
        i = i+7
    print('Disparities in 15 largest MSAs for race:')
    print('Baseline %.3f +/- %.3f'%(np.nanmean(ratio_pre),np.nanstd(ratio_pre)))
    print('Lockdown %.3f +/- %.3f'%(np.nanmean(ratio_post),np.nanstd(ratio_post)))
    print('\n')
    # Aesthetics 
    ax1.set_xlim([-0.5e15,10e15])
    ax1.set_xticks(np.arange(0e15,10e15,1e15))
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(citynames)
    ax1.tick_params(axis='y', left=False, length=0)
    ax1.set_xlabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', x=0.26, 
        color='darkgrey')
    ax1.tick_params(axis='x', colors='darkgrey')
    ax1.xaxis.offsetText.set_visible(False)
    for side in ['right', 'left', 'top', 'bottom']:
        ax1.spines[side].set_visible(False)
    ax1.grid(axis='x', zorder=0, color='darkgrey')
    ax1.invert_yaxis()
    # # # Most versus least wealthy
    i = 0 
    yticks = []
    mostwealthy = harmonized.loc[harmonized['AJZAE001'] > 
        np.nanpercentile(harmonized['AJZAE001'], 90)]
    leastwealthy = harmonized.loc[harmonized['AJZAE001'] < 
        np.nanpercentile(harmonized['AJZAE001'], 10)]
    ratio_pre = (leastwealthy['PRENO2'].mean()/mostwealthy['PRENO2'].mean())
    ratio_post = (leastwealthy['POSTNO2'].mean()/mostwealthy['POSTNO2'].mean())
    print('Disparities in all tracts for income:')
    print('Baseline %.3f'%(ratio_pre))
    print('Lockdown %.3f'%(ratio_post))
    print('\n')
    pval_preno2 = ks_2samp(mostwealthy['PRENO2'], 
        leastwealthy['PRENO2']).pvalue
    pval_postno2 = ks_2samp(mostwealthy['POSTNO2'], 
        leastwealthy['POSTNO2']).pvalue
    Z = ratio_significance(leastwealthy['PRENO2'], mostwealthy['PRENO2'], 
        leastwealthy['POSTNO2'], mostwealthy['POSTNO2'])    
    if pval_preno2 >= 0.05:
        ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'x', color=color_white, 
            zorder=12)
        ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'x', color=color_non, 
            zorder=12)   
    else:        
        ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
        ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
        ax2.plot((np.min([mostwealthy['PRENO2'].mean(),
            leastwealthy['PRENO2'].mean()]), np.min([mostwealthy[
            'PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+np.abs(np.diff(
            [mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
            [i-os,i-os], color='k', ls='-', zorder=10)
    if pval_postno2 >= 0.05:
        ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'x', color=color_white, zorder=12)
        ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'x', color=color_non, zorder=12)
    else: 
        ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
        ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
        ax2.plot((np.min([mostwealthy['POSTNO2'].mean(),
            leastwealthy['POSTNO2'].mean()]), np.min([
            mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
            np.abs(np.diff([mostwealthy['POSTNO2'].mean(), 
            leastwealthy['POSTNO2'].mean()]))[0]), [i+os,i+os], color='k', 
            ls='--', zorder=10)    
    if np.abs(Z) < 1.96:
        ax2.fill([0,9e15,9e15,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
            alpha=0.15, facecolor='grey', edgecolor='grey')                
    yticks.append(np.nanmean([i]))
    i = i+7    
    mostwealthy = harmonized_rural.loc[harmonized_rural['AJZAE001'] > 
        np.nanpercentile(harmonized_rural['AJZAE001'], 90)]
    leastwealthy = harmonized_rural.loc[harmonized_rural['AJZAE001'] < 
        np.nanpercentile(harmonized_rural['AJZAE001'], 10)]
    ratio_pre = (leastwealthy['PRENO2'].mean()/mostwealthy['PRENO2'].mean())
    ratio_post = (leastwealthy['POSTNO2'].mean()/mostwealthy['POSTNO2'].mean())
    print('Disparities in all rural for income:')
    print('Baseline %.3f'%(ratio_pre))
    print('Lockdown %.3f'%(ratio_post))
    print('\n')
    pval_preno2 = ks_2samp(mostwealthy['PRENO2'], 
        leastwealthy['PRENO2']).pvalue
    pval_postno2 = ks_2samp(mostwealthy['POSTNO2'], 
        leastwealthy['POSTNO2']).pvalue
    Z = ratio_significance(leastwealthy['PRENO2'], mostwealthy['PRENO2'], 
        leastwealthy['POSTNO2'], mostwealthy['POSTNO2'])      
    if pval_preno2 >= 0.05:
        ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'x', 
            color=color_white, zorder=12)
        ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'x', color=color_non, 
            zorder=12)           
    else:        
        ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
        ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
        ax2.plot((np.min([mostwealthy['PRENO2'].mean(),
            leastwealthy['PRENO2'].mean()]), np.min([mostwealthy[
            'PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+np.abs(np.diff(
            [mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
            [i-os,i-os], color='k', ls='-', zorder=10)  
    if pval_postno2 >= 0.05:
        ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'x', color=color_white, zorder=12)
        ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'x', color=color_non, zorder=12)
    else: 
        ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
        ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
        ax2.plot((np.min([mostwealthy['POSTNO2'].mean(),
            leastwealthy['POSTNO2'].mean()]), np.min([
            mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
            np.abs(np.diff([mostwealthy['POSTNO2'].mean(), 
            leastwealthy['POSTNO2'].mean()]))[0]), [i+os,i+os], color='k', 
            ls='--', zorder=10)
    if np.abs(Z) < 1.96:
        ax2.fill([0,9e15,9e15,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
            alpha=0.15, facecolor='grey', edgecolor='grey')                              
    yticks.append(np.nanmean([i]))
    i = i+7  
    mostwealthy = harmonized_urban.loc[harmonized_urban['AJZAE001'] > 
        np.nanpercentile(harmonized_urban['AJZAE001'], 90)]
    leastwealthy = harmonized_urban.loc[harmonized_urban['AJZAE001'] < 
        np.nanpercentile(harmonized_urban['AJZAE001'], 10)]
    ratio_pre = (leastwealthy['PRENO2'].mean()/mostwealthy['PRENO2'].mean())
    ratio_post = (leastwealthy['POSTNO2'].mean()/mostwealthy['POSTNO2'].mean())
    print('Disparities in urban tracts for income:')
    print('Baseline %.3f'%(ratio_pre))
    print('Lockdown %.3f'%(ratio_post))
    print('\n')
    pval_preno2 = ks_2samp(mostwealthy['PRENO2'], 
        leastwealthy['PRENO2']).pvalue
    pval_postno2 = ks_2samp(mostwealthy['POSTNO2'], 
        leastwealthy['POSTNO2']).pvalue
    Z = ratio_significance(leastwealthy['PRENO2'], mostwealthy['PRENO2'], 
        leastwealthy['POSTNO2'], mostwealthy['POSTNO2'])          
    if pval_preno2 >= 0.05:
        ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'x', 
            color=color_white, zorder=12)
        ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'x', color=color_non, 
            zorder=12)           
    else:        
        ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
        ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
        ax2.plot((np.min([mostwealthy['PRENO2'].mean(),
            leastwealthy['PRENO2'].mean()]), np.min([mostwealthy[
            'PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+np.abs(np.diff(
            [mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
            [i-os,i-os], color='k', ls='-', zorder=10)  
    if pval_postno2 >= 0.05:
        ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'x', color=color_white, zorder=12)
        ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'x', color=color_non, zorder=12)
    else: 
        ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
        ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
        ax2.plot((np.min([mostwealthy['POSTNO2'].mean(),
            leastwealthy['POSTNO2'].mean()]), np.min([
            mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
            np.abs(np.diff([mostwealthy['POSTNO2'].mean(), 
            leastwealthy['POSTNO2'].mean()]))[0]), [i+os,i+os], color='k', 
            ls='--', zorder=10)
    if np.abs(Z) < 1.96:
        ax2.fill([0,9e15,9e15,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
            alpha=0.15, facecolor='grey', edgecolor='grey')                              
    yticks.append(np.nanmean([i]))
    i = i+7      
    ratio_pre = []
    ratio_post = []
    for city in [newyork, losangeles, chicago, dallas, houston, washington,
        miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
        riverside, detroit, seattle]:
        harmonized_city = tropomi_census_utils.subset_harmonized_bycountyfips(
            harmonized, city)
        mostwealthy = harmonized_city.loc[harmonized_city['AJZAE001'] > 
            np.nanpercentile(harmonized_city['AJZAE001'], 90)]
        leastwealthy = harmonized_city.loc[harmonized_city['AJZAE001'] < 
            np.nanpercentile(harmonized_city['AJZAE001'], 10)]
        # Calculate significance
        pval_preno2 = ks_2samp(mostwealthy['PRENO2'], 
            leastwealthy['PRENO2']).pvalue
        pval_postno2 = ks_2samp(mostwealthy['POSTNO2'], 
            leastwealthy['POSTNO2']).pvalue
        Z = ratio_significance(leastwealthy['PRENO2'], mostwealthy['PRENO2'], 
            leastwealthy['POSTNO2'], mostwealthy['POSTNO2'])          
        if pval_preno2 >= 0.05:
            ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'x', 
                color=color_white, zorder=12)
            ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'x', color=color_non, 
                zorder=12)           
        else:        
            ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=
                color_white, zorder=12)
            ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=
                color_non, zorder=12)
            ax2.plot((np.min([mostwealthy['PRENO2'].mean(),
                leastwealthy['PRENO2'].mean()]), np.min([mostwealthy[
                'PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+np.abs(np.diff(
                [mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
                [i-os,i-os], color='k', ls='-', zorder=10)  
        if pval_postno2 >= 0.05:
            ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'x', color=
                color_white, zorder=12)
            ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'x', color=
                color_non, zorder=12)
        else: 
            ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=
                color_white, zorder=12)
            ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=
                color_non, zorder=12)
            ax2.plot((np.min([mostwealthy['POSTNO2'].mean(),
                leastwealthy['POSTNO2'].mean()]), np.min([
                mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
                np.abs(np.diff([mostwealthy['POSTNO2'].mean(), 
                leastwealthy['POSTNO2'].mean()]))[0]), [i+os,i+os], color='k', 
                ls='--', zorder=10)
        if np.abs(Z) < 1.96:
            ax2.fill([0,9e15,9e15,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
                alpha=0.15, facecolor='grey', edgecolor='grey')                    
        yticks.append(np.nanmean([i]))
        ratio_pre.append(leastwealthy['PRENO2'].mean()/mostwealthy['PRENO2'].mean())
        ratio_post.append(leastwealthy['POSTNO2'].mean()/mostwealthy['POSTNO2'].mean())
        i = i+7
    print('Disparities in 15 largest MSAs for income:')
    print('Baseline %.3f +/- %.3f'%(np.nanmean(ratio_pre),np.nanstd(ratio_pre)))
    print('Lockdown %.3f +/- %.3f'%(np.nanmean(ratio_post),np.nanstd(ratio_post)))
    print('\n')
    # Aesthetics 
    ax2.set_xlim([-0.5e15,10e15])
    ax2.set_xticks(np.arange(0e15,10e15,1e15))
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([])
    ax2.tick_params(axis='y', left=False)
    ax2.set_xlabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', x=0.26, 
        color='darkgrey')
    ax2.tick_params(axis='x', colors='darkgrey')
    ax2.xaxis.offsetText.set_visible(False)
    for side in ['right', 'left', 'top', 'bottom']:
        ax2.spines[side].set_visible(False)
    ax2.grid(axis='x', zorder=0, color='darkgrey')
    ax2.invert_yaxis()
    # Most versus least educated
    i = 0 
    yticks = []
    frac_educated = (harmonized.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        harmonized['AJYPE001'])
    mosteducated = harmonized.iloc[np.where(frac_educated > 
        np.nanpercentile(frac_educated, 90))]
    leasteducated = harmonized.iloc[np.where(frac_educated < 
        np.nanpercentile(frac_educated, 10))]
    ratio_pre = (leasteducated['PRENO2'].mean()/mosteducated['PRENO2'].mean())    
    ratio_post = (leasteducated['POSTNO2'].mean()/mosteducated['POSTNO2'].mean())
    print('Disparities in all tracts for education:')
    print('Baseline %.3f'%(ratio_pre))
    print('Lockdown %.3f'%(ratio_post))
    print('\n')
    # Calculate significance
    pval_preno2 = ks_2samp(mosteducated['PRENO2'], 
        leasteducated['PRENO2']).pvalue
    pval_postno2 = ks_2samp(mosteducated['POSTNO2'], 
        leasteducated['POSTNO2']).pvalue
    Z = ratio_significance(leasteducated['PRENO2'], mosteducated['PRENO2'], 
        leasteducated['POSTNO2'], mosteducated['POSTNO2'])          
    if pval_preno2 >= 0.05:
        ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'x', color=color_white, 
            zorder=12)
        ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'x', color=color_non, 
            zorder=12)           
    else:        
        ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
        ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
        ax3.plot((np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()]), 
            np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()])+
            np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
            [i-os,i-os], color='k', ls='-', zorder=10)
    if pval_postno2 >= 0.05:
        ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'x', color=color_white, zorder=12)
        ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'x', color=color_non, zorder=12)
    else: 
        ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
        ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
        ax3.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
            np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
            np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
            [i+os,i+os], color='k', ls='--', zorder=10)
    if np.abs(Z) < 1.96:
        ax3.fill([0,9e15,9e15,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
            alpha=0.15, facecolor='grey', edgecolor='grey')            
    yticks.append(np.nanmean([i]))
    i = i+7    
    frac_educated = (harmonized_rural.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        harmonized_rural['AJYPE001'])
    mosteducated = harmonized_rural.iloc[np.where(frac_educated > 
        np.nanpercentile(frac_educated, 90))]
    leasteducated = harmonized_rural.iloc[np.where(frac_educated < 
        np.nanpercentile(frac_educated, 10))]
    ratio_pre = (leasteducated['PRENO2'].mean()/mosteducated['PRENO2'].mean())    
    ratio_post = (leasteducated['POSTNO2'].mean()/mosteducated['POSTNO2'].mean())
    print('Disparities in rural tracts for education:')
    print('Baseline %.3f'%(ratio_pre))
    print('Lockdown %.3f'%(ratio_post))
    print('\n')
    pval_preno2 = ks_2samp(mosteducated['PRENO2'], 
        leasteducated['PRENO2']).pvalue
    pval_postno2 = ks_2samp(mosteducated['POSTNO2'], 
        leasteducated['POSTNO2']).pvalue 
    Z = ratio_significance(leasteducated['PRENO2'], mosteducated['PRENO2'], 
        leasteducated['POSTNO2'], mosteducated['POSTNO2'])              
    if pval_preno2 >= 0.05:
        ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'x', color=color_white, 
            zorder=12)
        ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'x', color=color_non, 
            zorder=12)           
    else:        
        ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, 
            zorder=12)
        ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, 
            zorder=12)
        ax3.plot((np.min([mosteducated['PRENO2'].mean(),
            leasteducated['PRENO2'].mean()]), np.min([mosteducated['PRENO2'
            ].mean(),leasteducated['PRENO2'].mean()])+np.abs(np.diff([
            mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]
            ))[0]), [i-os,i-os], color='k', ls='-', zorder=10)
    if pval_postno2 >= 0.05:
        ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'x', color=color_white, 
            zorder=12)
        ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'x', color=color_non, 
            zorder=12)
    else: 
        ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, 
            zorder=12)
        ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, 
            zorder=12)
        ax3.plot((np.min([mosteducated['POSTNO2'].mean(), 
            leasteducated['POSTNO2'].mean()]), np.min([mosteducated['POSTNO2'
            ].mean(),leasteducated['POSTNO2'].mean()])+np.abs(np.diff([
            mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]
            ))[0]), [i+os,i+os], color='k', ls='--', zorder=10)    
    if np.abs(Z) < 1.96:
        ax3.fill([0,9e15,9e15,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
            alpha=0.15, facecolor='grey', edgecolor='grey')                
    yticks.append(np.nanmean([i]))
    i = i+7    
    frac_educated = (harmonized_urban.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        harmonized_urban['AJYPE001'])
    mosteducated = harmonized_urban.iloc[np.where(frac_educated > 
        np.nanpercentile(frac_educated, 90))]
    leasteducated = harmonized_urban.iloc[np.where(frac_educated < 
        np.nanpercentile(frac_educated, 10))]
    ratio_pre = (leasteducated['PRENO2'].mean()/mosteducated['PRENO2'].mean())    
    ratio_post = (leasteducated['POSTNO2'].mean()/mosteducated['POSTNO2'].mean())
    print('Disparities in urban tracts for education:')
    print('Baseline %.3f'%(ratio_pre))
    print('Lockdown %.3f'%(ratio_post))
    print('\n')
    pval_preno2 = ks_2samp(mosteducated['PRENO2'], 
        leasteducated['PRENO2']).pvalue
    pval_postno2 = ks_2samp(mosteducated['POSTNO2'], 
        leasteducated['POSTNO2']).pvalue 
    Z = ratio_significance(leasteducated['PRENO2'], mosteducated['PRENO2'], 
        leasteducated['POSTNO2'], mosteducated['POSTNO2'])              
    if pval_preno2 >= 0.05:
        ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'x', color=color_white, 
            zorder=12)
        ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'x', color=color_non, 
            zorder=12)           
    else:        
        ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, 
            zorder=12)
        ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, 
            zorder=12)
        ax3.plot((np.min([mosteducated['PRENO2'].mean(),
            leasteducated['PRENO2'].mean()]), np.min([mosteducated['PRENO2'
            ].mean(),leasteducated['PRENO2'].mean()])+np.abs(np.diff([
            mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]
            ))[0]), [i-os,i-os], color='k', ls='-', zorder=10)
    if pval_postno2 >= 0.05:
        ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'x', color=color_white, 
            zorder=12)
        ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'x', color=color_non, 
            zorder=12)
    else: 
        ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, 
            zorder=12)
        ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, 
            zorder=12)
        ax3.plot((np.min([mosteducated['POSTNO2'].mean(), 
            leasteducated['POSTNO2'].mean()]), np.min([mosteducated['POSTNO2'
            ].mean(),leasteducated['POSTNO2'].mean()])+np.abs(np.diff([
            mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]
            ))[0]), [i+os,i+os], color='k', ls='--', zorder=10)
    if np.abs(Z) < 1.96:
        ax3.fill([0,9e15,9e15,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
            alpha=0.15, facecolor='grey', edgecolor='grey')                    
    yticks.append(np.nanmean([i]))
    i = i+7    
    ratio_pre = []
    ratio_post = []
    for city in [newyork, losangeles, chicago, dallas, houston, washington,
        miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
        riverside, detroit, seattle]:
        harmonized_city = tropomi_census_utils.subset_harmonized_bycountyfips(
            harmonized, city)
        frac_educated = (harmonized_city.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
            harmonized_city['AJYPE001'])
        mosteducated = harmonized_city.iloc[np.where(frac_educated > 
            np.nanpercentile(frac_educated, 90))]
        leasteducated = harmonized_city.iloc[np.where(frac_educated < 
            np.nanpercentile(frac_educated, 10))]
        pval_preno2 = ks_2samp(mosteducated['PRENO2'], 
            leasteducated['PRENO2']).pvalue
        pval_postno2 = ks_2samp(mosteducated['POSTNO2'], 
            leasteducated['POSTNO2']).pvalue
        Z = ratio_significance(leasteducated['PRENO2'], mosteducated['PRENO2'], 
            leasteducated['POSTNO2'], mosteducated['POSTNO2'])
        if pval_preno2 >= 0.05:
            ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'x', color=color_white, 
                zorder=12)
            ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'x', color=color_non, 
                zorder=12)           
        else:        
            ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, 
                zorder=12)
            ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, 
                zorder=12)
            ax3.plot((np.min([mosteducated['PRENO2'].mean(),
                leasteducated['PRENO2'].mean()]), np.min([mosteducated['PRENO2'
                ].mean(),leasteducated['PRENO2'].mean()])+np.abs(np.diff([
                mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]
                ))[0]), [i-os,i-os], color='k', ls='-', zorder=10)
        if pval_postno2 >= 0.05:
            ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'x', color=color_white, 
                zorder=12)
            ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'x', color=color_non, 
                zorder=12)
        else: 
            ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, 
                zorder=12)
            ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, 
                zorder=12)
            ax3.plot((np.min([mosteducated['POSTNO2'].mean(), 
                leasteducated['POSTNO2'].mean()]), np.min([mosteducated['POSTNO2'
                ].mean(),leasteducated['POSTNO2'].mean()])+np.abs(np.diff([
                mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]
                ))[0]), [i+os,i+os], color='k', ls='--', zorder=10)  
        if np.abs(Z) < 1.96:
            ax3.fill([0,9e15,9e15,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
                alpha=0.15, facecolor='grey', edgecolor='grey')                        
        yticks.append(np.nanmean([i]))
        ratio_pre.append(leasteducated['PRENO2'].mean()/mosteducated['PRENO2'].mean())
        ratio_post.append(leasteducated['POSTNO2'].mean()/mosteducated['POSTNO2'].mean())    
        i = i+7
    print('Disparities in 15 largest MSAs for education:')
    print('Baseline %.3f +/- %.3f'%(np.nanmean(ratio_pre),np.nanstd(ratio_pre)))
    print('Lockdown %.3f +/- %.3f'%(np.nanmean(ratio_post),np.nanstd(ratio_post)))
    print('\n')
    ax3.set_xlim([-0.5e15,10e15])
    ax3.set_xticks(np.arange(0e15,10e15,1e15))
    ax3.set_yticks(yticks)
    ax3.set_yticklabels([])
    ax3.tick_params(axis='y', left=False)
    ax3.set_xlabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', x=0.26, 
        color='darkgrey')
    ax3.tick_params(axis='x', colors='darkgrey')
    ax3.xaxis.offsetText.set_visible(False)
    for side in ['right', 'left', 'top', 'bottom']:
        ax3.spines[side].set_visible(False)
    ax3.grid(axis='x', zorder=0, color='darkgrey')
    ax3.invert_yaxis()
    ax1.set_title('(a) Racial background', loc='left', fontsize=10)
    ax2.set_title('(b) Median household income', loc='left', fontsize=10)
    ax3.set_title('(c) Educational attainment', loc='left', fontsize=10)
    plt.subplots_adjust(wspace=0.05, left=0.09, top=0.95, bottom=0.17, 
        right=0.98)
    # Custom legends for different colored scatterpoints
    custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
        Line2D([0], [0], marker='o', color=color_non, lw=0)]
    ax1.legend(custom_lines, ['Most white', 'Least white'], 
        bbox_to_anchor=(0.48, -0.15), loc=8, ncol=2, fontsize=10, 
        frameon=False)
    custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
        Line2D([0], [0], marker='o', color=color_non, lw=0)]
    ax2.legend(custom_lines, ['Highest income', 'Lowest income'], 
        bbox_to_anchor=(0.48, -0.15), loc=8, ncol=2, fontsize=10, 
        frameon=False)
    custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
        Line2D([0], [0], marker='o', color=color_non, lw=0)]
    ax3.legend(custom_lines, ['Most educated', 'Least educated'], 
        bbox_to_anchor=(0.48, -0.15), loc=8, ncol=2, fontsize=10, 
        frameon=False)
    # Custom legend for baseline vs. lockdown bars
    ax1t = ax1.twinx()
    ax1t.axis('off')
    custom_lines = [Line2D([0], [0], color='k', marker='o', lw=2),
        Line2D([0], [0], color='k', marker='o', ls='--', lw=2)]
    ax1t.legend(custom_lines, ['Baseline', 'Lockdown'], ncol=2, loc=8, 
        bbox_to_anchor=(0.59, -0.2), numpoints=2, frameon=False, 
        handlelength=5)
    plt.savefig(DIR_FIGS+'fig2_revised.pdf', dpi=1000)
    plt.show()
    # Check ratios on ethnicity 
    frac = 1-(harmonized['AJWWE003']/harmonized['AJWWE001'])
    most = harmonized.iloc[np.where(frac > 
        np.nanpercentile(frac, ptile_upper))]
    least = harmonized.iloc[np.where(frac < 
        np.nanpercentile(frac, ptile_lower))]
    ratio_pre = (least['PRENO2'].mean()/most['PRENO2'].mean())    
    ratio_post = (least['POSTNO2'].mean()/most['POSTNO2'].mean())
    print('Disparities in all tracts for ethnicity:')
    print('Baseline %.3f'%(ratio_pre))
    print('Lockdown %.3f'%(ratio_post))
    print('\n')
    ratio_pre = []
    ratio_post = []
    for city in [newyork, losangeles, chicago, dallas, houston, washington,
        miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
        riverside, detroit, seattle]:
        harmonized_city = tropomi_census_utils.subset_harmonized_bycountyfips(
            harmonized, city)
        frac = 1-(harmonized_city['AJWWE003']/harmonized_city['AJWWE001'])
        most = harmonized_city.iloc[np.where(frac > 
            np.nanpercentile(frac, ptile_upper))]
        least = harmonized_city.iloc[np.where(frac < 
            np.nanpercentile(frac, ptile_lower))]
        ratio_pre.append(least['PRENO2'].mean()/most['PRENO2'].mean())
        ratio_post.append(least['POSTNO2'].mean()/most['POSTNO2'].mean())    
    print('Disparities in 15 largest MSAs for ethnicity:')
    print('Baseline %.3f +/- %.3f'%(np.nanmean(ratio_pre),np.nanstd(ratio_pre)))
    print('Lockdown %.3f +/- %.3f'%(np.nanmean(ratio_post),np.nanstd(ratio_post)))
    print('\n')
    return

def fig3(harmonized):
    """
    """
    import numpy as np
    import scipy.stats as st
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy import stats
    priroad_byincome = []
    byincome_cilow, byincome_cihigh = [], []
    priroad_byrace = []
    byrace_cilow, byrace_cihigh = [], []
    priroad_byeducation = []
    byeducation_cilow, byeducation_cihigh = [], []
    priroad_byethnicity = []
    byethnicity_cilow, byethnicity_cihigh = [], []
    priroad_byvehicle = []
    byvehicle_cilow, byvehicle_cihigh = [], []
    pri_density_mean = []
    dno2_mean = []
    ci_low, ci_high = [], []
    # Find tracts with a change in NO2 in the given range
    for ptilel, ptileu in zip(np.arange(0,100,10), np.arange(10,110,10)):    
        dno2 = harmonized.loc[(harmonized['NO2_ABS']>
            np.nanpercentile(harmonized['NO2_ABS'], ptilel)) & (
            harmonized['NO2_ABS']<=np.nanpercentile(harmonized['NO2_ABS'], ptileu))]
        # Find mean primary/secondary road density within 1 km and change in 
        # NO2 for every percentile range
        pri_density_mean.append(dno2['PrimaryWithin1'].mean())
        dno2_mean.append(dno2['NO2_ABS'].mean())
        ci = st.t.interval(0.95, len(dno2)-1, loc=np.mean(dno2['PrimaryWithin1']), 
            scale=st.sem(dno2['PrimaryWithin1']))    
        ci_low.append(ci[0])
        ci_high.append(ci[1])    
    income = harmonized['AJZAE001']
    race = (harmonized['AJWNE002']/harmonized['AJWBE001'])
    education = (harmonized.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/harmonized['AJYPE001'])
    ethnicity = (harmonized['AJWWE002']/harmonized['AJWWE001'])
    vehicle = 1-harmonized['FracNoCar']
    for ptilel, ptileu in zip(np.arange(0,100,10), np.arange(10,110,10)):    
        # Primary road density and confidence intervals by income
        decile_income = harmonized.loc[(income>
            np.nanpercentile(income, ptilel)) & (
            income<=np.nanpercentile(income, ptileu))]
        ci = st.t.interval(0.95, len(decile_income)-1, 
            loc=np.mean(decile_income['PrimaryWithin1']), scale=st.sem(
            decile_income['PrimaryWithin1']))
        priroad_byincome.append(decile_income['PrimaryWithin1'].mean())        
        byincome_cilow.append(ci[0])
        byincome_cihigh.append(ci[1])
        # By race            
        decile_race = harmonized.loc[(race>
            np.nanpercentile(race, ptilel)) & (
            race<=np.nanpercentile(race, ptileu))] 
        ci = st.t.interval(0.95, len(decile_race)-1, 
            loc=np.mean(decile_race['PrimaryWithin1']), scale=st.sem(
            decile_race['PrimaryWithin1']))
        priroad_byrace.append(decile_race['PrimaryWithin1'].mean())    
        byrace_cilow.append(ci[0])
        byrace_cihigh.append(ci[1])            
        # By education
        decile_education = harmonized.loc[(education>
            np.nanpercentile(education, ptilel)) & (
            education<=np.nanpercentile(education, ptileu))]      
        ci = st.t.interval(0.95, len(decile_education)-1, 
            loc=np.mean(decile_education['PrimaryWithin1']), scale=st.sem(
            decile_education['PrimaryWithin1']))
        priroad_byeducation.append(decile_education['PrimaryWithin1'].mean())    
        byeducation_cilow.append(ci[0])
        byeducation_cihigh.append(ci[1])
        # By ethnicity
        decile_ethnicity = harmonized.loc[(ethnicity>
            np.nanpercentile(ethnicity, ptilel)) & (
            ethnicity<=np.nanpercentile(ethnicity, ptileu))]
        ci = st.t.interval(0.95, len(decile_ethnicity)-1, 
            loc=np.mean(decile_ethnicity['PrimaryWithin1']), scale=st.sem(
            decile_ethnicity['PrimaryWithin1']))
        priroad_byethnicity.append(decile_ethnicity['PrimaryWithin1'].mean())    
        byethnicity_cilow.append(ci[0])
        byethnicity_cihigh.append(ci[1])
        # By vehicle ownership
        decile_vehicle = harmonized.loc[(vehicle>
            np.nanpercentile(vehicle, ptilel)) & (
            vehicle<=np.nanpercentile(vehicle, ptileu))]    
        ci = st.t.interval(0.95, len(decile_vehicle)-1, 
            loc=np.mean(decile_vehicle['PrimaryWithin1']), scale=st.sem(
            decile_vehicle['PrimaryWithin1']))
        priroad_byvehicle.append(decile_vehicle['PrimaryWithin1'].mean())    
        byvehicle_cilow.append(ci[0])
        byvehicle_cihigh.append(ci[1])    
    # Initialize figure, axis
    fig = plt.figure(figsize=(8,5))
    ax1 = plt.subplot2grid((1,1),(0,0))
    color_density = 'k'
    color1 = '#0095A8'
    color2 = '#FF7043'
    color3 = '#5D69B1'
    color4 = '#CC3A8E'
    color5 = '#4daf4a'
    # Plotting
    ax1.plot(pri_density_mean, lw=2, marker='o', 
        markeredgecolor=color_density, markerfacecolor='w', zorder=10,
        color=color_density)
    ax1.fill_between(np.arange(0,10,1),ci_low, ci_high, facecolor=color_density,
        alpha=0.2)
    ax1.plot(priroad_byincome, ls='-', lw=2, color=color1, zorder=11)
    ax1.plot(priroad_byeducation, ls='-', lw=2, color=color2, zorder=11)
    ax1.plot(priroad_byrace, ls='-', lw=2, color=color3, zorder=11)
    ax1.plot(priroad_byethnicity, ls='-', lw=2, color=color4, zorder=11)
    ax1.plot(priroad_byvehicle, ls='-', lw=2, color=color5, zorder=11)
    # Arrows to indicate cut off values
    ax1.annotate('',xy=(0.25,0.58),xytext=(0.25,0.64), 
        arrowprops=dict(arrowstyle= '<|-', lw=1, color='k'))
    ax1.text(0.25, 0.56, '%.2f'%pri_density_mean[0], fontsize=10, 
        color='k', ha='center', va='center')
    ax1.annotate('',xy=(1.02,0.58),xytext=(1.02,0.64), 
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color5))    
    ax1.text(1.02, 0.56, '%.2f'%priroad_byvehicle[0], fontsize=10, 
        color=color5, ha='center', va='center')  
    # Legend
    ax1.text(4.5, 0.62, '$\mathregular{\Delta}$ NO$_{2}$', fontsize=12, 
        color=color_density, ha='center', va='center')
    ax1.annotate('Smaller',xy=(4.9,0.62),xytext=(6,0.62), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color_density), 
        color=color_density, fontsize=12)
    ax1.annotate('Larger',xy=(4.1,0.62),xytext=(2.6,0.62), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color_density), 
        color=color_density, fontsize=12)
    ax1.text(4.5, 0.585, 'Income', fontsize=12, va='center',
        color=color1, ha='center')
    ax1.annotate('Higher',xy=(4.9,0.585),xytext=(6,0.585), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color1), 
        fontsize=12, color=color1)
    ax1.annotate('Lower',xy=(4.1,0.585),xytext=(2.6,0.585), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color1), 
        fontsize=12, color=color1)
    ax1.text(4.5, 0.55, 'Education', fontsize=12, 
        color=color2, va='center', ha='center')
    ax1.annotate('More',xy=(5.1,0.55),xytext=(6,0.55), va='center', 
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color2), fontsize=12,
        color=color2)
    ax1.annotate('Less',xy=(3.9,0.55),xytext=(2.6,0.55), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color2), 
        fontsize=12, color=color2)
    ax1.text(4.5, 0.515, 'White', fontsize=12, va='center',
        color=color3, ha='center')
    ax1.annotate('More',xy=(4.85,0.515),xytext=(6,0.515), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color3), color=color3,
        fontsize=12)
    ax1.annotate('Less',xy=(4.15,0.515),xytext=(2.6,0.515), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color3), fontsize=12, 
        color=color3)
    ax1.text(4.5, 0.48, 'Hispanic', fontsize=12, 
        color=color4, ha='center', va='center')
    ax1.annotate('Less',xy=(4.95,0.48),xytext=(6,0.48), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color4), fontsize=12, 
        color=color4)
    ax1.annotate('More',xy=(4.05,0.48),xytext=(2.6,0.48), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color4), fontsize=12,
        color=color4)
    ax1.text(4.5, 0.445, 'Vehicle ownership', fontsize=12, ha='center',
        va='center', color=color5)
    ax1.annotate('More',xy=(5.5,0.445),xytext=(6,0.445), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color5), fontsize=12, 
        color=color5)
    ax1.annotate('Less',xy=(3.55,0.445),xytext=(2.6,0.445), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color5), fontsize=12,
        color=color5)
    # Aesthetics 
    ax1.set_xlim([0,9])
    ax1.set_xticks(np.arange(0,10,1))
    ax1.set_xticklabels(['First', 'Second', 'Third', 'Fourth', 'Fifth', 
        'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth'], fontsize=10)
    ax1.set_ylim([0.05,0.65])
    ax1.set_xlabel('Decile', fontsize=12)
    ax1.set_ylabel('Primary road density [roads (1 km radius)$^{-1}$]', 
        fontsize=12)
    plt.savefig(DIR_FIGS+'fig3_revised.pdf',dpi=1000)
    return

def fig4(harmonized, lat_dg, lng_dg, no2_post_dg, no2_pre_dg):
    """
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['hatch.linewidth'] = 0.3     
    import numpy as np
    import matplotlib as mpl
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import shapereader
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # Initialize figure, axes
    fig = plt.figure(figsize=(12,9))
    proj = ccrs.PlateCarree(central_longitude=0.0)
    ax1 = plt.subplot2grid((1,2),(0,0), projection=ccrs.PlateCarree(
        central_longitude=0.))
    ax2 = plt.subplot2grid((1,2),(0,1), projection=ccrs.PlateCarree(
        central_longitude=0.))
    fips = ['36','34']
    extents = [-74.05, -73.8, 40.68, 40.9]
    citycodes = ['36047','36081','36061','36005','36085','36119', '34003', 
        '34017','34031','36079','36087','36103','36059','34023','34025',
        '34029','34035','34013','34039','34027','34037'	,'34019']
    ticks = [np.linspace(-1.5e15,1.5e15,5),
        np.linspace(-0.6e15,0.6e15,5),
        np.linspace(-0.5e15,0.5e15,5)]
    ticklabels = [['%.2f'%x for x in np.linspace(-1.5,1.5,5)],
        ['%.2f'%x for x in np.linspace(-0.6,0.6,5)],
        ['%.2f'%x for x in np.linspace(-0.5,0.5,5)]]
    # Load U.S. counties
    reader = shapereader.Reader(DIR_GEO+'counties/tl_2019_us_county/'+
        'tl_2019_us_county.shp')
    counties = list(reader.geometries())
    counties = cfeature.ShapelyFeature(np.array(counties, dtype=object), proj)
    # Colormaps    
    cmapno2 = plt.get_cmap('coolwarm', 16)
    normno2 = [matplotlib.colors.Normalize(vmin=-1.5e15, vmax=1.5e15),
        matplotlib.colors.Normalize(vmin=-0.6e15, vmax=0.6e15),
        matplotlib.colors.Normalize(vmin=-0.5e15, vmax=0.5e15)]
    cmapincome = plt.get_cmap('Blues_r', 9)
    normincome = matplotlib.colors.Normalize(vmin=15000, vmax=60000)
    cmaprace = plt.get_cmap('Blues_r', 11)
    normrace = matplotlib.colors.Normalize(vmin=0, vmax=100)
    # Loop through cities of interest
    for i in np.arange(0,3,1):
        axa = axes[i*3]
        axb = axes[i*3+1]
        axc = axes[i*3+2]
        # Load demographics for each city 
        harmonized_city = tropomi_census_utils.subset_harmonized_bycountyfips(
            harmonized, citycodes[i])    
        # Find indicies of ~city
        down = (np.abs(lat_dg-extents[i][2])).argmin()
        up = (np.abs(lat_dg-extents[i][3])).argmin()
        left = (np.abs(lng_dg-extents[i][0])).argmin()
        right = (np.abs(lng_dg-extents[i][1])).argmin()
        (np.nanmean(no2_post_dg[down:up+1,left:right+1])-
            np.nanmean(no2_pre_dg[down:up+1,left:right+1]))        
        
        # Calculate ~city average change in NO2 during lockdowns
        # diff_cityavg = (harmonized_city['POSTNO2'].values.mean()-
        #     harmonized_city['PRENO2'].values.mean())
        diff_cityavg = (np.nanmean(no2_post_dg[down:up+1,left:right+1])-
            np.nanmean(no2_pre_dg[down:up+1,left:right+1]))   
        sf = fips[i]
        records, tracts = [], []
        for sfi in sf:
            shp = shapereader.Reader(DIR_GEO+'tigerline/'+
                'tl_2019_%s_tract/tl_2019_%s_tract.shp'%(sfi,sfi))
            recordsi = shp.records()
            tractsi = shp.geometries()
            recordsi = list(recordsi)
            tractsi = list(tractsi)
            tracts.append(tractsi)
            records.append(recordsi)
        # Find records and tracts in city 
        geoids_records = [x.attributes['GEOID'] for x in np.hstack(records)]
        geoids_records = np.where(np.in1d(np.array(geoids_records), 
            harmonized_city.index)==True)[0]
        # Slice records and tracts for only entries in city
        records = list(np.hstack(records)[geoids_records])
        tracts = list(np.hstack(tracts)[geoids_records])
        # Loop through shapefiles in city
        for geoid in harmonized_city.index:
            where_geoid = np.where(np.array([x.attributes['GEOID'] for x in 
                records])==geoid)[0][0]
            tract = tracts[where_geoid]
            # Find demographic data/TROPOMI data in tract
            harmonized_tract = harmonized_city.loc[harmonized_city.index.isin(
                [geoid])]
            # Plot tract-averaged NO2
            no2_tract = (harmonized_tract['POSTNO2'].values[0]-
                harmonized_tract['PRENO2'].values[0])-diff_cityavg
            axa.add_geometries([tract], proj, facecolor=cmapno2(normno2[i](
                no2_tract)), edgecolor='None', alpha=1., zorder=10)                 
            # Income in tract
            income_tract = harmonized_tract['AJZAE001'].values[0]
            if np.isnan(income_tract)==True:
                axb.add_geometries([tract], proj, hatch='\\\\\\\\\\\\\\', 
                    edgecolor='k', linewidth=0, facecolor='None', alpha=1.,
                    zorder=10)            
            else:
                axb.add_geometries([tract], proj, facecolor=cmapincome(normincome(
                    income_tract)), edgecolor='None', alpha=1., zorder=10)         
            # Race (percent white) in tract
            white_tract = (harmonized_tract['AJWNE002'].values[0]/
                harmonized_tract['AJWBE001'].values[0])*100.
            if np.isnan(white_tract)==True:
                axc.add_geometries([tract], proj, hatch='\\\\\\\\\\\\\\', 
                    edgecolor='k', linewidth=0, facecolor='None', alpha=1., 
                    zorder=10)          
            else:            
                axc.add_geometries([tract], proj, facecolor=cmaprace(normrace(
                    white_tract)), edgecolor='None', alpha=1., zorder=10)              
        # Add colorbars
        # Delta NO2
        divider = make_axes_locatable(axa)
        cax = divider.append_axes('right', size='5%', pad=0.1, 
            axes_class=plt.Axes)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmapno2, norm=normno2[i], 
            ticks = ticks[i], spacing='proportional', orientation='vertical', 
            extend='both')      
        cax.yaxis.offsetText.set_visible(False)
        cbar.ax.set_yticklabels(ticklabels[i])
        # Income
        if i == 2: 
            divider = make_axes_locatable(axb)
            cax = divider.append_axes('right', size='5%', pad=0.1, 
                axes_class=plt.Axes)
            mpl.colorbar.ColorbarBase(cax, cmap=cmapincome, norm=normincome, 
                spacing='proportional', orientation='vertical', extend='both')      
            # Percent white
            divider = make_axes_locatable(axc)
            cax = divider.append_axes('right', size='5%', pad=0.1, 
                axes_class=plt.Axes)
            mpl.colorbar.ColorbarBase(cax, cmap=cmaprace, norm=normrace, 
                spacing='proportional', orientation='vertical', extend='neither')  
        # Add roads
        for sfi in sf:
            shp = shapereader.Reader(DIR_GEO+
                'tigerline/roads/tl_2019_%s_prisecroads/'%sfi+
                'tl_2019_%s_prisecroads.shp'%sfi)   
            roads_records = list(shp.records())
            roads = list(shp.geometries())
            # Select only interstates
            roads_rttyp = [x.attributes['RTTYP'] for x in roads_records]
            where_interstate = np.where((np.array(roads_rttyp)=='I') |
                                        (np.array(roads_rttyp)=='U'))[0]
            roads_i = []
            roads_i += [roads[x] for x in where_interstate]
            roads = cfeature.ShapelyFeature(roads_i, proj)
            axa.add_feature(roads, facecolor='None', edgecolor='k', 
                zorder=16, lw=0.75)    
        for ax in [axa, axb, axc]:
            ax.set_extent(extents[i], proj)
            ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='10m',
                facecolor='none', name='coastline', lw=0.5, 
                edgecolor='k'), zorder=15)
            ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='10m',
                facecolor='grey', name='lakes', lw=0.5, 
                edgecolor='k'), zorder=15)
            ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='10m',
                facecolor='grey', name='ocean', lw=0.5, 
                edgecolor='k'), zorder=15)        
            ax.add_feature(counties, facecolor='None', lw=0.5, edgecolor='k', 
                zorder=11)    
    # Add axis titles
    ax1.set_title('(a) New York', loc='left')
    ax2.set_title('(b) Atlanta', loc='left')
    ax3.set_title('(c) Detroit', loc='left')
    ax4.set_title('(d)', loc='left')
    ax5.set_title('(e)', loc='left')
    ax6.set_title('(f)', loc='left')
    ax7.set_title('(g)', loc='left')
    ax8.set_title('(h)', loc='left')
    ax9.set_title('(i)', loc='left')
    # Add axis labels
    titstr = '($\mathregular{\delta}$NO$_{2\mathregular{, local}}$ - '+\
        '$\mathregular{\delta}$NO$_{2\mathregular{, city\:average}}$)'+\
        '/10$^{\mathregular{15}}$\n[molec cm$^{\mathregular{-2}}$]'
    ax1.text(-0.1, 0.5, titstr, ha='center', rotation='vertical', 
        rotation_mode='anchor', transform=ax1.transAxes, fontsize=12)
    ax4.text(-0.1, 0.5, 'Median household income [$]', ha='center', 
        rotation='vertical', rotation_mode='anchor', transform=ax4.transAxes, 
        fontsize=12)
    ax7.text(-0.1, 0.5, 'White [%]', ha='center', rotation='vertical', 
        rotation_mode='anchor', transform=ax7.transAxes, fontsize=12)
    for ax in axes: 
        ax.set_aspect('auto')
        ax.outline_patch.set_zorder(20)
    plt.subplots_adjust(left=0.07, top=0.95, bottom=0.05, wspace=0.3)
    plt.savefig(DIR_FIGS+'fig4_revised.pdf', dpi=1000)
    return

def figS1():
    """
    """
    def get_merged_csv(flist, **kwargs):
        """Function reads CSV files in the list comprehension loop, this list of
        DataFrames will be passed to the pd.concat() function which will return 
        single concatenated DataFrame. Adapted from: 
        https://stackoverflow.com/questions/35973782/reading-multiple-csv-
        files-concatenate-list-of-file-names-them-into-a-singe-dat
        """
        from dask import dataframe as dd
        return dd.concat([dd.read_csv(f, **kwargs) for f in flist])
    import time
    start_time = time.time()
    print('# # # # Loading AQS NO2 ...') 
    import numpy as np
    from decimal import Decimal
    from dask import dataframe as dd
    import pandas as pd
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('/Users/ghkerr/phd/utils/')
    from geo_idx import geo_idx
    PATH_AQS = '/Users/ghkerr/GW/data/aq/aqs/'
    years = [2019]
    date_start, date_end = '2019-03-13', '2019-06-13'
    dtype = {'State Code' : np.str,'County Code' : np.str,'Site Num' : np.str,
        'Parameter Code' : np.str, 'POC' : np.str, 'Latitude' : np.float64,
        'Longitude' : np.float64, 'Datum' : np.str, 'Parameter Name' : np.str,
        'Date Local' : np.str, 'Time Local' : np.str, 'Date GMT' : np.str,
        'Time GMT' : np.str, 'Sample Measurement' : np.float64, 
        'Units of Measure' : np.str, 'MDL' : np.str, 'Uncertainty' : np.str,
        'Qualifier' : np.str, 'Method Type' : np.str, 'Method Code' : np.str,
        'Method Name' : np.str, 'State Name' : np.str, 
        'County Name' : np.str, 'Date of Last Change' : np.str}
    filenames_no2 = []
    # Fetch file names for years of interest
    for year in years:
        filenames_no2.append(PATH_AQS+'hourly_42602_%s.csv'%year)
    filenames_no2.sort()
    # Read multiple CSV files (yearly) into Pandas dataframe 
    aqs_no2_raw = get_merged_csv(filenames_no2, dtype=dtype, 
        usecols=list(dtype.keys()))
    # Create site ID column 
    aqs_no2_raw['Site ID'] = aqs_no2_raw['State Code']+'-'+\
        aqs_no2_raw['County Code']+'-'+aqs_no2_raw['Site Num']
    # Drop unneeded columns; drop latitude/longitude coordinates for 
    # temperature observations as the merging of the O3 and temperature 
    # DataFrames will supply these coordinates 
    to_drop = ['Parameter Code', 'POC', 'Datum', 'Parameter Name',
        'Date GMT', 'Time GMT', 'Units of Measure', 'MDL', 'Uncertainty', 
        'Qualifier', 'Method Type', 'Method Code', 'Method Name', 'State Name',
        'County Name', 'Date of Last Change', 'State Code', 'County Code', 
        'Site Num']
    aqs_no2_raw = aqs_no2_raw.drop(to_drop, axis=1)
    # Select months in measuring period     
    aqs_no2_raw = aqs_no2_raw.loc[dd.to_datetime(aqs_no2_raw['Date Local']).isin(
        pd.date_range(date_start,date_end))]
    aqs_no2 = aqs_no2_raw.groupby(['Site ID']).mean()
    # Turns lazy Dask collection into its in-memory equivalent
    aqs_no2 = aqs_no2.compute()
    aqs_no2_raw = aqs_no2_raw.compute()
    # Loop through rows (stations) and find closest TROPOMI grid cell
    tropomi_no2_atstations = []
    for row in np.arange(0, len(aqs_no2), 1):
        aqs_no2_station = aqs_no2.iloc[row]
        lng_station = aqs_no2_station['Longitude']
        lat_station = aqs_no2_station['Latitude']
        lng_tropomi_near = geo_idx(lng_station, lng_dg)
        lat_tropomi_near = geo_idx(lat_station, lat_dg)
        if (lng_tropomi_near is None) or (lat_tropomi_near is None):
            tropomi_no2_atstations.append(np.nan)
        else:
            tropomi_no2_station = no2_pre_dg[lat_tropomi_near,lng_tropomi_near]
            tropomi_no2_atstations.append(tropomi_no2_station)
    aqs_no2['TROPOMINO2'] = tropomi_no2_atstations
    # Site IDs for on-road monitoring sites 
    # from https://www3.epa.gov/ttnamti1/nearroad.html
    onroad = ['13-121-0056','13-089-0003','48-453-1068','06-029-2019',
        '24-027-0006','24-005-0009','01-073-2059','16-001-0023','25-025-0044',
        '25-017-0010','36-029-0023','37-119-0045','17-031-0218','17-031-0118',
        '39-061-0048','39-035-0073','39-049-0038','48-113-1067','48-439-1053',
        '08-031-0027','08-031-0028','19-153-6011','26-163-0093','26-163-0095',
        '06-019-2016','09-003-0025','48-201-1066','48-201-1052','18-097-0087',
        '12-031-0108','29-095-0042','32-003-1501','32-003-1502','06-059-0008',
        '06-037-4008','21-111-0075','47-157-0100','12-011-0035','12-086-0035',
        '55-079-0056','27-053-0962','27-037-0480','47-037-0040','22-071-0021',
        '34-003-0010','36-081-0125','40-109-0097','12-095-0009','42-101-0075',
        '42-101-0076','04-013-4019','04-013-4020','42-003-1376','41-067-0005',
        '44-007-0030','37-183-0021','51-760-0025','06-071-0026','06-071-0027', 
        '36-055-0015','06-067-0015','49-035-4002','48-029-1069','06-073-1017',
        '06-001-0012','06-001-0013','06-001-0015','06-085-0006','72-061-0006',
        '53-033-0030','53-053-0024','29-510-0094','29-189-0016','12-057-0113',
        '12-057-1111','12-103-0027','51-059-0031','11-001-0051']
    fig = plt.figure(figsize=(7,9))
    ax = plt.subplot2grid((2,1),(0,0))
    ax2 = plt.subplot2grid((2,1),(1,0))
    # Axis titles
    ax.set_title('(a)', loc='left', fontsize=12)
    ax2.set_title('(b)', loc='left', fontsize=12)
    color_white = '#0095A8'
    color_non = '#FF7043'
    # Select mobile sites vs other AQS sites
    aqs_onroad = aqs_no2.loc[aqs_no2.index.isin(onroad)]
    aqs_other = aqs_no2.loc[~aqs_no2.index.isin(onroad)]
    # Plotting
    ax.plot(aqs_onroad['TROPOMINO2'].values, 
        aqs_onroad['Sample Measurement'].values, 'o', markersize=3, 
        label='Near-road', color='darkgrey')
    ax.plot(aqs_other['TROPOMINO2'].values, 
        aqs_other['Sample Measurement'].values, 'ko', markersize=4, 
        label='Not near-road')
    ax.set_xlabel('TROPOMI NO$_{2}$/10$^{16}$ [molec cm$^{-2}$]', fontsize=12)
    ax.set_ylabel('AQS NO$_{2}$ [ppbv]', fontsize=12)
    ax.legend(frameon=False, loc=4, fontsize=12)
    ax.xaxis.offsetText.set_visible(False)
    # Line of best fit for non-near road 
    idx = np.isfinite(aqs_other['TROPOMINO2'].values) & \
        np.isfinite(aqs_other['Sample Measurement'].values)
    m, b = np.polyfit(aqs_other['TROPOMINO2'].values[idx], 
        aqs_other['Sample Measurement'].values[idx], 1)
    r2 = np.corrcoef(aqs_other['TROPOMINO2'].values[idx], 
        aqs_other['Sample Measurement'].values[idx])[0,1]**2
    ax.plot(np.sort(aqs_other['TROPOMINO2'].values), (m*
        np.sort(aqs_other['TROPOMINO2'].values)+b), color=color_non, 
        label='Linear fit')
    print('Check to ensure that the plot says'+
        ' %.2E, as this is hard-coded in!'%Decimal(m))
    ax.set_xlim([0,1.25e16])
    ax.set_ylim([0,26])
    # Most and least polluted collocated NO2 from TROPOMI
    tropomi_90 = aqs_other['TROPOMINO2'].values[np.where(
        aqs_other['TROPOMINO2'].values > np.nanpercentile(
        aqs_other['TROPOMINO2'].values, 90))].mean()
    tropomi_10 = aqs_other['TROPOMINO2'].values[np.where(
        aqs_other['TROPOMINO2'].values < np.nanpercentile(
        aqs_other['TROPOMINO2'].values, 10))].mean()
    # AQS NO2 at most and least polluted TROPOMI NO2 sites
    aqs_90 = aqs_other['Sample Measurement'].values[np.where(
        aqs_other['TROPOMINO2'].values > np.nanpercentile(
        aqs_other['TROPOMINO2'].values, 90))].mean()
    aqs_10 = aqs_other['Sample Measurement'].values[np.where(
        aqs_other['TROPOMINO2'].values < np.nanpercentile(
        aqs_other['TROPOMINO2'].values, 10))].mean()    
    # # Indicate the the 90th percentile vs. 10th percentile AQS ratio divided 
    # # by the 90th percentile vs. 10th percentile TROPOMI ratio
    # ratio = (tropomi_90/tropomi_10)/(aqs_90/aqs_10)
    ax.text(0.05e16, 19.5, 'm = 1.5 x 10$^{-15}$ ppbv (molec cm$^{-2}$)$^{-1}$'+
        '\nb = %.1f\nR$^{2}$ = 0.62'%b, color=color_non, fontsize=12)
    # ax.text(0.05e16, 18.5, 'Ratio = %.1f'%(ratio), color=color_non, fontsize=12)
    # Remove on-road stations: 
    aqs_no2_raw = aqs_no2_raw.loc[~aqs_no2_raw['Site ID'].isin(onroad)]
    # Select stations where TROPOMI values indicate very polluted (> 90th 
    # percentile) conditions and not polluted (< 10th percentile)
    tropomi_polluted = aqs_other.iloc[np.where(aqs_other['TROPOMINO2'].values > 
        np.nanpercentile(aqs_other['TROPOMINO2'].values, 90))].index
    tropomi_notpolluted = aqs_other.iloc[np.where(aqs_other['TROPOMINO2'].values < 
        np.nanpercentile(aqs_other['TROPOMINO2'].values, 10))].index
    aqs_no2_raw_polluted = aqs_no2_raw.loc[aqs_no2_raw['Site ID'].isin(
        tropomi_polluted)]
    aqs_no2_raw_notpolluted = aqs_no2_raw.loc[aqs_no2_raw['Site ID'].isin(
        tropomi_notpolluted)]
    aqs_no2_raw_polluted = aqs_no2_raw_polluted.groupby(['Time Local']).mean()
    aqs_no2_raw_notpolluted = aqs_no2_raw_notpolluted.groupby(['Time Local']).mean()
    aqs_no2_raw_mean = aqs_no2_raw.groupby(['Time Local']).mean()
    ax2.plot(aqs_no2_raw_notpolluted['Sample Measurement'], '-', color='#0095A8',
              label='Least polluted')
    ax2.plot(aqs_no2_raw_polluted['Sample Measurement'], '-', color='#FF7043',
              label='Most polluted')
    # ax2.plot(aqs_no2_raw_mean['Sample Measurement'], '-k', label='Non-on-road')
    ax2.legend(frameon=False, loc=1)
    ax2.set_xlabel('Local time', fontsize=12)
    ax2.set_ylabel('AQS NO$_{2}$ [ppbv]', fontsize=12)
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)
    ax2.vlines(x='13:00', ymin=0, ymax=20, linestyle='--', color='darkgrey',
        zorder=0)
    ax2.set_xlim(['00:00','23:00'])
    ax2.set_ylim([0,18])
    # Include differences between 24-hour average and the 13:00 value for 
    # all three curves
    ratio = (aqs_no2_raw_polluted['Sample Measurement'].values.mean()/
        aqs_no2_raw_polluted['Sample Measurement']['13:00'])
    ax2.text('01:00', 8, '24-hour average/13:00 hours = %.1f'%(ratio), 
        color='#FF7043', fontsize=12)
    ratio = (aqs_no2_raw_notpolluted['Sample Measurement'].values.mean()/
        aqs_no2_raw_notpolluted['Sample Measurement']['13:00'])
    ax2.text('01:00', 2.5, '24-hour average/13:00 hours = %.1f'%(ratio), 
        color='#0095A8', fontsize=12)
    plt.subplots_adjust(hspace=0.35)
    plt.savefig(DIR_FIGS+'figS1_revised.pdf', dpi=1000)
    return 

def figS2(FIPS):
    """
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    import pandas as pd
    # NHGIS census information version (may need to change if there are 
    # updates to census information)
    nhgis_version = '0003_ds239_20185_2018'
    # DataFrame that will be filled with harmonzied data for multiple states
    harmonizedms = pd.DataFrame()
    # Loop through states of interest and read in harmonized NO2/census data
    for FIPS_i in FIPS:
        state_harm_i = pd.read_csv(DIR_HARM+'metsens/'
            'Tropomi_NO2_interpolated_%s_nhgis%s_tract_metsens.csv'%(FIPS_i, 
            nhgis_version), delimiter=',', header=0, engine='python')
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
        harmonizedms = harmonizedms.append(state_harm_i)
    # Split into rural and urban tracts
    harmonizedms_urban, harmonizedms_rural = \
        tropomi_census_utils.split_harmonized_byruralurban(harmonizedms)
    # All
    frac_white = ((harmonizedms['AJWNE002'])/harmonizedms['AJWBE001'])
    mostwhite = harmonizedms.iloc[np.where(frac_white > 
        np.nanpercentile(frac_white, ptile_upper))]
    leastwhite = harmonizedms.iloc[np.where(frac_white < 
        np.nanpercentile(frac_white, ptile_lower))]
    mostwealthy = harmonizedms.loc[harmonizedms['AJZAE001'] > 
        np.nanpercentile(harmonizedms['AJZAE001'], 90)]
    leastwealthy = harmonizedms.loc[harmonizedms['AJZAE001'] < 
        np.nanpercentile(harmonizedms['AJZAE001'], 10)]
    frac_educated = (harmonizedms.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        harmonizedms['AJYPE001'])
    mosteducated = harmonizedms.iloc[np.where(frac_educated > 
        np.nanpercentile(frac_educated, 90))]
    leasteducated = harmonizedms.iloc[np.where(frac_educated < 
        np.nanpercentile(frac_educated, 10))]
    mosts = [mostwhite, mostwealthy, mosteducated]
    leasts = [leastwhite, leastwealthy, leasteducated]
    # Urban    
    frac_white = ((harmonizedms_urban['AJWNE002'])/harmonizedms_urban['AJWBE001'])
    mostwhite = harmonizedms_urban.iloc[np.where(frac_white > 
        np.nanpercentile(frac_white, ptile_upper))]
    leastwhite = harmonizedms_urban.iloc[np.where(frac_white < 
        np.nanpercentile(frac_white, ptile_lower))]
    mostwealthy = harmonizedms_urban.loc[harmonizedms_urban['AJZAE001'] > 
        np.nanpercentile(harmonizedms_urban['AJZAE001'], 90)]
    leastwealthy = harmonizedms_urban.loc[harmonizedms_urban['AJZAE001'] < 
        np.nanpercentile(harmonizedms_urban['AJZAE001'], 10)]
    frac_educated = (harmonizedms_urban.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        harmonizedms_urban['AJYPE001'])
    mosteducated = harmonizedms_urban.iloc[np.where(frac_educated > 
        np.nanpercentile(frac_educated, 90))]
    leasteducated = harmonizedms_urban.iloc[np.where(frac_educated < 
        np.nanpercentile(frac_educated, 10))]
    mosts_urban = [mostwhite, mostwealthy, mosteducated]
    leasts_urban = [leastwhite, leastwealthy, leasteducated]
    # Plotting
    color_most = '#0095A8'
    color_least = '#FF7043'
    fig = plt.figure(figsize=(12,5))
    ax1 = plt.subplot2grid((2,3),(0,0),rowspan=2)
    ax2 = plt.subplot2grid((2,3),(0,1),rowspan=2)
    ax3 = plt.subplot2grid((2,3),(0,2),rowspan=2)
    alpha = 1
    yticks = [0.,1,2,3,4,5,6,7,8.,9,10,11,12,13,14]
    for i, ax in enumerate([ax1, ax2, ax3]):
        most = mosts[i]
        least = leasts[i]
        # Seven day averages
        ax.plot(most['trop7_031519'].mean(), 1, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_031519'].mean(), 1, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_032219'].mean(), 1, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_032219'].mean(), 1, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_032919'].mean(), 1, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_032919'].mean(), 1, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_040519'].mean(), 1, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_040519'].mean(), 1, 'o', alpha=alpha,
            color=color_least)
        ax.plot(most['trop7_041219'].mean(), 1, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_041219'].mean(), 1, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_041919'].mean(), 1, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_041919'].mean(), 1, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_042619'].mean(), 1, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_042619'].mean(), 1, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_050319'].mean(), 1, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_050319'].mean(), 1, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_051019'].mean(), 1, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_051019'].mean(), 1, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_051719'].mean(), 1, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_051719'].mean(), 1, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_052419'].mean(), 1, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_052419'].mean(), 1, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_053119'].mean(), 1, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_053119'].mean(), 1, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_060719'].mean(), 1, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_060719'].mean(), 1, 'o', alpha=alpha, 
            color=color_least)
        # 14 day averages
        ax.plot(most['trop14_031519'].mean(), 2, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop14_031519'].mean(), 2, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop14_032919'].mean(), 2, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop14_032919'].mean(), 2, 'o', alpha=alpha,
            color=color_least)
        ax.plot(most['trop14_041219'].mean(), 2, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop14_041219'].mean(), 2, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop14_042619'].mean(), 2, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop14_042619'].mean(), 2, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop14_051019'].mean(), 2, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop14_051019'].mean(), 2, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop14_052419'].mean(), 2, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop14_052419'].mean(), 2, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop14_060719'].mean(), 2, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop14_060719'].mean(), 2, 'o', alpha=alpha, 
            color=color_least)
        # 31 day averages
        ax.plot(most['trop31_031319'].mean(), 3, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop31_031319'].mean(), 3, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop31_041319'].mean(), 3, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop31_041319'].mean(), 3, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop31_051419'].mean(), 3, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop31_051419'].mean(), 3, 'o', alpha=alpha, 
            color=color_least)
        # 3 month period
        ax.plot(most['trop_base'].mean(), 4, 'o', color=color_most)
        ax.plot(least['trop_base'].mean(), 4, 'o', color=color_least)
        # 1 year period
        ax.plot(most['trop_2019'].mean(), 5, 'o', color=color_most)
        ax.plot(least['trop_2019'].mean(), 5, 'o', color=color_least)
        # TROPOMI record period
        ax.plot(most['trop_all'].mean(), 6, 'o', color=color_most)
        ax.plot(least['trop_all'].mean(), 6, 'o', color=color_least)
        # # # # Urban    
        most = mosts_urban[i]
        least = leasts_urban[i]
        ax.plot(most['trop7_031519'].mean(), 9, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_031519'].mean(), 9, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_032219'].mean(), 9, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_032219'].mean(), 9, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_032919'].mean(), 9, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_032919'].mean(), 9, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_040519'].mean(), 9, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_040519'].mean(), 9, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_041219'].mean(), 9, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_041219'].mean(), 9, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_041919'].mean(), 9, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_041919'].mean(), 9, 'o', alpha=alpha,
            color=color_least)
        ax.plot(most['trop7_042619'].mean(), 9, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_042619'].mean(), 9, 'o', alpha=alpha,
            color=color_least)
        ax.plot(most['trop7_050319'].mean(), 9, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_050319'].mean(), 9, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_051019'].mean(), 9, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_051019'].mean(), 9, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_051719'].mean(), 9, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_051719'].mean(), 9, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_052419'].mean(), 9, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_052419'].mean(), 9, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_053119'].mean(), 9, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_053119'].mean(), 9, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop7_060719'].mean(), 9, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop7_060719'].mean(), 9, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop14_031519'].mean(), 10, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop14_031519'].mean(), 10, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop14_032919'].mean(), 10, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop14_032919'].mean(), 10, 'o', alpha=alpha,
            color=color_least)
        ax.plot(most['trop14_041219'].mean(), 10, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop14_041219'].mean(), 10, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop14_042619'].mean(), 10, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop14_042619'].mean(), 10, 'o', alpha=alpha,
            color=color_least)
        ax.plot(most['trop14_051019'].mean(), 10, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop14_051019'].mean(), 10, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop14_052419'].mean(), 10, 'o', alpha=alpha,
            color=color_most)
        ax.plot(least['trop14_052419'].mean(), 10, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop14_060719'].mean(), 10, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop14_060719'].mean(), 10, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop31_031319'].mean(), 11, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop31_031319'].mean(), 11, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop31_041319'].mean(), 11, 'o', alpha=alpha,
            color=color_most)
        ax.plot(least['trop31_041319'].mean(), 11, 'o', alpha=alpha, 
            color=color_least)
        ax.plot(most['trop31_051419'].mean(), 11, 'o', alpha=alpha, 
            color=color_most)
        ax.plot(least['trop31_051419'].mean(), 11, 'o', alpha=alpha,
            color=color_least)
        ax.plot(most['trop_base'].mean(), 12, 'o', color=color_most)
        ax.plot(least['trop_base'].mean(), 12, 'o', color=color_least)
        ax.plot(most['trop_2019'].mean(), 13, 'o', color=color_most)
        ax.plot(least['trop_2019'].mean(), 13, 'o', color=color_least)
        ax.plot(most['trop_all'].mean(), 14, 'o', color=color_most)
        ax.plot(least['trop_all'].mean(), 14, 'o', color=color_least)    
        ax.set_xlim([-0.5e15,9e15])
        ax.set_ylim([-0.5,14.5])
        ax.set_xticks(np.arange(0e15,9e15,1e15))
        ax.set_yticks(yticks)
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False)
        ax.set_xlabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', x=0.26, 
            color='darkgrey')
        ax.tick_params(axis='x', colors='darkgrey')
        ax.xaxis.offsetText.set_visible(False)
        for side in ['right', 'left', 'top', 'bottom']:
            ax.spines[side].set_visible(False)
        ax.grid(axis='x', zorder=0, color='darkgrey')
        ax.invert_yaxis()
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([r'$\bf{All}$', '1 week', '2 weeks', '1 month', 'Baseline', '2019', 
        '2018-2019', '', r'$\bf{Urban}$', '1 week', '2 weeks', '1 month', 'Baseline', '2019', 
        '2018-2019'])
    ax1.set_title('(a) Racial background', loc='left', fontsize=10)
    ax2.set_title('(b) Median household income', loc='left', fontsize=10)
    ax3.set_title('(c) Educational attainment', loc='left', fontsize=10)
    plt.subplots_adjust(wspace=0.05, left=0.09, top=0.95, bottom=0.17, 
        right=0.98)
    # Custom legends for different colored scatterpoints
    custom_lines = [Line2D([0], [0], marker='o', color=color_most, lw=0),
        Line2D([0], [0], marker='o', color=color_least, lw=0)]
    ax1.legend(custom_lines, ['Most white', 'Least white'], 
        bbox_to_anchor=(0.48, -0.22), loc=8, ncol=2, fontsize=10, 
        frameon=False)
    custom_lines = [Line2D([0], [0], marker='o', color=color_most, lw=0),
        Line2D([0], [0], marker='o', color=color_least, lw=0)]
    ax2.legend(custom_lines, ['Highest income', 'Lowest income'], 
        bbox_to_anchor=(0.48, -0.22), loc=8, ncol=2, fontsize=10, 
        frameon=False)
    custom_lines = [Line2D([0], [0], marker='o', color=color_most, lw=0),
        Line2D([0], [0], marker='o', color=color_least, lw=0)]
    ax3.legend(custom_lines, ['Most educated', 'Least educated'], 
        bbox_to_anchor=(0.48, -0.22), loc=8, ncol=2, fontsize=10, 
        frameon=False)
    plt.savefig(DIR_FIGS+'figS2_revised.pdf', dpi=1000)
    return 

def figS3(harmonized, harmonized_rural):
    """
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from decimal import Decimal
    fig = plt.figure(figsize=(9,9))
    ax1 = plt.subplot2grid((13,2),(0,0), rowspan=2)
    ax2 = plt.subplot2grid((13,2),(0,1), rowspan=2)
    ax3 = plt.subplot2grid((13,2),(2,0), rowspan=2)
    ax4 = plt.subplot2grid((13,2),(2,1), rowspan=2)
    ax5 = plt.subplot2grid((13,2),(4,0), rowspan=2)
    ax6 = plt.subplot2grid((13,2),(4,1), rowspan=2)
    ax1b = plt.subplot2grid((13,2),(7,0), rowspan=2)
    ax2b = plt.subplot2grid((13,2),(7,1), rowspan=2)
    ax3b = plt.subplot2grid((13,2),(9,0), rowspan=2)
    ax4b = plt.subplot2grid((13,2),(9,1), rowspan=2)
    ax5b = plt.subplot2grid((13,2),(11,0), rowspan=2)
    ax6b = plt.subplot2grid((13,2),(11,1), rowspan=2)
    colors = ['#0095A8','darkgrey','#FF7043']
    # # # # Bar charts for demographics
    # Largest, smallest and median gains
    increaseno2_all = harmonized.loc[harmonized['NO2_ABS']>
        np.nanpercentile(harmonized['NO2_ABS'], ptile_upper)]
    decreaseno2_all = harmonized.loc[harmonized['NO2_ABS']<
        np.nanpercentile(harmonized['NO2_ABS'], ptile_lower)]
    increaseno2_rural = harmonized_rural.loc[harmonized_rural['NO2_ABS']>
        np.nanpercentile(harmonized_rural['NO2_ABS'], ptile_upper)]
    decreaseno2_rural = harmonized_rural.loc[harmonized_rural['NO2_ABS']<
        np.nanpercentile(harmonized_rural['NO2_ABS'], ptile_lower)]
    # # # # Change in NO2
    # All 
    ax1.barh([2,1,0], [decreaseno2_all['NO2_ABS'].mean(), 
        harmonized['NO2_ABS'].mean(), increaseno2_all['NO2_ABS'].mean()], 
        color=colors[0])
    print('Largest gains for NO2 = %.2E'%Decimal(
        decreaseno2_all['NO2_ABS'].mean()))
    ax1.text(decreaseno2_all['NO2_ABS'].mean()+2e13, 2, '-2.63', color='k', 
        va='center')
    print('Mean gains for NO2 = %.2E'%Decimal(harmonized['NO2_ABS'].mean()))
    ax1.text(harmonized['NO2_ABS'].mean()+2e13, 1, '-0.54', color='k', 
        va='center')
    print('Smallest gains for NO2 = %.2E'%Decimal(
        increaseno2_all['NO2_ABS'].mean()))
    ax1.text(increaseno2_all['NO2_ABS'].mean()+2e13, 0, '0.22', color='k', 
        va='center')
    ax1b.barh([2,1,0], [decreaseno2_rural['NO2_ABS'].mean(), harmonized_rural[
        'NO2_ABS'].mean(), increaseno2_rural['NO2_ABS'].mean()], 
        color=colors[0])
    print('Largest gains for NO2 = %.2E'%Decimal(
        decreaseno2_rural['NO2_ABS'].mean()))
    ax1b.text(decreaseno2_rural['NO2_ABS'].mean()+8e13, 2, '-0.79', color='k', 
        va='center')
    print('Mean gains for NO2 = %.2E'%Decimal(
        harmonized_rural['NO2_ABS'].mean()))
    ax1b.text(harmonized_rural['NO2_ABS'].mean()-3.2e14, 1, '-0.15', color='k', 
        va='center')
    print('Smallest gains for NO2 = %.2E'%Decimal(
        increaseno2_rural['NO2_ABS'].mean()))
    ax1b.text(increaseno2_rural['NO2_ABS'].mean()+2e13, 0, '0.26', color='k', 
        va='center')
    # # # # Income
    ax3.barh([2,1,0], [decreaseno2_all['AJZAE001'].mean(), 
        harmonized['AJZAE001'].mean(), increaseno2_all['AJZAE001'].mean()], 
        color=colors[0])
    ax3.text(53000, 2, ' %d'%(decreaseno2_all['AJZAE001'].mean()), color='k', 
        va='center')
    ax3.text(53000, 1, ' %d'%(harmonized['AJZAE001'].mean()), color='k', 
        va='center')
    ax3.text(53000, 0, ' %d'%(increaseno2_all['AJZAE001'].mean()), color='k', 
        va='center')
    ax3b.barh([2,1,0], [decreaseno2_rural['AJZAE001'].mean(), 
        harmonized['AJZAE001'].mean(), increaseno2_rural['AJZAE001'].mean()], 
        color=colors[0])
    ax3b.text(53000, 2, ' %d'%(decreaseno2_rural['AJZAE001'].mean()), 
        color='k', va='center')
    ax3b.text(53000, 1, ' %d'%(harmonized_rural['AJZAE001'].mean()), 
        color='k', va='center')
    ax3b.text(53000, 0, ' %d'%(increaseno2_rural['AJZAE001'].mean()), 
        color='k', va='center')
    # # # # Racial background
    left, i = 0, 0
    labels = ['White', 'Black', 'Other']
    # Largest gains
    for data, color in zip([(decreaseno2_all['AJWNE002']/
        decreaseno2_all['AJWBE001']).mean(),
        (decreaseno2_all['AJWNE003']/decreaseno2_all['AJWBE001']).mean(),      
        ((decreaseno2_all['AJWNE004']+decreaseno2_all['AJWNE005']+
          decreaseno2_all['AJWNE006']+decreaseno2_all['AJWNE007']+
        decreaseno2_all['AJWNE008'])/decreaseno2_all['AJWBE001']).mean()], 
            colors):             
        ax5.barh(2, data, color=color, left=left)
        ax5.text(left+0.01, 2, '%d'%(np.round(data,2)*100), color='k', 
            va='center')
        left += data
        i = i+1
    left, i = 0, 0   
    for data, color in zip([(decreaseno2_rural['AJWNE002']/
        decreaseno2_rural['AJWBE001']).mean(),
        (decreaseno2_rural['AJWNE003']/decreaseno2_rural['AJWBE001']).mean(),      
        ((decreaseno2_rural['AJWNE004']+decreaseno2_rural['AJWNE005']+
          decreaseno2_rural['AJWNE006']+decreaseno2_rural['AJWNE007']+
        decreaseno2_rural['AJWNE008'])/decreaseno2_rural['AJWBE001']).mean()], 
            colors):             
        ax5b.barh(2, data, color=color, left=left)
        ax5b.text(left+0.01, 2, '%d'%(np.round(data,2)*100), color='k', 
            va='center')
        left += data
        i = i+1
    # Mean demographics
    left, i = 0, 0
    for data, color in zip([(harmonized['AJWNE002']/
        harmonized['AJWBE001']).mean(),
        (harmonized['AJWNE003']/harmonized['AJWBE001']).mean(),      
        ((harmonized['AJWNE004']+harmonized['AJWNE005']+
          harmonized['AJWNE006']+harmonized['AJWNE007']+
        harmonized['AJWNE008'])/harmonized['AJWBE001']).mean()], colors):             
        ax5.barh(1, data, color=color, left=left)
        ax5.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='k', 
            va='center')    
        left += data
        i = i+1    
    left, i = 0, 0
    for data, color in zip([(harmonized_rural['AJWNE002']/
        harmonized_rural['AJWBE001']).mean(),
        (harmonized_rural['AJWNE003']/harmonized_rural['AJWBE001']).mean(),      
        ((harmonized_rural['AJWNE004']+harmonized_rural['AJWNE005']+
          harmonized_rural['AJWNE006']+harmonized_rural['AJWNE007']+
        harmonized_rural['AJWNE008'])/harmonized_rural['AJWBE001']).mean()], 
            colors):             
        ax5b.barh(1, data, color=color, left=left)
        ax5b.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='k', 
            va='center')    
        left += data
        i = i+1        
    # Smallest gains
    left, i = 0, 0
    for data, color in zip([(increaseno2_all['AJWNE002']/
        increaseno2_all['AJWBE001']).mean(),
        (increaseno2_all['AJWNE003']/increaseno2_all['AJWBE001']).mean(),      
        ((increaseno2_all['AJWNE004']+increaseno2_all['AJWNE005']+
          increaseno2_all['AJWNE006']+increaseno2_all['AJWNE007']+
        increaseno2_all['AJWNE008'])/increaseno2_all['AJWBE001']).mean()], 
            colors):             
        ax5.barh(0, data, color=color, left=left)
        ax5.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
            va='center')    
        if i==2:
            ax5.text(0.88, -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold')
        if i==1:
            ax5.text(0.75, -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold')
        if i==0:
            ax5.text(left+0.01, -0.9, labels[i], color=colors[i], va='center',
                      fontweight='bold')
        left += data       
        i = i+1    
    left, i = 0, 0
    for data, color in zip([(increaseno2_rural['AJWNE002']/
        increaseno2_rural['AJWBE001']).mean(),
        (increaseno2_rural['AJWNE003']/increaseno2_rural['AJWBE001']).mean(),      
        ((increaseno2_rural['AJWNE004']+increaseno2_rural['AJWNE005']+
          increaseno2_rural['AJWNE006']+increaseno2_rural['AJWNE007']+
        increaseno2_rural['AJWNE008'])/increaseno2_rural['AJWBE001']).mean()], 
            colors):             
        ax5b.barh(0, data, color=color, left=left)
        if i==1:
            ax5b.text(left+0.005, 0, '%d'%(np.round(data,2)*100), color='k', 
                va='center')
        else:        
            ax5b.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
                va='center')
        if i==2:
            ax5b.text(0.88, -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold')
        if i==1:
            ax5b.text(0.75, -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold')
        if i==0:
            ax5b.text(left+0.01, -0.9, labels[i], color=colors[i], va='center',
                      fontweight='bold')
        left += data       
        i = i+1
    # # # # Ethnic background
    # Largest gains
    left, i = 0, 0
    labels = ['Hispanic', 'Non-Hispanic']
    for data, color in zip([(decreaseno2_all['AJWWE003']/
        decreaseno2_all['AJWWE001']).mean(),
        (decreaseno2_all['AJWWE002']/decreaseno2_all['AJWWE001']).mean()], 
            colors):             
        ax2.barh(2, data, color=color, left=left)
        ax2.text(left+0.01, 2, '%d'%(np.round(data,2)*100), color='k', 
            va='center')
        left += data
        i = i+1
    left, i = 0, 0
    for data, color in zip([(decreaseno2_rural['AJWWE003']/
        decreaseno2_rural['AJWWE001']).mean(),
        (decreaseno2_rural['AJWWE002']/decreaseno2_rural['AJWWE001']).mean()], 
            colors):             
        ax2b.barh(2, data, color=color, left=left)
        ax2b.text(left+0.01, 2, '%d'%(np.round(data,2)*100), color='k', 
            va='center')
        left += data
        i = i+1    
    # Mean demographics
    left, i = 0, 0
    for data, color in zip([(harmonized['AJWWE003']/
        harmonized['AJWWE001']).mean(),
        (harmonized['AJWWE002']/harmonized['AJWWE001']).mean()], colors):             
        ax2.barh(1, data, color=color, left=left)
        ax2.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='k', 
            va='center')
        left += data
        i = i+1
    left, i = 0, 0
    for data, color in zip([(harmonized_rural['AJWWE003']/
        harmonized_rural['AJWWE001']).mean(),
        (harmonized_rural['AJWWE002']/harmonized_rural['AJWWE001']).mean()], 
            colors):
        ax2b.barh(1, data, color=color, left=left)
        ax2b.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='k', 
            va='center')
        left += data
        i = i+1
    # Smallest gains
    left, i = 0, 0
    for data, color in zip([(increaseno2_all['AJWWE003']/
        increaseno2_all['AJWWE001']).mean(),
        (increaseno2_all['AJWWE002']/increaseno2_all['AJWWE001']).mean()], 
            colors): 
        ax2.barh(0, data, color=color, left=left)
        ax2.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
            va='center')    
        if i==1:
            ax2.text(0.2, -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold')            
        else:
            ax2.text(left+0.01, -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold')    
        left += data
        i = i+1
    left, i = 0, 0
    for data, color in zip([(increaseno2_rural['AJWWE003']/
        increaseno2_rural['AJWWE001']).mean(),
        (increaseno2_rural['AJWWE002']/increaseno2_rural['AJWWE001']).mean()], 
            colors): 
        ax2b.barh(0, data, color=color, left=left)
        ax2b.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
            va='center')    
        if i==1:
            ax2b.text(0.2, -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold')            
        else:
            ax2b.text(left+0.01, -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold')             
        left += data
        i = i+1
    # # # # Educational attainment
    # Largest gains
    left, i = 0, 0
    for data, color in zip([
        (decreaseno2_all.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        decreaseno2_all['AJYPE001']).mean(),
        (decreaseno2_all.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        decreaseno2_all['AJYPE001']).mean(),
        (decreaseno2_all.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        decreaseno2_all['AJYPE001']).mean()], colors):             
        ax4.barh(2, data, color=color, left=left)
        ax4.text(left+0.01, 2, '%d'%(np.round(data,2)*100), 
            color='black', va='center') 
        left += data
        i = i+1    
    left, i = 0, 0
    for data, color in zip([
        (decreaseno2_rural.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        decreaseno2_rural['AJYPE001']).mean(),
        (decreaseno2_rural.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        decreaseno2_rural['AJYPE001']).mean(),
        (decreaseno2_rural.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        decreaseno2_rural['AJYPE001']).mean()], colors):             
        ax4b.barh(2, data, color=color, left=left)
        ax4b.text(left+0.01, 2, '%d'%(np.round(data,2)*100), 
            color='k', va='center') 
        left += data
        i = i+1        
    # Mean demographics
    left, i = 0, 0
    for data, color in zip([
        (harmonized.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        harmonized['AJYPE001']).mean(),
        (harmonized.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        harmonized['AJYPE001']).mean(),
        (harmonized.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        harmonized['AJYPE001']).mean()], colors):             
        ax4.barh(1, data, color=color, left=left)
        ax4.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='k', 
            va='center') 
        left += data
        i = i+1 
    left, i = 0, 0
    for data, color in zip([
        (harmonized_rural.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        harmonized_rural['AJYPE001']).mean(),
        (harmonized_rural.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        harmonized_rural['AJYPE001']).mean(),
        (harmonized_rural.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        harmonized_rural['AJYPE001']).mean()], colors):             
        ax4b.barh(1, data, color=color, left=left)
        ax4b.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='k', 
            va='center') 
        left += data
        i = i+1     
    # Smallest gains
    left, i = 0, 0
    labels = ['High school', 'College', 'Graduate']
    for data, color in zip([
        (increaseno2_all.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        increaseno2_all['AJYPE001']).mean(),
        (increaseno2_all.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        increaseno2_all['AJYPE001']).mean(),
        (increaseno2_all.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        increaseno2_all['AJYPE001']).mean()], colors):             
        ax4.barh(0, data, color=color, left=left)
        ax4.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
            va='center')
        if i==2:
            ax4.text(0.82, -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold', fontsize=10)         
        else:   
            ax4.text(left, -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold', fontsize=10) 
        left += data
        i = i+1  
    left, i = 0, 0
    for data, color in zip([
        (increaseno2_rural.loc[:,'AJYPE002':'AJYPE018'].sum(axis=1)/
        increaseno2_rural['AJYPE001']).mean(),
        (increaseno2_rural.loc[:,'AJYPE019':'AJYPE022'].sum(axis=1)/
        increaseno2_rural['AJYPE001']).mean(),
        (increaseno2_rural.loc[:,'AJYPE023':'AJYPE025'].sum(axis=1)/
        increaseno2_rural['AJYPE001']).mean()], colors):             
        ax4b.barh(0, data, color=color, left=left)
        ax4b.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
            va='center')
        if i==2:
            ax4b.text(0.82, -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold', fontsize=10)         
        else:   
            ax4b.text(left, -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold', fontsize=10) 
        left += data
        i = i+1  
    # # # # Vehicle ownership
    # Largest gains
    left, i = 0, 0
    labels = ['None', 'One or more']
    for data, color in zip([decreaseno2_all['FracNoCar'].mean(),
        (1-decreaseno2_all['FracNoCar'].mean())], colors):             
        ax6.barh(2, data, color=color, left=left)
        ax6.text(left+0.01, 2, '%d'%(np.round(data,2)*100),
            color='k', va='center') 
        left += data
        i = i+1
    left, i = 0, 0
    for data, color in zip([decreaseno2_rural['FracNoCar'].mean(),
        (1-decreaseno2_rural['FracNoCar'].mean())], colors):             
        ax6b.barh(2, data, color=color, left=left)
        ax6b.text(left+0.01, 2, '%d'%(np.round(data,2)*100),
            color='k', va='center') 
        left += data
        i = i+1    
    # Mean demographics
    left, i = 0, 0
    for data, color in zip([harmonized['FracNoCar'].mean(),
        (1-harmonized['FracNoCar'].mean())], colors):                        
        ax6.barh(1, data, color=color, left=left)
        ax6.text(left+0.01, 1, '%d'%(np.round(data,2)*100),
            color='k', va='center')     
        left += data
        i = i+1
    left, i = 0, 0
    for data, color in zip([harmonized_rural['FracNoCar'].mean(),
        (1-harmonized_rural['FracNoCar'].mean())], colors):                        
        ax6b.barh(1, data, color=color, left=left)
        ax6b.text(left+0.01, 1, '%d'%(np.round(data,2)*100), color='k', 
            va='center') 
        left += data
        i = i+1    
    # Smallest gains
    left, i = 0, 0
    for data, color in zip([increaseno2_all['FracNoCar'].mean(),
        (1-increaseno2_all['FracNoCar'].mean())], colors):             
        ax6.barh(0, data, color=color, left=left)
        ax6.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
            va='center') 
        if i==1:
            ax6.text(0.15, -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold', fontsize=10)
        else:       
            ax6.text((left*1.5), -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold', fontsize=10)          
        left += data       
        i = i+1
    left, i = 0, 0
    for data, color in zip([increaseno2_rural['FracNoCar'].mean(),
        (1-increaseno2_rural['FracNoCar'].mean())], colors):             
        ax6b.barh(0, data, color=color, left=left)
        ax6b.text(left+0.01, 0, '%d'%(np.round(data,2)*100), color='k', 
            va='center') 
        if i==1:
            ax6b.text(0.15, -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold', fontsize=10)
        else:       
            ax6b.text((left*1.5), -0.9, labels[i], color=colors[i], va='center',
                fontweight='bold', fontsize=10)
        left += data       
        i = i+1    
    # Aesthetics    
    for ax in [ax1, ax1b]:
        ax.set_xlim(-2.7e15,3e14)
        ax.set_xticks([])    
    for ax in [ax3,ax3b]:
        ax.set_xlim([53000,80000])
        ax.set_xticks([])
    for ax in [ax5,ax5b,ax2,ax2b,ax4,ax4b,ax6,ax6b]:
        ax.set_xlim([0,1.])
        ax.set_xticks([])
        ax.set_xticklabels([])
    for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax1b,ax2b,ax3b,ax4b,ax5b,ax6b]:
        ax.set_ylim([-0.5,2.5])
        ax.set_yticks([0,1,2])
        ax.set_yticklabels([])
        for pos in ['right','top','bottom']:
            ax.spines[pos].set_visible(False)
            ax.spines[pos].set_visible(False)    
    for ax in [ax1,ax3,ax5,ax1b,ax3b,ax5b]:
        ax.set_yticklabels(['Smallest drops', 'Average', 'Largest drops'], 
            fontsize=10)
    # Axis titles
    ax1.set_title('(a) $\Delta\:$NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', 
        loc='left', fontsize=10)
    ax1b.set_title('(g) $\Delta\:$NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', 
        loc='left', fontsize=10)
    ax2.set_title('(b) Ethnic background [%]', loc='left', fontsize=10)
    ax2b.set_title('(h) Ethnic background [%]', loc='left', fontsize=10)
    ax3.set_title('(c) Median household income [$]', loc='left', fontsize=10)
    ax3b.set_title('(i) Median household income [$]', loc='left', fontsize=10)
    ax4.set_title('(d) Educational attainment [%]',loc='left', fontsize=10)
    ax4b.set_title('(j) Educational attainment [%]',loc='left', fontsize=10)
    ax5.set_title('(e) Racial background [%]', loc='left', fontsize=10)
    ax5b.set_title('(k) Racial background [%]', loc='left', fontsize=10)
    ax6.set_title('(f) Household vehicle ownership [%]',loc='left', fontsize=10)
    ax6b.set_title('(l) Household vehicle ownership [%]',loc='left', fontsize=10)
    fig.text(0.5, 0.98, '$\mathbf{All}$', fontsize=14, ha='center')
    fig.text(0.5, 0.48, '$\mathbf{Rural}$', fontsize=14, ha='center')
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=3)
    plt.savefig(DIR_FIGS+'figS3_revised.pdf', dpi=1000)
    return 

def figS4(harmonized, harmonized_urban, harmonized_rural):
    """
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import ks_2samp
    def ratio_ci(xpre, ypre, xpost, ypost):
        """Calculate the confidence interval for the ratio NO2 in the least 
        white/least income/least educated tracts to the most white/most income/
        most educated tracts based on the standard error for ratios. Note that the 
        confidence interval is equal to 1.96 * (standard error). 
        
        Parameters
        ----------
        xpre : float
            Baseline NO2 for the least white/lower income/least educated
        ypre : float 
            Baseline NO2 for the most white/highest income/most educated
        xpost : float
            Lockdown NO2 for the least white/lower income/least educated
        ypost : float
            Lockdown NO2 for the most white/highest income/most educated    
        
        Returns
        -------
        ci_pre : float
            Confidence interval for baseline ratio at the 95% level
        ci_pre : float
            Confidence interval for lockdown ratio at the 95% level        
        """
        import numpy as np
        xpre = xpre
        ypre = ypre
        xpost = xpost
        ypost = ypost
        # Mean of the baseline NO2 distributions
        xmeanpre = np.nanmean(xpre)
        ymeanpre = np.nanmean(ypre)
        # Standard error for the baseline NO2 distributions; standard error is
        # the sample standard deviation divided by the root of the number of 
        # samples
        xsepre = np.nanstd(xpre)/np.sqrt(len(xpre))
        ysepre = np.nanstd(ypre)/np.sqrt(len(ypre))
        # Same as above but for the lockdown NO2 distributions 
        xmeanpost = np.nanmean(xpost)
        ymeanpost = np.nanmean(ypost)
        xsepost = np.nanstd(xpost)/np.sqrt(len(xpost))
        ysepost = np.nanstd(ypost)/np.sqrt(len(ypost))
        # Ratios
        ratio_pre = xmeanpre/ymeanpre
        ratio_post = xmeanpost/ymeanpost
        # Calculate the standard error of the ratio: the standard error of a 
        # ratio (where the numerator is not a subset of the denominator) is 
        # approximated as
        # SE(hX/hY) = (1/hY)*sqrt(SE(X)^2 + (hX^2/hY^2*(SE(Y)^2)))
        # where hY and hX stand for "hat Y" and "hat X" - the mean values 
        # from https://www2.census.gov/programs-surveys/acs/tech_docs/accuracy/
        # 2018_ACS_Accuracy_Document_Worked_Examples.pdf?
        se_pre = (1/xmeanpre)*np.sqrt((xsepre**2)+
            ((ymeanpre**2/xmeanpre**2)*(ysepre**2)))
        se_post = (1/xmeanpost)*np.sqrt((xsepost**2)+
            ((ymeanpost**2/xmeanpost**2)*(ysepost**2)))
        ci_pre = 1.96*se_pre
        ci_post = 1.96*se_post
        return ci_pre, ci_post
    # # # # 
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
    color_base = '#0095A8'
    color_lock = '#FF7043'
    os = 1.2
    # Initialize figure
    fig = plt.figure(figsize=(12,7))
    ax1 = plt.subplot2grid((2,3),(0,0),rowspan=2)
    ax2 = plt.subplot2grid((2,3),(0,1),rowspan=2)
    ax3 = plt.subplot2grid((2,3),(0,2),rowspan=2)
    i = 0 
    yticks = []
    # For all tracts
    frac_white = ((harmonized['AJWNE002'])/harmonized['AJWBE001'])
    mostwhite = harmonized.iloc[np.where(frac_white > 
        np.nanpercentile(frac_white, ptile_upper))]
    leastwhite = harmonized.iloc[np.where(frac_white < 
        np.nanpercentile(frac_white, ptile_lower))]
    ci_pre, ci_post = ratio_ci(leastwhite['PRENO2'], mostwhite['PRENO2'], 
        leastwhite['POSTNO2'], mostwhite['POSTNO2'])
    ax1.errorbar(2.1, i-os, 
        xerr=ci_pre, fmt='o', ecolor=color_base, color=color_base,
        elinewidth=2, capsize=4)
    ax1.annotate('', xy=(2.14, -1.2), xycoords='data',
        xytext=(2.3, -1.2), textcoords='data', arrowprops=dict(
        arrowstyle='<-', color=color_base))
    ax1.text(2.31, -1.2, '2.56', va='center', color=color_base)
    ax1.errorbar(leastwhite['POSTNO2'].mean()/mostwhite['POSTNO2'].mean(), i+os, 
        xerr=ci_post, fmt='o', ecolor=color_lock, color=color_lock,
        elinewidth=2, capsize=4)
    yticks.append(np.nanmean([i]))
    i = i+7    
    # For rural tracts
    frac_white = ((harmonized_rural['AJWNE002'])/harmonized_rural['AJWBE001'])
    mostwhite = harmonized_rural.iloc[np.where(frac_white > 
        np.nanpercentile(frac_white, ptile_upper))]
    leastwhite = harmonized_rural.iloc[np.where(frac_white < 
        np.nanpercentile(frac_white, ptile_lower))]
    ci_pre, ci_post = ratio_ci(leastwhite['PRENO2'], mostwhite['PRENO2'], 
        leastwhite['POSTNO2'], mostwhite['POSTNO2'])    
    ax1.errorbar(leastwhite['PRENO2'].mean()/mostwhite['PRENO2'].mean(), i-os, 
        xerr=ci_pre, fmt='o', ecolor=color_base, color=color_base,
        elinewidth=2, capsize=4)
    ax1.errorbar(leastwhite['POSTNO2'].mean()/mostwhite['POSTNO2'].mean(), i+os, 
        xerr=ci_post, fmt='o', ecolor=color_lock, color=color_lock,
        elinewidth=2, capsize=4)                                         
    yticks.append(np.nanmean([i]))
    i = i+7  
    # For urban tracts
    frac_white = ((harmonized_urban['AJWNE002'])/harmonized_urban['AJWBE001'])
    mostwhite = harmonized_urban.iloc[np.where(frac_white > 
        np.nanpercentile(frac_white, ptile_upper))]
    leastwhite = harmonized_urban.iloc[np.where(frac_white < 
        np.nanpercentile(frac_white, ptile_lower))]
    ci_pre, ci_post = ratio_ci(leastwhite['PRENO2'], mostwhite['PRENO2'], 
        leastwhite['POSTNO2'], mostwhite['POSTNO2'])
    ax1.errorbar(leastwhite['PRENO2'].mean()/mostwhite['PRENO2'].mean(), i-os, 
        xerr=ci_pre, fmt='o', ecolor=color_base, color=color_base,
        elinewidth=2, capsize=4)
    ax1.errorbar(leastwhite['POSTNO2'].mean()/mostwhite['POSTNO2'].mean(), i+os, 
        xerr=ci_post, fmt='o', ecolor=color_lock, color=color_lock,
        elinewidth=2, capsize=4)
    yticks.append(np.nanmean([i]))
    i = i+7        
    ratio_pre = []
    ratio_post = []
    citynames = [r'$\bf{All}$', r'$\bf{Rural}$', r'$\bf{Urban}$',
        'New York', 'Los Angeles', 'Chicago', 'Dallas', 'Houston', 
        'Washington', 'Miami', 'Philadelphia', 'Atlanta', 'Phoenix', 
        'Boston', 'San Francisco', 'Riverside', 'Detroit', 'Seattle']   
    for city in [newyork, losangeles, chicago, dallas, houston, washington,
        miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
        riverside, detroit, seattle]:
        # Subset for city
        harmonized_city = tropomi_census_utils.subset_harmonized_bycountyfips(
            harmonized, city)
        # Find particular demographic for each city
        frac_white = (harmonized_city['AJWNE002']/harmonized_city['AJWBE001'])            
        mostwhite = harmonized_city.iloc[np.where(frac_white > 
            np.nanpercentile(frac_white, ptile_upper))]
        leastwhite = harmonized_city.iloc[np.where(frac_white < 
            np.nanpercentile(frac_white, ptile_lower))]
        # Calculate significance
        ci_pre, ci_post = ratio_ci(leastwhite['PRENO2'], mostwhite['PRENO2'], 
            leastwhite['POSTNO2'], mostwhite['POSTNO2'])
        e = ax1.errorbar(leastwhite['PRENO2'].mean()/mostwhite['PRENO2'].mean(), i-os, 
            xerr=ci_pre, fmt='o', ecolor=color_base, color=color_base,
            elinewidth=2, capsize=4)
        for b in e[1]:
            b.set_clip_on(False)
        for b in e[2]:
            b.set_clip_on(False)
        e = ax1.errorbar(leastwhite['POSTNO2'].mean()/mostwhite['POSTNO2'].mean(), 
            i+os, xerr=ci_post, fmt='o', ecolor=color_lock, color=color_lock,
            elinewidth=2, capsize=4)
        for b in e[1]:
            b.set_clip_on(False)    
        if ((leastwhite['POSTNO2'].mean()/mostwhite['POSTNO2'].mean()+ci_post)>
            (leastwhite['PRENO2'].mean()/mostwhite['PRENO2'].mean()-ci_pre)):
            ax1.fill([0.5,4,4,0.5], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
                alpha=0.15, facecolor='grey', edgecolor='grey')
        yticks.append(np.nanmean([i]))
        i = i+7
    ax1.set_xlim([0.42,2.26])
    ax1.set_xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2., 2.25])
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(citynames)
    ax1.tick_params(axis='y', left=False, length=0)
    ax1.set_xlabel('Ratio [$\cdot$]', color='darkgrey')
    ax1.tick_params(axis='x', which='both', colors='darkgrey')
    ax1.xaxis.offsetText.set_visible(False)
    for side in ['right', 'left', 'top', 'bottom']:
        ax1.spines[side].set_visible(False)
    ax1.grid(axis='x', zorder=0, color='darkgrey')
    ax1.invert_yaxis()
    # # # # Most versus least wealthy
    i = 0 
    yticks = []
    mostwealthy = harmonized.loc[harmonized['AJZAE001'] > 
        np.nanpercentile(harmonized['AJZAE001'], 90)]
    leastwealthy = harmonized.loc[harmonized['AJZAE001'] < 
        np.nanpercentile(harmonized['AJZAE001'], 10)]
    ci_pre, ci_post = ratio_ci(leastwealthy['PRENO2'], mostwealthy['PRENO2'], 
        leastwealthy['POSTNO2'], mostwealthy['POSTNO2'])
    ax2.errorbar(leastwealthy['PRENO2'].mean()/mostwealthy['PRENO2'].mean(), i-os, 
        xerr=ci_pre, fmt='o', ecolor=color_base, color=color_base,
        elinewidth=2, capsize=4)
    ax2.errorbar(leastwealthy['POSTNO2'].mean()/mostwealthy['POSTNO2'].mean(), 
        i+os, xerr=ci_post, fmt='o', ecolor=color_lock, color=color_lock,
        elinewidth=2, capsize=4)
    ax2.fill([0,4,4,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
        alpha=0.15, facecolor='grey', edgecolor='grey')
    yticks.append(np.nanmean([i]))
    i = i+7    
    mostwealthy = harmonized_rural.loc[harmonized_rural['AJZAE001'] > 
        np.nanpercentile(harmonized_rural['AJZAE001'], 90)]
    leastwealthy = harmonized_rural.loc[harmonized_rural['AJZAE001'] < 
        np.nanpercentile(harmonized_rural['AJZAE001'], 10)]
    ci_pre, ci_post = ratio_ci(leastwealthy['PRENO2'], mostwealthy['PRENO2'], 
        leastwealthy['POSTNO2'], mostwealthy['POSTNO2'])
    ax2.errorbar(leastwealthy['PRENO2'].mean()/mostwealthy['PRENO2'].mean(), i-os, 
        xerr=ci_pre, fmt='o', ecolor=color_base, color=color_base,
        elinewidth=2, capsize=4)
    ax2.errorbar(leastwealthy['POSTNO2'].mean()/mostwealthy['POSTNO2'].mean(), 
        i+os, xerr=ci_post, fmt='o', ecolor=color_lock, color=color_lock,
        elinewidth=2, capsize=4)
    yticks.append(np.nanmean([i]))
    i = i+7  
    mostwealthy = harmonized_urban.loc[harmonized_urban['AJZAE001'] > 
        np.nanpercentile(harmonized_urban['AJZAE001'], 90)]
    leastwealthy = harmonized_urban.loc[harmonized_urban['AJZAE001'] < 
        np.nanpercentile(harmonized_urban['AJZAE001'], 10)]
    ci_pre, ci_post = ratio_ci(leastwealthy['PRENO2'], mostwealthy['PRENO2'], 
        leastwealthy['POSTNO2'], mostwealthy['POSTNO2'])
    ax2.errorbar(leastwealthy['PRENO2'].mean()/mostwealthy['PRENO2'].mean(), i-os, 
        xerr=ci_pre, fmt='o', ecolor=color_base, color=color_base,
        elinewidth=2, capsize=4)
    ax2.errorbar(leastwealthy['POSTNO2'].mean()/mostwealthy['POSTNO2'].mean(), 
        i+os, xerr=ci_post, fmt='o', ecolor=color_lock, color=color_lock,
        elinewidth=2, capsize=4)
    ax2.fill([0,4,4,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
        alpha=0.15, facecolor='grey', edgecolor='grey')
    yticks.append(np.nanmean([i]))
    i = i+7      
    for city in [newyork, losangeles, chicago, dallas, houston, washington,
        miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
        riverside, detroit, seattle]:
        harmonized_city = tropomi_census_utils.subset_harmonized_bycountyfips(
            harmonized, city)
        mostwealthy = harmonized_city.loc[harmonized_city['AJZAE001'] > 
            np.nanpercentile(harmonized_city['AJZAE001'], 90)]
        leastwealthy = harmonized_city.loc[harmonized_city['AJZAE001'] < 
            np.nanpercentile(harmonized_city['AJZAE001'], 10)]
        ci_pre, ci_post = ratio_ci(leastwealthy['PRENO2'], mostwealthy['PRENO2'], 
            leastwealthy['POSTNO2'], mostwealthy['POSTNO2'])
        ax2.errorbar(leastwealthy['PRENO2'].mean()/mostwealthy['PRENO2'].mean(), 
            i-os, xerr=ci_pre, fmt='o', ecolor=color_base, color=color_base,
            elinewidth=2, capsize=4)
        ax2.errorbar(leastwealthy['POSTNO2'].mean()/mostwealthy['POSTNO2'].mean(), 
            i+os, xerr=ci_post, fmt='o', ecolor=color_lock, color=color_lock,
            elinewidth=2, capsize=4)
        if ((leastwealthy['POSTNO2'].mean()/mostwealthy['POSTNO2'].mean()+ci_post)>
            (leastwealthy['PRENO2'].mean()/mostwealthy['PRENO2'].mean()-ci_pre)):
            ax2.fill([0,4,4,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
                alpha=0.15, facecolor='grey', edgecolor='grey')
        i = i+7
    # Aesthetics 
    ax2.set_xlim([0.5,2.26])
    ax2.set_xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2., 2.25])
    ax2.tick_params(axis='x', colors='darkgrey')
    ax2.set_xlabel('Ratio [$\cdot$]', color='darkgrey')
    ax2.set_ylim([ax1.get_ylim()[0], ax1.get_ylim()[1]])
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([])
    ax2.tick_params(axis='y', left=False)
    ax2.xaxis.offsetText.set_visible(False)
    for side in ['right', 'left', 'top', 'bottom']:
        ax2.spines[side].set_visible(False)
    ax2.grid(axis='x', zorder=0, color='darkgrey')
    # Most versus least educated
    i = 0 
    yticks = []
    frac_educated = (harmonized.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        harmonized['AJYPE001'])
    mosteducated = harmonized.iloc[np.where(frac_educated > 
        np.nanpercentile(frac_educated, 90))]
    leasteducated = harmonized.iloc[np.where(frac_educated < 
        np.nanpercentile(frac_educated, 10))]
    ci_pre, ci_post = ratio_ci(leasteducated['PRENO2'], mosteducated['PRENO2'], 
        leasteducated['POSTNO2'], mosteducated['POSTNO2'])
    ax3.errorbar(leasteducated['PRENO2'].mean()/mosteducated['PRENO2'].mean(), i-os, 
        xerr=ci_pre, fmt='o', ecolor=color_base, color=color_base,
        elinewidth=2, capsize=4)
    ax3.errorbar(leasteducated['POSTNO2'].mean()/mosteducated['POSTNO2'].mean(), 
        i+os, xerr=ci_post, fmt='o', ecolor=color_lock, color=color_lock,
        elinewidth=2, capsize=4)
    ax3.fill([0,4,4,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
        alpha=0.15, facecolor='grey', edgecolor='grey') 
    yticks.append(np.nanmean([i]))
    i = i+7
    frac_educated = (harmonized_rural.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        harmonized_rural['AJYPE001'])
    mosteducated = harmonized_rural.iloc[np.where(frac_educated > 
        np.nanpercentile(frac_educated, 90))]
    leasteducated = harmonized_rural.iloc[np.where(frac_educated < 
        np.nanpercentile(frac_educated, 10))]
    ci_pre, ci_post = ratio_ci(leasteducated['PRENO2'], mosteducated['PRENO2'], 
        leasteducated['POSTNO2'], mosteducated['POSTNO2'])
    ax3.errorbar(leasteducated['PRENO2'].mean()/mosteducated['PRENO2'].mean(), i-os, 
        xerr=ci_pre, fmt='o', ecolor=color_base, color=color_base,
        elinewidth=2, capsize=4)
    ax3.errorbar(leasteducated['POSTNO2'].mean()/mosteducated['POSTNO2'].mean(), 
        i+os, xerr=ci_post, fmt='o', ecolor=color_lock, color=color_lock,
        elinewidth=2, capsize=4)
    yticks.append(np.nanmean([i]))
    i = i+7    
    # Urban 
    frac_educated = (harmonized_urban.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        harmonized_urban['AJYPE001'])
    mosteducated = harmonized_urban.iloc[np.where(frac_educated > 
        np.nanpercentile(frac_educated, 90))]
    leasteducated = harmonized_urban.iloc[np.where(frac_educated < 
        np.nanpercentile(frac_educated, 10))]
    ci_pre, ci_post = ratio_ci(leasteducated['PRENO2'], mosteducated['PRENO2'], 
        leasteducated['POSTNO2'], mosteducated['POSTNO2'])
    ax3.errorbar(leasteducated['PRENO2'].mean()/mosteducated['PRENO2'].mean(), i-os, 
        xerr=ci_pre, fmt='o', ecolor=color_base, color=color_base,
        elinewidth=2, capsize=4)
    ax3.errorbar(leasteducated['POSTNO2'].mean()/mosteducated['POSTNO2'].mean(), 
        i+os, xerr=ci_post, fmt='o', ecolor=color_lock, color=color_lock,
        elinewidth=2, capsize=4)
    ax3.fill([0,4,4,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
        alpha=0.15, facecolor='grey', edgecolor='grey')
    yticks.append(np.nanmean([i]))
    i = i+7    
    for city in [newyork, losangeles, chicago, dallas, houston, washington,
        miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
        riverside, detroit, seattle]:
        harmonized_city = tropomi_census_utils.subset_harmonized_bycountyfips(
            harmonized, city)
        frac_educated = (harmonized_city.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
            harmonized_city['AJYPE001'])
        mosteducated = harmonized_city.iloc[np.where(frac_educated > 
            np.nanpercentile(frac_educated, 90))]
        leasteducated = harmonized_city.iloc[np.where(frac_educated < 
            np.nanpercentile(frac_educated, 10))]
        ci_pre, ci_post = ratio_ci(leasteducated['PRENO2'], mosteducated['PRENO2'], 
            leasteducated['POSTNO2'], mosteducated['POSTNO2'])
        ax3.errorbar(leasteducated['PRENO2'].mean()/mosteducated['PRENO2'].mean(), 
            i-os, xerr=ci_pre, fmt='o', ecolor=color_base, color=color_base,
            elinewidth=2, capsize=4)
        ax3.errorbar(leasteducated['POSTNO2'].mean()/mosteducated['POSTNO2'].mean(), 
            i+os, xerr=ci_post, fmt='o', ecolor=color_lock, color=color_lock,
            elinewidth=2, capsize=4)
        if ((leasteducated['POSTNO2'].mean()/mosteducated['POSTNO2'].mean()+ci_post)>
            (leasteducated['PRENO2'].mean()/mosteducated['PRENO2'].mean()-ci_pre)):
            ax3.fill([0,4,4,0], [i-2.1*os,i-2.1*os,i+2.1*os,i+2.1*os], 
                alpha=0.15, facecolor='grey', edgecolor='grey')    
        yticks.append(np.nanmean([i]))
        i = i+7
    ax3.set_xlim([0.5,2.26])
    ax3.set_xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2., 2.25])
    ax3.tick_params(axis='x', colors='darkgrey')
    ax3.set_xlabel('Ratio [$\cdot$]', color='darkgrey')
    ax3.set_ylim([ax1.get_ylim()[0], ax1.get_ylim()[1]])
    ax3.set_yticks(yticks)
    ax3.set_yticklabels([])
    ax3.tick_params(axis='y', left=False)
    ax3.xaxis.offsetText.set_visible(False)
    for side in ['right', 'left', 'top', 'bottom']:
        ax3.spines[side].set_visible(False)
    ax3.grid(axis='x', zorder=0, color='darkgrey')
    ax1.set_title('(a) Racial background', loc='left', fontsize=10)
    ax2.set_title('(b) Median household income', loc='left', fontsize=10)
    ax3.set_title('(c) Educational attainment', loc='left', fontsize=10)
    plt.subplots_adjust(wspace=0.15, left=0.09, top=0.95, bottom=0.17, 
        right=0.98)
    # Custom legends for different colored scatterpoints
    custom_lines = [Line2D([0], [0], marker='o', color=color_base, lw=0),
        Line2D([0], [0], marker='o', color=color_lock, lw=0)]
    ax1.legend(custom_lines, ['Baseline', 'Lockdown'], 
        bbox_to_anchor=(0.2, -0.15), loc=8, ncol=2, fontsize=10, 
        frameon=False)
    plt.savefig(DIR_FIGS+'figS4_revised.pdf', dpi=1000)
    plt.show()
    return 

def figS5(harmonized, harmonized_rural, harmonized_urban):
    """
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # Colors
    color_white = '#0095A8'
    color_non = '#FF7043'
    os = 1.2
    # Initialize figure
    fig = plt.figure(figsize=(11,9))
    # All
    ax1 = plt.subplot2grid((5,3),(0,0))
    ax2 = plt.subplot2grid((5,3),(1,0))
    ax3 = plt.subplot2grid((5,3),(2,0))
    ax4 = plt.subplot2grid((5,3),(3,0))
    ax5 = plt.subplot2grid((5,3),(4,0))
    # Rural
    ax6 = plt.subplot2grid((5,3),(0,1))
    ax7 = plt.subplot2grid((5,3),(1,1))
    ax8 = plt.subplot2grid((5,3),(2,1))
    ax9 = plt.subplot2grid((5,3),(3,1))
    ax10 = plt.subplot2grid((5,3),(4,1))
    # Urban
    ax11 = plt.subplot2grid((5,3),(0,2))
    ax12 = plt.subplot2grid((5,3),(1,2))
    ax13 = plt.subplot2grid((5,3),(2,2))
    ax14 = plt.subplot2grid((5,3),(3,2))
    ax15 = plt.subplot2grid((5,3),(4,2))
    # Loop through sets of axies (sets correspond to all, rural, and urban)
    for harm, axes in zip([harmonized, harmonized_rural, harmonized_urban],
        [[ax1,ax2,ax3,ax4,ax5],[ax6,ax7,ax8,ax9,ax10],[ax11,ax12,ax13,ax14,ax15]]):
        axa = axes[0]
        axb = axes[1] 
        axc = axes[2]
        axd = axes[3]
        axe = axes[4]
        i, yticks = 0, [] 
        for ptilepair in [(5,95),(10,90),(20,80),(25,75)]:
            up = ptilepair[-1]
            down = ptilepair[0]
            # Median household income
            mostwealthy = harm.loc[harm['AJZAE001'] > 
                np.nanpercentile(harm['AJZAE001'], up)]
            leastwealthy = harm.loc[harm['AJZAE001'] < 
                np.nanpercentile(harm['AJZAE001'], down)]
            axa.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, 
                  zorder=12, clip_on=False)
            axa.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, 
                  zorder=12, clip_on=False)
            axa.plot((np.min([mostwealthy['PRENO2'].mean(),
                leastwealthy['PRENO2'].mean()]), 
                np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+
                np.abs(np.diff([mostwealthy['PRENO2'].mean(), 
                leastwealthy['PRENO2'].mean()]))[0]), [i-os,i-os], color='k', 
                ls='-', zorder=10)    
            axa.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', 
                color=color_white, zorder=12, clip_on=False)
            axa.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', 
                color=color_non, zorder=12, clip_on=False)
            axa.plot((np.min([mostwealthy['POSTNO2'].mean(),
                leastwealthy['POSTNO2'].mean()]), 
                np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
                np.abs(np.diff([mostwealthy['POSTNO2'].mean(), 
                leastwealthy['POSTNO2'].mean()]))[0]), [i+os,i+os], color='k', 
                ls='--', zorder=10)
            # Racial background
            frac_white = (harm['AJWNE002']/harm['AJWBE001'])
            mostwhite = harm.iloc[np.where(frac_white > 
                np.nanpercentile(frac_white, up))]
            leastwhite = harm.iloc[np.where(frac_white < 
                np.nanpercentile(frac_white, down))]
            axb.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, 
                  zorder=12, clip_on=False)
            axb.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, 
                  zorder=12, clip_on=False)
            axb.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()]), 
                np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()])+
                np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))[0]), 
                [i-os,i-os], color='k', ls='-', zorder=10)    
            axb.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, 
                zorder=12, clip_on=False)
            axb.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, 
                zorder=12, clip_on=False)
            axb.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
                np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
                np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))[0]), 
                [i+os,i+os], color='k', ls='--', zorder=10)
            # Ethnic background
            frac_hispanic = 1-(harm['AJWWE003']/harm['AJWWE001'])
            most = harm.iloc[np.where(frac_hispanic > 
                np.nanpercentile(frac_hispanic, up))]
            least = harm.iloc[np.where(frac_hispanic < 
                np.nanpercentile(frac_hispanic, down))]
            axc.plot(most['PRENO2'].mean(), i-os, 'o', color=color_white, 
                zorder=12, clip_on=False)
            axc.plot(least['PRENO2'].mean(), i-os, 'o', color=color_non, 
                zorder=12, clip_on=False)
            axc.plot((np.min([most['PRENO2'].mean(),least['PRENO2'].mean()]), 
                np.min([most['PRENO2'].mean(),least['PRENO2'].mean()])+
                np.abs(np.diff([most['PRENO2'].mean(), least['PRENO2'].mean()]))[0]), 
                [i-os,i-os], color='k', ls='-', zorder=10)    
            axc.plot(most['POSTNO2'].mean(), i+os, 'o', color=color_white, 
                zorder=12, clip_on=False)
            axc.plot(least['POSTNO2'].mean(), i+os, 'o', color=color_non, 
                zorder=12, clip_on=False)
            axc.plot((np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()]), 
                np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()])+
                np.abs(np.diff([most['POSTNO2'].mean(), least['POSTNO2'].mean()]))[0]), 
                [i+os,i+os], color='k', ls='--', zorder=10)
            # Educational attainment 
            frac_educated = (harm.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
                harm['AJYPE001'])
            mosteducated = harm.iloc[np.where(frac_educated > 
                np.nanpercentile(frac_educated, up))]
            leasteducated = harm.iloc[np.where(frac_educated < 
                np.nanpercentile(frac_educated, down))]
            axd.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, 
                zorder=12, clip_on=False)
            axd.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, 
                zorder=12, clip_on=False)
            axd.plot((np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()]), 
                np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()])+
                np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
                [i-os,i-os], color='k', ls='-', zorder=10)    
            axd.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, 
                zorder=12, clip_on=False)
            axd.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, 
                zorder=12, clip_on=False)
            axd.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
                np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
                np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
                [i+os,i+os], color='k', ls='--', zorder=10)
            # Vehicle ownership
            frac = 1-harm['FracNoCar']
            most = harm.iloc[np.where(frac > 
                np.nanpercentile(frac, up))]
            least = harm.iloc[np.where(frac < 
                np.nanpercentile(frac, down))]
            axe.plot(most['PRENO2'].mean(), i-os, 'o', color=color_white, 
                zorder=12, clip_on=False)
            axe.plot(least['PRENO2'].mean(), i-os, 'o', color=color_non, 
                zorder=12, clip_on=False)
            axe.plot((np.min([most['PRENO2'].mean(),least['PRENO2'].mean()]), 
                np.min([most['PRENO2'].mean(),least['PRENO2'].mean()])+
                np.abs(np.diff([most['PRENO2'].mean(), least['PRENO2'].mean()]))[0]), 
                [i-os,i-os], color='k', ls='-', clip_on=False, zorder=10)    
            axe.plot(most['POSTNO2'].mean(), i+os, 'o', color=color_white, 
                zorder=12, clip_on=False)
            axe.plot(least['POSTNO2'].mean(), i+os, 'o', color=color_non, 
                zorder=12, clip_on=False)
            axe.plot((np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()]), 
                np.min([most['POSTNO2'].mean(),least['POSTNO2'].mean()])+
                np.abs(np.diff([most['POSTNO2'].mean(), least['POSTNO2'].mean()]))[0]), 
                [i+os,i+os], color='k', ls='--', clip_on=False, zorder=10)
            yticks.append(np.nanmean([i]))   
            i = i+7
    # Aesthetics
    ax1.set_title('(a) All', loc='left', fontsize=12)
    ax2.set_title('(d)', loc='left', fontsize=12)
    ax3.set_title('(g)', loc='left', fontsize=12)
    ax4.set_title('(j)', loc='left', fontsize=12)
    ax5.set_title('(m)', loc='left', fontsize=12)        
    ax6.set_title('(b) Rural', loc='left', fontsize=12)
    ax7.set_title('(e)', loc='left', fontsize=12)
    ax8.set_title('(h)', loc='left', fontsize=12)
    ax9.set_title('(k)', loc='left', fontsize=12)
    ax10.set_title('(n)', loc='left', fontsize=12)
    ax11.set_title('(c) Urban', loc='left', fontsize=12)
    ax12.set_title('(f)', loc='left', fontsize=12)
    ax13.set_title('(i)', loc='left', fontsize=12)
    ax14.set_title('(l)', loc='left', fontsize=12)
    ax15.set_title('(o)', loc='left', fontsize=12)
    ax1.set_ylabel('Median household\nincome', fontsize=12)
    ax2.set_ylabel('Racial\nbackground', fontsize=12)
    ax3.set_ylabel('Ethnic\nbackground', fontsize=12)
    ax4.set_ylabel('Educational\nattainment', fontsize=12)
    ax5.set_ylabel('Vehicle\nownership', fontsize=12)
    # Axes ticks        
    for ax in [ax1,ax2,ax3,ax4,ax5]:
        ax.set_xlim([1.5e15,6.1e15])
        ax.set_xticks([1.5e15,3e15,4.5e15,6e15])
        ax.set_xticklabels([])
    ax5.set_xticklabels(['1.5','3','4.5','6'])
    for ax in [ax6,ax7,ax8,ax9,ax10]:
        ax.set_xlim([0.95e15,2.55e15])
        ax.set_xticks([1e15,1.5e15,2e15,2.5e15])
        ax.set_xticklabels([])
    ax10.set_xticklabels(['1','1.5','2','2.5'])
    for ax in [ax11,ax12,ax13,ax14,ax15]:
        ax.set_xlim([1.95e15,8.05e15])    
        ax.set_xticks([2e15,4e15,6e15,8e15])
        ax.set_xticklabels([])    
    ax15.set_xticklabels(['2','4','6','8'])
    for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15]:    
        ax.set_yticks(yticks)
        ax.set_yticklabels([])
        # Hide spines
        for side in ['right', 'left', 'top', 'bottom']:
            ax.spines[side].set_visible(False)
        ax.tick_params(axis='x', colors='darkgrey')
        ax.xaxis.offsetText.set_visible(False)
        ax.grid(axis='x', zorder=0, which='both', color='darkgrey')
        ax.invert_yaxis()
        ax.tick_params(axis='y', left=False)
    for ax in [ax5,ax10,ax15]:
        ax.set_xlabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', x=0.35, labelpad=4,
            color='darkgrey')
    ticks = ['5/95', r'$\bf{{10/90}}$', '20/80', '25/75']   
    for ax in [ax1,ax2,ax3,ax4,ax5]:    
        ax.set_yticklabels(ticks)
    plt.subplots_adjust(left=0.1, hspace=0.4, wspace=0.4, right=0.8)
    # Custom legend
    custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
        Line2D([0], [0], marker='o', color=color_non, lw=0)]
    ax11.legend(custom_lines, ['Highest income', 'Lowest income'], 
        bbox_to_anchor=(1.5, 0.25), loc=8, ncol=1, frameon=False)
    custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
        Line2D([0], [0], marker='o', color=color_non, lw=0)]
    ax12.legend(custom_lines, ['Most white', 'Least white'], 
        bbox_to_anchor=(1.46, 0.25), loc=8, ncol=1, frameon=False)
    custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
        Line2D([0], [0], marker='o', color=color_non, lw=0)]
    ax13.legend(custom_lines, ['Least Hispanic', 'Most Hispanic'], 
        bbox_to_anchor=(1.49, 0.25), loc=8, ncol=1, frameon=False)
    custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
        Line2D([0], [0], marker='o', color=color_non, lw=0)]
    ax14.legend(custom_lines, ['Most educated', 'Least educated'], 
        bbox_to_anchor=(1.5, 0.25), loc=8, ncol=1, frameon=False)
    custom_lines = [Line2D([0], [0], marker='o', color=color_white, lw=0),
        Line2D([0], [0], marker='o', color=color_non, lw=0),  
        Line2D([0], [0], color='k', lw=1.0),
        Line2D([0], [0], color='k', ls='--', lw=1)]
    ax15.legend(custom_lines, ['Highest ownership', 'Lowest ownership', 
        'Baseline', 'Lockdown'], bbox_to_anchor=(1.54, -0.07), loc=8, ncol=1, 
        frameon=False)
    plt.savefig(DIR_FIGS+'figS5_revised.pdf', dpi=1000)
    plt.show()
    return

def figS6(harmonized_urban): 
    """
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import scipy.stats as st
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy import stats
    import pandas as pd
    # Read in csv file for vehicle ownership/road density
    noxsource = pd.read_csv(DIR_HARM+'noxsourcedensity_us_v2.csv', delimiter=',', 
        header=0, engine='python')
    # Leading 0 for state FIPS codes < 10 
    noxsource['GEOID'] = noxsource['GEOID'].map(lambda x: f'{x:0>11}')
    # Make GEOID a string and index row 
    noxsource = noxsource.set_index('GEOID')
    noxsource.index = noxsource.index.map(str)
    # Make other columns floats
    for col in noxsource.columns:
        noxsource[col] = noxsource[col].astype(float)
    # Merge with harmonized census data
    harmonized_noxsource = harmonized_urban.merge(noxsource, left_index=True, 
        right_index=True)
    ports_byrace = []
    ports_byincome = []
    ports_byeducation = [] 
    ports_byethnicity = []
    ports_byvehicle = []
    cems_byrace, cemsp90_byrace = [], []
    cems_byincome, cemsp90_byincome = [], []
    cems_byeducation, cemsp90_byeducation = [], []
    cems_byethnicity, cemsp90_byethnicity = [], []
    cems_byvehicle, cemsp90_byvehicle = [], []
    airports_byrace = []
    airports_byincome = []
    airports_byeducation = []
    airports_byethnicity = []
    airports_byvehicle = []
    rail_byrace = []
    rail_byincome = []
    rail_byeducation = []
    rail_byethnicity = []
    rail_byvehicle = []
    income = harmonized_noxsource['AJZAE001']
    race = (harmonized_noxsource['AJWNE002']/harmonized_noxsource['AJWBE001'])
    education = (harmonized_noxsource.loc[:,'AJYPE019':'AJYPE025'
        ].sum(axis=1)/harmonized_noxsource['AJYPE001'])
    ethnicity = (harmonized_noxsource['AJWWE002']/harmonized_noxsource['AJWWE001'])
    vehicle = 1-harmonized_noxsource['FracNoCar']
    for ptilel, ptileu in zip(np.arange(0,100,10), np.arange(10,110,10)):    
        # By income
        decile_income = harmonized_noxsource.loc[(income>
            np.nanpercentile(income, ptilel)) & (
            income<=np.nanpercentile(income, ptileu))]
        ports_byincome.append(decile_income['portswithin1'].mean())             
        cems_byincome.append(decile_income['CEMSwithin1'].mean())  
        cemsp90_byincome.append(decile_income['CEMSwithin1_p80'].mean())
        airports_byincome.append(decile_income['airportswithin1'].mean())  
        rail_byincome.append(decile_income['railwithin1'].mean())  
        # By race            
        decile_race = harmonized_noxsource.loc[(race>
            np.nanpercentile(race, ptilel)) & (
            race<=np.nanpercentile(race, ptileu))] 
        ports_byrace.append(decile_race['portswithin1'].mean())             
        cems_byrace.append(decile_race['CEMSwithin1'].mean())  
        cemsp90_byrace.append(decile_race['CEMSwithin1_p80'].mean())          
        airports_byrace.append(decile_race['airportswithin1'].mean())  
        rail_byrace.append(decile_race['railwithin1'].mean())
        # By education 
        decile_education = harmonized_noxsource.loc[(education>np.nanpercentile(
            education, ptilel)) & (education<=np.nanpercentile(education, 
            ptileu))]
        ports_byeducation.append(decile_education['portswithin1'].mean())             
        cems_byeducation.append(decile_education['CEMSwithin1'].mean())  
        cemsp90_byeducation.append(decile_education['CEMSwithin1_p80'].mean())  
        airports_byeducation.append(decile_education['airportswithin1'].mean())  
        rail_byeducation.append(decile_education['railwithin1'].mean())
        # By ethnicity
        decile_ethnicity = harmonized_noxsource.loc[(ethnicity>
            np.nanpercentile(ethnicity, ptilel)) & (
            ethnicity<=np.nanpercentile(ethnicity, ptileu))]    
        ports_byethnicity.append(decile_ethnicity['portswithin1'].mean())             
        cems_byethnicity.append(decile_ethnicity['CEMSwithin1'].mean())  
        cemsp90_byethnicity.append(decile_ethnicity['CEMSwithin1_p80'].mean())  
        airports_byethnicity.append(decile_ethnicity['airportswithin1'].mean())  
        rail_byethnicity.append(decile_ethnicity['railwithin1'].mean())
        # By vehicle ownership            
        decile_vehicle = harmonized_noxsource.loc[(vehicle>
            np.nanpercentile(vehicle, ptilel)) & (vehicle<=np.nanpercentile(
            vehicle, ptileu))]
        ports_byvehicle.append(decile_vehicle['portswithin1'].mean())             
        cems_byvehicle.append(decile_vehicle['CEMSwithin1'].mean())  
        cemsp90_byvehicle.append(decile_vehicle['CEMSwithin1_p80'].mean())  
        airports_byvehicle.append(decile_vehicle['airportswithin1'].mean())  
        rail_byvehicle.append(decile_vehicle['railwithin1'].mean())    
    fig = plt.figure(figsize=(11.5,7))
    ax1 = plt.subplot2grid((2,2),(0,0))
    ax2 = plt.subplot2grid((2,2),(0,1))
    ax3 = plt.subplot2grid((2,2),(1,0))
    ax4 = plt.subplot2grid((2,2),(1,1))
    # Colors for each demographic variable
    color1 = '#0095A8'
    color2 = '#FF7043'
    color3 = '#5D69B1'
    color4 = '#CC3A8E'
    color5 = '#4daf4a'
    axes = [ax1, ax2, ax3, ax4]
    curves = [[ports_byrace, ports_byincome, ports_byeducation, 
          ports_byethnicity, ports_byvehicle], 
        [cems_byrace, cems_byincome, cems_byeducation, cems_byethnicity, 
          cems_byvehicle], 
        [airports_byrace, airports_byincome, airports_byeducation, 
          airports_byethnicity, airports_byvehicle],
        [rail_byrace, rail_byincome, rail_byeducation, rail_byethnicity, 
          rail_byvehicle]]
    # Loop through NOx sources
    for i in np.arange(0,4,1):
        # Plotting
        axes[i].plot(curves[i][1], ls='-', lw=2, color=color1, zorder=11)
        axes[i].plot(curves[i][2], ls='-', lw=2, color=color2, zorder=11)
        axes[i].plot(curves[i][0], ls='-', lw=2, color=color3, zorder=11)
        axes[i].plot(curves[i][3], ls='-', lw=2, color=color4, zorder=11)
        axes[i].plot(curves[i][4], ls='-', lw=2, color=color5, zorder=11)
    # Inset axis for large CEMS sources
    ax2ins = ax2.inset_axes([0.45, 0.5, 0.5, 0.45]) #Left, Bottom, Width, Height
    ax2ins.plot(cemsp90_byrace, ls='-', lw=2, color=color1, zorder=11)
    ax2ins.plot(cemsp90_byincome, ls='-', lw=2, color=color2, zorder=11)
    ax2ins.plot(cemsp90_byeducation, ls='-', lw=2, color=color3, zorder=11)
    ax2ins.plot(cemsp90_byethnicity, ls='-', lw=2, color=color4, zorder=11)
    ax2ins.plot(cemsp90_byvehicle, ls='-', lw=2, color=color5, zorder=11)   
    ax2ins.set_xlim([0,9])
    ax2ins.set_xticks(np.arange(0,10,1))
    ax2ins.set_xticklabels([])
    ax2ins.set_ylim([0,0.0015])
    ax2ins.set_yticks(np.linspace(0,0.0018,5))
    ax2ins.set_yticklabels(['0', '', '9', '', 
        '18$\:$x$\:$10$^{\mathregular{-4}}$'])
    # Legend
    ax1.text(0.5, 0.92, 'Income', fontsize=12, va='center',
        color=color1, ha='center', transform=ax1.transAxes)
    ax1.annotate('Higher',xy=(0.58,0.92),xytext=(0.78,0.92), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color1), 
        fontsize=12, color=color1, xycoords=ax1.transAxes)
    ax1.annotate('Lower',xy=(0.41,0.92),xytext=(0.1,0.92), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color1), 
        fontsize=12, color=color1, xycoords=ax1.transAxes)
    ax1.text(0.5, 0.84, 'Education', fontsize=12, color=color2, va='center', 
        ha='center', transform=ax1.transAxes)
    ax1.annotate('More',xy=(0.61,0.84),xytext=(0.78,0.84), va='center', 
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color2), fontsize=12,
        color=color2, xycoords=ax1.transAxes)
    ax1.annotate('Less',xy=(0.38,0.84),xytext=(0.1,0.84), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color2), 
        fontsize=12, color=color2, xycoords=ax1.transAxes)
    ax1.text(0.5, 0.76, 'White', fontsize=12, va='center',
        color=color3, ha='center', transform=ax1.transAxes)
    ax1.annotate('More',xy=(0.57,0.76),xytext=(0.78,0.76), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color3), color=color3,
        fontsize=12, xycoords=ax1.transAxes)
    ax1.annotate('Less',xy=(0.43,0.76),xytext=(0.1,0.76), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color3), fontsize=12, 
        color=color3, xycoords=ax1.transAxes)
    ax1.text(0.5, 0.68, 'Hispanic', fontsize=12, 
        color=color4, ha='center', va='center', transform=ax1.transAxes)
    ax1.annotate('Less',xy=(0.59,0.68),xytext=(0.78,0.68), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color4), fontsize=12, 
        color=color4, xycoords=ax1.transAxes)
    ax1.annotate('More',xy=(0.4,0.68),xytext=(0.1,0.68), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color4), fontsize=12,
        color=color4, xycoords=ax1.transAxes)
    ax1.text(0.5, 0.6, 'Vehicle ownership', fontsize=12, ha='center',
        va='center', color=color5, transform=ax1.transAxes)
    ax1.annotate('More',xy=(0.65,0.6),xytext=(0.78,0.6), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color5), fontsize=12, 
        color=color5, xycoords=ax1.transAxes)
    ax1.annotate('Less',xy=(0.34,0.6),xytext=(0.1,0.6), va='center',
        arrowprops=dict(arrowstyle= '<|-', lw=1, color=color5), fontsize=12,
        color=color5, xycoords=ax1.transAxes)
    for ax in axes:
        ax.set_xlim([0,9])
        ax.set_xticks(np.arange(0,10,1))
        ax.set_xticklabels([])
    for ax in [ax3, ax4]:
        ax.set_xticklabels(['First', 'Second', 'Third', 'Fourth', 'Fifth', 
            'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth'], fontsize=10)
        ax.set_xlabel('Decile', fontsize=12)
    ax1.set_ylim([0,0.01])
    ax1.set_yticks(np.linspace(0,0.01,9))
    ax1.set_yticklabels(['0','', '2.5', '', '5.0', '', '7.5', '',
        '10$\:$x$\:$10$^{\mathregular{-2}}$'])
    ax2.set_ylim([0.0, 0.04])
    ax2.set_yticks(np.linspace(0,0.04,9))
    ax2.set_yticklabels(['0', '', '1', '', '2', '', '3', '', 
        '4$\:$x$\:$10$^{\mathregular{-2}}$'])
    ax3.set_ylim([0,0.006])
    ax3.set_yticks(np.linspace(0,0.006,9))
    ax3.set_yticklabels(['0', '', '1.5' , '', '3.0' , '', '4.5' , '', 
        '6.0$\:$x$\:$10$^{\mathregular{-3}}$'])
    ax4.set_ylim([0,4])
    ax4.set_yticks(np.linspace(0,4,9))
    ax4.set_yticklabels(['0','','1','','2','','3','','4'])
    ax1.set_title('(a) Port density [ports (1 km radius)$^{-1}$]', 
        fontsize=12, loc='left')
    ax2.set_title('(b) Industry density [industries (1 km radius)$^{-1}$]', 
        fontsize=12, loc='left')
    ax3.set_title('(c) Airport density [airports (1 km radius)$^{-1}$]', 
        fontsize=12, loc='left')
    ax4.set_title('(d) Railroad density [railroads (1 km radius)$^{-1}$]', 
        fontsize=12, loc='left')
    plt.subplots_adjust(left=0.08, right=0.95)
    plt.savefig(DIR_FIGS+'figS6_revised.pdf', dpi=1000)
    return

def figS7():   
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.patches as mpatches
    nei = '/Users/ghkerr/GW/data/emissions/2017neiJan_county_tribe_allsector/'+\
        'esg_cty_sector_15468.csv'
    nei = pd.read_csv(nei, delimiter=',', engine='python')
    nei = nei.loc[nei['pollutant code']=='NOX']
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
    # Plotting
    fig = plt.figure(figsize=(7,7))
    axa = plt.subplot2grid((4,4),(0,0))
    axb = plt.subplot2grid((4,4),(0,1))
    axc = plt.subplot2grid((4,4),(0,2))
    axd = plt.subplot2grid((4,4),(0,3))
    axe = plt.subplot2grid((4,4),(1,0))
    axf = plt.subplot2grid((4,4),(1,1))
    axg = plt.subplot2grid((4,4),(1,2))
    axh = plt.subplot2grid((4,4),(1,3))
    axi = plt.subplot2grid((4,4),(2,0))
    axj = plt.subplot2grid((4,4),(2,1))
    axk = plt.subplot2grid((4,4),(2,2))
    axl = plt.subplot2grid((4,4),(2,3))
    axm = plt.subplot2grid((4,4),(3,0))
    axn = plt.subplot2grid((4,4),(3,1))
    axo = plt.subplot2grid((4,4),(3,2))
    color_ld = '#a6cee3'
    color_hd = '#1f78b4'
    color_nr = '#b2df8a'
    color_fc = '#33a02c'
    color_ma = '#fb9a99'
    color_lo = '#e31a1c'
    color_ac = '#fdbf6f'
    axes = [axa, axb, axc, axd, axe, axf, axg, axh, axi, axj, axk, axl,
        axm, axn, axo]
    citynames = ['New York', 'Los Angeles', 'Chicago', 'Dallas', 'Houston', 
        'Washington', 'Miami', 'Philadelphia', 'Atlanta', 'Phoenix', 
        'Boston', 'San Francisco', 'Riverside', 'Detroit', 'Seattle']  
    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)',
        '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)']
    i = 0
    for city in [newyork, losangeles, chicago, dallas, houston, washington,
        miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
        riverside, detroit, seattle]:
        # Select MSA and sum over individual counties in city
        nei_city = nei.loc[nei['fips code'].isin(city)]
        nei_city = nei_city.groupby(['sector']).sum()
        # Find fractions 
        nei_city['total emissions'] = (nei_city['total emissions']/
            nei_city['total emissions'].sum()).copy()
        nei_city = nei_city.sort_values(by='total emissions', axis=0, 
            ascending=False).reset_index()
        # Mobile - On-Road non-Diesel Light Duty Vehicles
        ld = nei_city.loc[nei_city['sector']==
            'Mobile - On-Road non-Diesel Light Duty Vehicles']['total emissions'
            ].values[0]
        # Mobile - On-Road Diesel Heavy Duty Vehicles
        hd = nei_city.loc[nei_city['sector']==
            'Mobile - On-Road Diesel Heavy Duty Vehicles']['total emissions'
            ].values[0]
        # Mobile - Non-Road Equipment - Diesel
        nr = nei_city.loc[nei_city['sector']==
            'Mobile - Non-Road Equipment - Diesel']['total emissions'
            ].values[0]
        # Fuel Comb
        fc = nei_city.loc[nei_city['sector'].str.startswith('Fuel Comb', 
            na=False)]['total emissions'].sum()
        # Mobile - Commercial Marine Vessels
        try: 
            ma = nei_city.loc[nei_city['sector']==
                'Mobile - Commercial Marine Vessels']['total emissions'
                ].values[0] 
        except IndexError:
            ma = 0.
        # Mobile - Locomotives
        lo = nei_city.loc[nei_city['sector']=='Mobile - Locomotives'][
            'total emissions'].values[0]
        # Mobile - Aircraft
        ac = nei_city.loc[nei_city['sector']=='Mobile - Aircraft'][
            'total emissions'].values[0]    
        ax = axes[i]
        ax.bar(0.00, ld, color=color_ld, width = 0.25)
        ax.bar(0.25, hd, color=color_hd, width = 0.25)
        ax.bar(0.50, nr, color=color_nr, width = 0.25)
        ax.bar(0.75, fc, color=color_fc, width = 0.25)
        ax.bar(1.00, ma, color=color_ma, width = 0.25)
        ax.bar(1.25, lo, color=color_lo, width = 0.25)
        ax.bar(1.50, ac, color=color_ac, width = 0.25)
        ax.set_title(letters[i]+' '+citynames[i], loc='left', y=1.03)
        ax.set_xlim([-0.25,1.75])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylim([0,0.4])
        ax.set_yticks([0,0.1,0.2,0.3,0.4])
        ax.set_yticklabels([])
        # Remove right and top spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()    
        i = i+1
    for ax in [axa, axe, axi, axm]:    
        ax.set_yticklabels(['0','','20','','40'])
        ax.set_ylabel('Contribution [%]', fontsize=12)
    # Add legend
    patch_ld = mpatches.Patch(color=color_ld, label='Light-duty non-diesel')
    patch_hd = mpatches.Patch(color=color_hd, label='Heavy-duty diesel')
    patch_nr = mpatches.Patch(color=color_nr, label='Non-road diesel')
    patch_fc = mpatches.Patch(color=color_fc, label='Ind., res., and comm.')
    patch_ma = mpatches.Patch(color=color_ma, label='Commercial marine')
    patch_lo = mpatches.Patch(color=color_lo, label='Rail')
    patch_ac = mpatches.Patch(color=color_ac, label='Aviation')
    plt.legend(handles=[patch_ld, patch_hd, patch_nr, patch_fc, patch_ma, 
        patch_lo, patch_ac], bbox_to_anchor=(1.15, 1.2), ncol=1,
        frameon=False)
    plt.subplots_adjust(top=0.92, right=0.93, left=0.08, bottom=0.06, 
        hspace=0.35)
    plt.savefig(DIR_FIGS+'figS7_revised.pdf', dpi=1000)
    return

def figS8():
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    nei = DIR+'data/emissions/2017neiJan_county_tribe_allsector/'+\
        'esg_cty_sector_15468.csv'
    nei = pd.read_csv(nei, delimiter=',', engine='python')
    nei = nei.loc[nei['pollutant code']=='NOX']
    # Urban-rural lookup
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
    for x in np.arange(0, len(nei), 1):    
        GEOID_statecountyonly = str(nei['fips code'].values[x])
        if GEOID_statecountyonly in urbancounties:
            urbantracts.append(nei['fips code'].iloc[x])
    # Main NOx emitters for urban areas
    nei_urban = nei.loc[nei['fips code'].isin(urbantracts)]
    nei_urban = nei_urban.groupby(['sector']).sum()
    # Find fractions 
    nei_urban['total emissions'] = (nei_urban['total emissions']/
        nei_urban['total emissions'].sum()).copy()
    nei_urban = nei_urban.sort_values(by='total emissions', axis=0, 
        ascending=False).reset_index()
    # Mobile - On-Road non-Diesel Light Duty Vehicles
    ld = nei_urban.loc[nei_urban['sector']==
        'Mobile - On-Road non-Diesel Light Duty Vehicles']['total emissions'
        ].values[0]
    # Mobile - On-Road Diesel Heavy Duty Vehicles
    hd = nei_urban.loc[nei_urban['sector']==
        'Mobile - On-Road Diesel Heavy Duty Vehicles']['total emissions'
        ].values[0]
    # Mobile - Non-Road Equipment - Diesel
    nr = nei_urban.loc[nei_urban['sector']==
        'Mobile - Non-Road Equipment - Diesel']['total emissions'
        ].values[0]
    # Fuel Comb
    fc = nei_urban.loc[nei_urban['sector'].str.startswith('Fuel Comb', 
        na=False)]['total emissions'].sum()
    # Mobile - Commercial Marine Vessels
    ma = nei_urban.loc[nei_urban['sector']==
        'Mobile - Commercial Marine Vessels']['total emissions'
        ].values[0] 
    # Mobile - Locomotives
    lo = nei_urban.loc[nei_urban['sector']=='Mobile - Locomotives'][
        'total emissions'].values[0]
    # Mobile - Aircraft
    ac = nei_urban.loc[nei_urban['sector']=='Mobile - Aircraft'][
        'total emissions'].values[0]
    # Rail and aviation changes
    # https://www.bts.gov/covid-19/week-in-transportation#aviation
    # Units of daily total flights
    plane = pd.read_csv('/Users/ghkerr/GW/data/mobility/'+
        'US_Commercial_Flights__TOTAL_data.csv', sep='\t', encoding='utf-16')
    plane['Week of Date'] = [datetime.strptime(s, '%B %d, %Y') for s in 
        plane['Week of Date']]
    plane.set_index('Week of Date', inplace=True)
    plane = plane.loc['2020-03-01':'2020-07-30']
    plane = plane.loc[plane['Measure Names']=='Last Year']
    # Values relative to the first week of March 
    plane['frac_change'] = plane['Current']/plane['Current'].values[0]
    # Rail data (https://ycharts.com/indicators/us_carloads_weekly_rail_traffic) 
    rail = pd.DataFrame([['February 15, 2020', 227447.0],
        ['February 22, 2020', 232869.0],
        ['February 29, 2020', 234652.0],
        ['March 07, 2020', 229742.0],
        ['March 14, 2020', 226039.0],
        ['March 21, 2020', 224048.0],
        ['March 28, 2020', 219844.0],
        ['April 04, 2020', 210911.0],
        ['April 11, 2020', 198726.0],
        ['April 18, 2020', 189598.0],
        ['April 25, 2020', 192110.0],
        ['May 02, 2020', 189190.0],
        ['May 09, 2020', 185144.0],
        ['May 16, 2020', 184415.0],
        ['May 23, 2020', 190639.0],
        ['May 30, 2020', 179973.0],
        ['June 06, 2020', 192494.0],
        ['June 13, 2020', 198437.0],
        ['June 20, 2020', 201823.0],
        ['June 27, 2020', 201502.0],
        ['July 04, 2020', 192767.0],
        ['July 11, 2020', 201703.0],
        ['July 18, 2020', 214685.0],
        ['July 25, 2020', 215171.0]], columns=['Date','Activity'])
    rail['Date'] = [datetime.strptime(s, '%B %d, %Y') for s in rail['Date']]
    rail.set_index('Date', inplace=True)
    rail = rail.loc['2020-02-29':'2020-07-01']
    # Values relative to the first week of March 
    rail['frac_change'] = rail['Activity']/rail['Activity'].values[0]
    # Traffic (click "Show as table" and transcribe)
    # https://app.powerbi.com/view?r=eyJrIjoiZmQzZDhmZTctMDNjYS00NGNmLWJ
    # lNzctZTE5NzRlYTk5NGNjIiwidCI6IjZhZDJlNGRhLThjOTItNGU1OC04ODc3LWVkM
    # DZiODkxODM3OSIsImMiOjZ9
    # Units of normalized trip count
    traffic = pd.read_csv('/Users/ghkerr/GW/data/mobility/'+
        'bts_transport_change.csv', sep=',', engine='python')
    traffic['Date'] = pd.to_datetime(traffic['Date'])
    traffic.set_index('Date', inplace=True)
    traffic_wkly = traffic.resample('7D').mean()
    # Industrial emissions (go to "Query" at https://ampd.epa.gov/ampd/ and 
    # select "All Programs" and "Emissions" then "Daily" for the desired date
    # range and then all states besides Hawaii and Alaska and then 
    # "National" for the aggregation.)
    # Units of tons
    cems = pd.read_csv('/Users/ghkerr/GW/data/cems/2020_dailyavg/'+
        'emission_02-15-2021_191910233.csv', sep=',', engine='python')
    cems.index = pd.to_datetime(cems.index)
    cems.sort_index(inplace=True)
    # Select 2020 data
    cems_2020 = cems.loc['2020-03-01':'2020-07-30']
    cems_2020['frac_change'] = cems_2020[' Year']/cems_2020[' Year'].values[0]
    cems_wkly = cems_2020.resample('7D').mean()
    # Port calls (for Canada, Mexico, and the U.S.) from https://
    # public.tableau.com/profile/uncomtrade#!/vizhome/AISPortCalls2/AISMonitor
    # (Zoom in to the timeseries and then select all points, right click on 
    # graph, and click on the table icon)
    port = pd.read_csv('/Users/ghkerr/GW/data/mobility/'+
        'Global__Regional_Port_Calls_Full_Data_data.csv', sep='\t', 
        encoding='utf-16')
    # Select port calls in Northern America
    port = port.loc[(port['Sub-region Name']=='Northern America')]
    port['Date'] = [datetime.strptime(s, '%B %d, %Y') for s in 
        port['Day of Date-Entry']]
    port = port.groupby(['Date']).sum()
    port = port[['Port Calls']]
    port = port.loc['2020-03-01':'2020-07-30']
    port_wkly = port.resample('7D').mean()
    port_wkly['frac_change'] = port_wkly['Port Calls']/port_wkly['Port Calls'].values[0]
    fig = plt.figure(figsize=(9,5))
    axb = plt.subplot2grid((1,1),(0,0))
    # Define colors 
    color_ld = '#33a02c'
    color_hd = '#1f78b4'
    color_nr = '#b2df8a'
    color_fc = '#a6cee3'
    color_ma = '#fb9a99'
    color_lo = '#e31a1c'
    color_ac = '#fdbf6f'
    # Plot nationally-averaged values
    axb.plot(traffic_wkly['Long-Haul Trucks']*hd, ls='-', lw=2, color=color_hd)
    axb.plot(traffic_wkly['Passenger']*ld, ls='-', lw=2, color=color_ld)
    axb.plot(plane['frac_change']*ac, ls='-', lw=2, color=color_ac)
    axb.plot(cems_wkly['frac_change']*fc, ls='-', lw=2, color=color_fc)
    axb.plot(rail['frac_change']*lo, ls='-', lw=2, color=color_lo)
    axb.plot(port_wkly['frac_change']*ma, ls='-', lw=2, color=color_ma)
    axb.set_xlim(['2020-03-01','2020-06-13'])
    axb.set_ylim([0,0.32])
    axb.set_yticks(np.linspace(0,0.32,5))
    axb.set_yticklabels(['0','8','16','24','32'])
    axb.set_ylabel(r'Fractional change $\mathregular{\times}$ Contribution [%]')
    # Add legend
    patch_ld = mpatches.Patch(color=color_ld, label='Light-duty non-diesel')
    patch_hd = mpatches.Patch(color=color_hd, label='Heavy-duty diesel')
    patch_fc = mpatches.Patch(color=color_fc, label='Ind., res., and comm.')
    patch_ma = mpatches.Patch(color=color_ma, label='Commercial Marine')
    patch_lo = mpatches.Patch(color=color_lo, label='Rail')
    patch_ac = mpatches.Patch(color=color_ac, label='Aviation')
    plt.subplots_adjust(top=0.95, right=0.98, left=0.08, bottom=0.18, hspace=0.7)
    plt.legend(handles=[patch_ld, patch_hd, patch_fc, patch_ma, 
        patch_lo, patch_ac], bbox_to_anchor=(0.62, 0.85), ncol=3,
        frameon=False)
    plt.savefig(DIR_FIGS+'figS8_revised.pdf', dpi=1000)
    return 

def figS9(harmonized_urban):
    """
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    # NHGIS census information version (may need to change if there are 
    # updates to census information)
    nhgis_version = '0003_ds239_20185_2018'
    # DataFrame that will be filled with harmonzied data for multiple states
    state_harm = pd.DataFrame()
    # Loop through states of interest and read in harmonized NO2/census data
    for FIPS_i in FIPS:
        state_harm_i = pd.read_csv(DIR_HARM+'v3/'+
            'Tropomi_NO2_interpolated_%s_nhgis%s_tract.csv'%(FIPS_i, 
            nhgis_version), delimiter=',', header=0, engine='python')
        # For states with FIPS codes 0-9, there is no leading zero in their 
        # GEOID row, so add one such that all GEOIDs for any particular state
        # are identical in length
        if FIPS_i in ['01','02','03','04','05','06','07','08','09']:
            state_harm_i['GEOID'] = state_harm_i['GEOID'].map(
            lambda x: f'{x:0>11}')
        # Make GEOID a string and index row 
        state_harm_i = state_harm_i.set_index('GEOID')
        state_harm_i.index = state_harm_i.index.map(str)
        # Make relevant columns floats
        col_relevant = ['PRENO2', 'POSTNO2', 'PRENO2APR', 'POSTNO2APR', 'ALLNO2']
        state_harm_i = state_harm_i[col_relevant]
        for col in col_relevant:
            state_harm_i[col] = state_harm_i.loc[:,col].astype(float)    
        state_harm = state_harm.append(state_harm_i)    
    # Only for urban
    harmonized_at = harmonized_urban.merge(state_harm, left_index=True, 
        right_index=True)
    # Fraction of White population in tract
    white = (harmonized_at['AJWNE002']/harmonized_at['AJWBE001'])
    mostwhite = harmonized_at.iloc[np.where(white > 
        np.nanpercentile(white, ptile_upper))]
    leastwhite = harmonized_at.iloc[np.where(white < 
        np.nanpercentile(white, ptile_lower))]
    # Fraction of population in tract with some college to a Doctorate degree
    education = (harmonized_at.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
        harmonized_at['AJYPE001'])
    mosteducated = harmonized_at.iloc[np.where(education > 
        np.nanpercentile(education, ptile_upper))]
    leasteducated = harmonized_at.iloc[np.where(education < 
        np.nanpercentile(education, ptile_lower))]
    # Fraction of non-Hispanic  or Latino population in tract
    ethnicity = (harmonized_at['AJWWE002']/harmonized_at['AJWWE001'])
    leastethnic = harmonized_at.iloc[np.where(ethnicity > 
        np.nanpercentile(ethnicity, ptile_upper))]
    mostethnic = harmonized_at.iloc[np.where(ethnicity < 
        np.nanpercentile(ethnicity, ptile_lower))]
    # Fraction of population without a vehicle
    vehicle = 1-harmonized_at['FracNoCar']
    mostvehicle = harmonized_at.iloc[np.where(vehicle > 
        np.nanpercentile(vehicle, ptile_upper))]
    leastvehicle = harmonized_at.iloc[np.where(vehicle < 
        np.nanpercentile(vehicle, ptile_lower))]
    # Highest vs. lowest income tracts
    income = harmonized_at['AJZAE001']
    highincome = harmonized_at.iloc[np.where(income > 
        np.nanpercentile(income, ptile_upper))]
    lowincome = harmonized_at.iloc[np.where(income < 
        np.nanpercentile(income, ptile_lower))]
    fig = plt.figure(figsize=(7,7))
    ax1 = plt.subplot2grid((3,2),(0,0))
    ax2 = plt.subplot2grid((3,2),(1,0))
    ax3 = plt.subplot2grid((3,2),(2,0))
    ax4 = plt.subplot2grid((3,2),(0,1))
    ax5 = plt.subplot2grid((3,2),(1,1))
    ax6 = plt.subplot2grid((3,2),(2,1))
    lefts = [highincome, mostwhite, leastethnic, mosteducated, mostvehicle]
    rights = [lowincome, leastwhite, mostethnic, leasteducated, leastvehicle]
    axes = [ax1, ax2, ax3, ax4, ax5]
    for ax, left, right in zip(axes, lefts, rights): 
        bpl = ax.boxplot([left['ALLNO2_x'].values[~np.isnan(left['ALLNO2_x'].values)], 
            left['PRENO2APR_x'].values[~np.isnan(left['PRENO2APR_x'].values)],
            left['PRENO2_x'][~np.isnan(left['PRENO2_x'])]], positions=[1,2,3],
            whis=[20,80], showfliers=False, patch_artist=True, showcaps=False)
        bpr = ax.boxplot([right['ALLNO2_x'].values[~np.isnan(right['ALLNO2_x'].values)], 
            right['PRENO2APR_x'].values[~np.isnan(right['PRENO2APR_x'].values)],
            right['PRENO2_x'][~np.isnan(right['PRENO2_x'])]], positions=[5,6,7],
            whis=[20,80], showfliers=False, patch_artist=True, showcaps=False)    
        for bp in [bpl, bpr]:
            for element in ['boxes']:
                plt.setp(bp[element][0], color='#0095A8')
                plt.setp(bp[element][1], color='darkgrey')    
                plt.setp(bp[element][2], color='#FF7043')
            for element in ['whiskers']:
                plt.setp(bp[element][0], color='#0095A8', linewidth=2)
                plt.setp(bp[element][1], color='#0095A8', linewidth=2)
                plt.setp(bp[element][2], color='darkgrey', linewidth=2)
                plt.setp(bp[element][3], color='darkgrey', linewidth=2)
                plt.setp(bp[element][4], color='#FF7043', linewidth=2)
                plt.setp(bp[element][5], color='#FF7043', linewidth=2)
            for element in ['medians']:
                plt.setp(bp[element], color='w', linewidth=2) 
        ax.set_xlim([0,8])
        ax.set_xticks([2,6])
        ax.set_ylim([0.1e16,1.2e16])
    # Labels 
    ax1.set_title('(a) Median household income', loc='left', fontsize=10)
    ax2.set_title('(b) Racial background', loc='left', fontsize=10)
    ax3.set_title('(c) Ethnic background', loc='left', fontsize=10)
    ax4.set_title('(d) Educational attainment',loc='left', fontsize=10)
    ax5.set_title('(e) Household vehicle ownership',loc='left', fontsize=10)
    ax1.set_xticklabels(['Highest income', 'Lowest income'], y=0.025)
    ax2.set_xticklabels(['Most White', 'Least White'], y=0.025)
    ax3.set_xticklabels(['Most Hispanic','Least Hispanic'], y=0.025)
    ax4.set_xticklabels(['Most educated','Least educated'], y=0.025)
    ax5.set_xticklabels(['Highest vehicle\nownership',
        'Lowest vehicle\nownership'], y=0.025)
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.yaxis.offsetText.set_visible(False)
        ax.set_yticks([2.5e15, 5e15, 7.5e15, 10e15])
        ax.set_yticklabels(['2.5','5.0','7.5','10'])    
        ax.tick_params(axis='x', which='both', bottom=False)
    ax4.set_yticklabels([])
    ax5.set_yticklabels([])    
    ax1.set_ylabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', fontsize=10)
    ax2.set_ylabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', fontsize=10)
    ax3.set_ylabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]', fontsize=10)
    # Create fake legend
    np.random.seed(10)
    collectn_1 = np.random.normal(100, 10, 200)    
    leg = ax6.boxplot([collectn_1, collectn_1, collectn_1], 
        positions=[0.5,2,3.5], whis=[10,90], vert=0, showfliers=False, patch_artist=True, 
        showcaps=False)
    for element in ['boxes']:
        plt.setp(leg[element][0], color='#0095A8')
        plt.setp(leg[element][1], color='darkgrey')    
        plt.setp(leg[element][2], color='#FF7043')
    for element in ['whiskers']:
        plt.setp(leg[element][0], color='#0095A8', linewidth=2)
        plt.setp(leg[element][1], color='#0095A8', linewidth=2)
        plt.setp(leg[element][2], color='darkgrey', linewidth=2)
        plt.setp(leg[element][3], color='darkgrey', linewidth=2)
        plt.setp(leg[element][4], color='#FF7043', linewidth=2)
        plt.setp(leg[element][5], color='#FF7043', linewidth=2)
    for element in ['medians']:
        plt.setp(leg[element], color='w', linewidth=2)
    ax6.set_xlim([65,115])
    ax6.set_ylim([-1,5])
    ax6.annotate('April 1 -\nJune 30, 2019', ha='right', xy=(86,3))
    ax6.annotate('March 13 -\nJune 13, 2019', ha='right', xy=(86,1.5))
    ax6.annotate('May 1, 2018 -\nDecember 31, 2019', ha='right', xy=(86,0.))
    ax6.axis('off')
    plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05)
    plt.savefig(DIR_FIGS+'figS9_revised.pdf', dpi=1000)
    return

def figS10(FIPS, harmonized_urban, harmonized_rural):
    """    
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from cartopy.io import shapereader
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
    geoids = []
    areas = []
    areas_urban = []
    areas_rural = []
    areas_newyork = []
    areas_losangeles = []
    areas_chicago = []
    areas_dallas = []
    areas_houston = []
    areas_washington = []
    areas_miami = []
    areas_philadelphia = []
    areas_atlanta = []
    areas_phoenix = []
    areas_boston = []
    areas_sanfrancisco = []
    areas_riverside = []
    areas_detroit = []
    areas_seattle = []
    for FIPS_i in FIPS: 
        print(FIPS_i)
        # Tigerline shapefile for state
        shp = shapereader.Reader(DIR_GEO+
            'tigerline/tl_2019_%s_tract/tl_2019_%s_tract.shp'%(FIPS_i, FIPS_i))
        records = shp.records()
        tracts = shp.geometries()
        tracts = list(tracts)
        records = list(records)
        for r in records:
            # Add ALAND and AWATER
            areas.append(r.attributes['ALAND']+r.attributes['AWATER'])
            geoids.append(r.attributes['GEOID'])
            if r.attributes['GEOID'] in harmonized_urban.index:
                areas_urban.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])
            if r.attributes['GEOID'] in harmonized_rural.index:            
                areas_rural.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])
            if r.attributes['GEOID'][:5] in newyork:
                areas_newyork.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])            
            if r.attributes['GEOID'][:5] in losangeles:          
                areas_losangeles.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])
            if r.attributes['GEOID'][:5] in chicago:          
                areas_chicago.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])
            if r.attributes['GEOID'][:5] in dallas:          
                areas_dallas.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])
            if r.attributes['GEOID'][:5] in houston:          
                areas_houston.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])
            if r.attributes['GEOID'][:5] in washington:          
                areas_washington.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])
            if r.attributes['GEOID'][:5] in miami:          
                areas_miami.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])
            if r.attributes['GEOID'][:5] in philadelphia:          
                areas_philadelphia.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])
            if r.attributes['GEOID'][:5] in atlanta:          
                areas_atlanta.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])
            if r.attributes['GEOID'][:5] in phoenix:          
                areas_phoenix.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])
            if r.attributes['GEOID'][:5] in boston:          
                areas_boston.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])
            if r.attributes['GEOID'][:5] in sanfrancisco:          
                areas_sanfrancisco.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])            
            if r.attributes['GEOID'][:5] in riverside:          
                areas_riverside.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])            
            if r.attributes['GEOID'][:5] in detroit:          
                areas_detroit.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])            
            if r.attributes['GEOID'][:5] in seattle:          
                areas_seattle.append(r.attributes['ALAND']+
                    r.attributes['AWATER'])            
    fig = plt.figure(figsize=(5,7))
    axr = plt.subplot2grid((18,1), (0,0), rowspan=3)
    ax = plt.subplot2grid((18,1), (3,0), rowspan=15)
    scaler = 1000*1000.
    areas_allr = [np.array(areas)/scaler, 
        np.array(areas_rural)/scaler]
    areas_all = [np.array(areas_urban)/scaler, 
        np.array(areas_newyork)/scaler, 
        np.array(areas_losangeles)/scaler,
        np.array(areas_chicago)/scaler, 
        np.array(areas_dallas)/scaler, 
        np.array(areas_houston)/scaler, 
        np.array(areas_washington)/scaler, 
        np.array(areas_miami)/scaler, 
        np.array(areas_philadelphia)/scaler, 
        np.array(areas_atlanta)/scaler, 
        np.array(areas_phoenix)/scaler, 
        np.array(areas_boston)/scaler, 
        np.array(areas_sanfrancisco)/scaler, 
        np.array(areas_riverside)/scaler, 
        np.array(areas_detroit)/scaler, 
        np.array(areas_seattle)/scaler]
    axr.boxplot(areas_allr, showfliers=False, vert=0, whis=[20, 80], 
        patch_artist=True, boxprops=dict(facecolor='w', color='k'), 
        medianprops=dict(color='k'), widths=(0.45), zorder=10)            
    ax.boxplot(areas_all, showfliers=False, vert=0, whis=[20, 80], 
        patch_artist=True, boxprops=dict(facecolor='w', color='k'), 
        medianprops=dict(color='k'), zorder=10)
    citynames = [r'$\bf{All}$', r'$\bf{Rural}$', r'$\bf{Urban}$',
        'New York', 'Los Angeles', 'Chicago', 'Dallas', 'Houston', 
        'Washington', 'Miami', 'Philadelphia', 'Atlanta', 'Phoenix', 
        'Boston', 'San Francisco', 'Riverside', 'Detroit', 'Seattle']   
    axr.set_ylim([0.3,2.6])
    axr.set_yticks(np.arange(1,3,1))
    axr.set_yticklabels(citynames[:2])
    ax.set_yticks(np.arange(1,17,1))
    ax.set_yticklabels(citynames[2:])
    for axi in [axr, ax]:
        axi.axvline(x=1, lw=1, ls='--', color='k')
        axi.invert_yaxis()
        axi.tick_params(axis='y', left=False, length=0)
        axi.tick_params(axis='x', colors='darkgrey')
        for side in ['right', 'left', 'top', 'bottom']:
            axi.spines[side].set_visible(False)
        axi.grid(axis='x', zorder=2, color='darkgrey')    
    # X axis limits    
    axr.set_xlim([-1,201])
    axr.set_xticks(np.linspace(0,200,6))
    ax.set_xlim([-.1,25.1])
    ax.set_xticks(np.linspace(0,25,6))
    ax.set_xlabel('Tract area [km$^{2}$]', x=0.2, color='darkgrey')
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.2, hspace=6.)
    plt.savefig(DIR_FIGS+'figS10_revised.pdf', dpi=1000)
    return

def figS11(harmonized):
    """
    """
    import numpy as np
    from scipy.stats import ks_2samp
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5,8))
    ax1 = plt.subplot2grid((2,1),(0,0))
    ax2 = plt.subplot2grid((2,1),(1,0))
    color_non = '#0095A8'
    color_margin = '#FF7043'
    bins = 60
    # # # # 
    # Illustrate a case with significant differences (baseline NO2 for all 
    # census tracts for race)
    frac_white = ((harmonized['AJWNE002'])/harmonized['AJWBE001'])
    mostwhite = harmonized.iloc[np.where(frac_white > 
        np.nanpercentile(frac_white, ptile_upper))]
    leastwhite = harmonized.iloc[np.where(frac_white < 
        np.nanpercentile(frac_white, ptile_lower))]
    # Histograms
    ax1.hist(mostwhite['PRENO2'], bins, facecolor=color_non, edgecolor='None', 
        label='Most white', alpha=0.9)
    ax1.hist(leastwhite['PRENO2'], bins, facecolor=color_margin, edgecolor=
        'None', label='Least white', alpha=0.9)
    ax1.legend(frameon=False, ncol=2)
    # Inset axes for ECDF over the main axes
    inset_axes1 = inset_axes(ax1, width="40%",
        height=1.0, # height : 1 inch
        loc=7)
    counts1, bin_edges1 = np.histogram(mostwhite['PRENO2'], bins=bins, 
        normed=True)
    cdf1 = np.cumsum(counts1)
    counts2, bin_edges2 = np.histogram(leastwhite['PRENO2'], bins=bins, 
        normed=True)
    cdf2 = np.cumsum(counts2)
    inset_axes1.plot(bin_edges1[1:], cdf1/cdf1[-1], ls='--', color=color_non)
    inset_axes1.plot(bin_edges2[1:], cdf2/cdf2[-1], ls='--', color=color_margin)
    inset_axes1.annotate('', xy=(bin_edges1[9], 0.18), xycoords='data',
        xytext=(bin_edges1[9], 0.78), textcoords='data', arrowprops={
        'arrowstyle': '<->'})
    inset_axes1.text(bin_edges1[16], 0.1, 
        'D=0.6\n$\mathregular{\mathit{p}}$<0.001')
    # ks_2samp(mostwhite['PRENO2'], leastwhite['PRENO2'])
    # # # # 
    # Illustrate a case with no significant differences (baseline NO2 for all 
    # census tracts for income)
    mostwealthy = harmonized.loc[harmonized['AJZAE001'] > 
        np.nanpercentile(harmonized['AJZAE001'], 90)]
    leastwealthy = harmonized.loc[harmonized['AJZAE001'] < 
        np.nanpercentile(harmonized['AJZAE001'], 10)]
    # Histograms
    ax2.hist(mostwealthy['PRENO2'], bins, facecolor=color_non, edgecolor='None', 
        label='Highest income', zorder=20, alpha=0.9)
    ax2.hist(leastwealthy['PRENO2'], bins, facecolor=color_margin, edgecolor=
        'None', label='Lowest income', alpha=0.9)
    ax2.legend(frameon=False, ncol=2)
    inset_axes2 = inset_axes(ax2, width="40%", # width = 30% of parent_bbox
        height=1.0, # height : 1 inch
        loc=7)
    counts1, bin_edges1 = np.histogram(mostwealthy['PRENO2'], bins=bins, 
        normed=True)
    cdf1 = np.cumsum(counts1)
    counts2, bin_edges2 = np.histogram(leastwealthy['PRENO2'], bins=bins, 
        normed=True)
    cdf2 = np.cumsum(counts2)
    inset_axes2.plot(bin_edges1[1:], cdf1/cdf1[-1], ls='--', color=color_non)
    inset_axes2.plot(bin_edges2[1:], cdf2/cdf2[-1], ls='--', color=color_margin)
    inset_axes2.annotate('', xy=(bin_edges1[9], 0.2), xycoords='data',
        xytext=(bin_edges1[9], 0.52), textcoords='data', arrowprops={
        'arrowstyle': '<->'})
    inset_axes2.text(bin_edges1[16], 0.21, 
        'D=0.3\n$\mathregular{\mathit{p}}$=0.6')
    # ks_2samp(mostwealthy['PRENO2'], leastwealthy['PRENO2'])
    # Axis labels, limits
    ax1.set_title('(a) Significant difference (All tracts, baseline, Figure 2a)', 
        loc='left', fontsize=10)
    ax2.set_title('(b) No significant difference (All tracts, baseline, '+
        'Figure 2b)', loc='left', fontsize=10)
    ax1.set_xlim([0, 1.2e16])
    ax1.set_xticks([0, 0.3e16, 0.6e16, 0.9e16, 1.2e16])
    ax1.set_xlabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]')
    ax1.xaxis.offsetText.set_visible(False)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax2.set_xlim([0, 1.2e16])
    ax2.set_xticks([0, 0.3e16, 0.6e16, 0.9e16, 1.2e16])
    ax2.set_xlabel('NO$_{2}$/10$^{15}$ [molec cm$^{-2}$]')
    ax2.xaxis.offsetText.set_visible(False)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.yaxis.set_label_coords(-0.11,0.5)
    for ia in [inset_axes1, inset_axes2]:
        ia.set_xlim([0, 1.2e16])
        ia.set_xticks([0, 0.3e16, 0.6e16, 0.9e16, 1.2e16])
        ia.set_xticklabels([])
        ia.xaxis.offsetText.set_visible(False)
        ia.set_ylim([0,1.03])
        ia.set_yticks([0,0.5,1.03])
        ia.set_yticklabels(['0.0','0.5','1.0'])
        ia.set_ylabel('Cumulative\nprobability')
    plt.subplots_adjust(hspace=0.33, top=0.95, bottom=0.1)
    plt.savefig(DIR_FIGS+'figS11_revised.pdf', dpi=1000)
    plt.show()
    return 
    
# import numpy as np
# import sys
# sys.path.append('/Users/ghkerr/GW/tropomi_ej/')
# import tropomi_census_utils
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
# FIPS = ['01', '04', '05', '06', '08', '09', '10', '11', '12', '13', '16', 
#         '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
#         '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', 
#         '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50',
#         '51', '53', '54', '55', '56']
# harmonized = tropomi_census_utils.open_census_no2_harmonzied(FIPS)
# # # Add vehicle ownership/road density data
# harmonized = tropomi_census_utils.merge_harmonized_vehicleownership(harmonized)
# # Split into rural and urban tracts
# harmonized_urban, harmonized_rural = \
#     tropomi_census_utils.split_harmonized_byruralurban(harmonized)

# # Main figures
# fig1(harmonized, harmonized_urban)
# fig2(harmonized, harmonized_rural, harmonized_urban)
# fig3(harmonized_urban)
# fig4(harmonized, lat_dg, lng_dg, no2_post_dg, no2_pre_dg) 

# # Supplementary figures
# figS1()
# figS2(FIPS)
# figS3(harmonized, harmonized_rural)
# figS4(harmonized, harmonized_urban, harmonized_rural)
# figS5(harmonized, harmonized_rural, harmonized_urban)
# figS6(harmonized_urban)
# figS7()
# figS8()
# figS9(harmonized_urban)
# figS10(FIPS, harmonized_urban, harmonized_rural)
# figS11(harmonized)

# # Figure for Dan's paper
# import netCDF4 as nc
# import h5py
# import matplotlib.font_manager as fm
# font = fm.FontProperties(family = 'Palatino-Bold', 
#     fname = DIR_TYPEFACE+'Palatino-Bold.ttf', size=12)
# # 13 March - 13 June 2019 and 2020 average NO2
# no2_2019_dg = nc.Dataset(DIR_TROPOMI+
#     'Tropomi_NO2_griddedon0.01grid_2019_QA75.ncf')
# no2_2019_dg = no2_2019_dg['NO2'][:]
# lat_dg = nc.Dataset(DIR_TROPOMI+'LatLonGrid.ncf')['LAT'][:].data
# lng_dg = nc.Dataset(DIR_TROPOMI+'LatLonGrid.ncf')['LON'][:].data
# # Open Cooper et al. (2020) dataset
# no2_2019_mc = h5py.File(DIR_TROPOMI+'tropomi_surface_no2_0p025x0p03125'+
#     '_northamerica.h5', 'r')
# lat_mc = no2_2019_mc['LAT_CENTER'][0,:]
# lng_mc = no2_2019_mc['LON_CENTER'][:,0]
# no2_2019_mc = no2_2019_mc['surface_NO2_2019'][:]
# def get_merged_csv(flist, **kwargs):
#     """Function reads CSV files in the list comprehension loop, this list of
#     DataFrames will be passed to the pd.concat() function which will return 
#     single concatenated DataFrame. Adapted from: 
#     https://stackoverflow.com/questions/35973782/reading-multiple-csv-
#     files-concatenate-list-of-file-names-them-into-a-singe-dat
#     """
#     from dask import dataframe as dd
#     return dd.concat([dd.read_csv(f, **kwargs) for f in flist])
# import time
# start_time = time.time()
# print('# # # # Loading AQS NO2 ...') 
# import numpy as np
# from decimal import Decimal
# from dask import dataframe as dd
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# import pandas as pd
# import sys
# sys.path.append('/Users/ghkerr/phd/utils/')
# from geo_idx import geo_idx
# PATH_AQS = '/Users/ghkerr/GW/data/aq/aqs/'
# years = [2019]
# date_start, date_end = '2019-01-01', '2019-12-31'
# dtype = {'State Code' : np.str,'County Code' : np.str,'Site Num' : np.str,
#     'Parameter Code' : np.str, 'POC' : np.str, 'Latitude' : np.float64,
#     'Longitude' : np.float64, 'Datum' : np.str, 'Parameter Name' : np.str,
#     'Date Local' : np.str, 'Time Local' : np.str, 'Date GMT' : np.str,
#     'Time GMT' : np.str, 'Sample Measurement' : np.float64, 
#     'Units of Measure' : np.str, 'MDL' : np.str, 'Uncertainty' : np.str,
#     'Qualifier' : np.str, 'Method Type' : np.str, 'Method Code' : np.str,
#     'Method Name' : np.str, 'State Name' : np.str, 
#     'County Name' : np.str, 'Date of Last Change' : np.str}
# filenames_no2 = []
# # Fetch file names for years of interest
# for year in years:
#     filenames_no2.append(PATH_AQS+'hourly_42602_%s.csv'%year)
# filenames_no2.sort()
# # Read multiple CSV files (yearly) into Pandas dataframe 
# aqs_no2_raw = get_merged_csv(filenames_no2, dtype=dtype, 
#     usecols=list(dtype.keys()))
# # Create site ID column 
# aqs_no2_raw['Site ID'] = aqs_no2_raw['State Code']+'-'+\
#     aqs_no2_raw['County Code']+'-'+aqs_no2_raw['Site Num']
# # Drop unneeded columns; drop latitude/longitude coordinates for 
# # temperature observations as the merging of the O3 and temperature 
# # DataFrames will supply these coordinates 
# to_drop = ['Parameter Code', 'POC', 'Datum', 'Parameter Name',
#     'Date GMT', 'Time GMT', 'Units of Measure', 'MDL', 'Uncertainty', 
#     'Qualifier', 'Method Type', 'Method Code', 'Method Name', 'State Name',
#     'County Name', 'Date of Last Change', 'State Code', 'County Code', 
#     'Site Num']
# aqs_no2_raw = aqs_no2_raw.drop(to_drop, axis=1)
# # Select months in measuring period     
# aqs_no2_raw = aqs_no2_raw.loc[dd.to_datetime(aqs_no2_raw['Date Local']).isin(
#     pd.date_range(date_start,date_end))]
# if filterhours == True:
#     aqs_no2_raw = aqs_no2_raw.loc[dd.to_datetime(aqs_no2_raw[
#         'Time Local']).isin(['13:00','14:00'])]
# aqs_no2 = aqs_no2_raw.groupby(['Site ID']).mean()
# # Turns lazy Dask collection into its in-memory equivalent
# aqs_no2 = aqs_no2.compute()
# # aqs_no2_raw = aqs_no2_raw.compute()
# # Loop through rows (stations) and find closest TROPOMI grid cell
# tropomi_no2_atstations = []
# cooper_no2_atstations = []
# for row in np.arange(0, len(aqs_no2), 1):
#     aqs_no2_station = aqs_no2.iloc[row]
#     lng_station = aqs_no2_station['Longitude']
#     lat_station = aqs_no2_station['Latitude']
#     # Closest indicies (Alaska/Hawaii will trip this)
#     lng_tropomi_near = geo_idx(lng_station, lng_dg)
#     lat_tropomi_near = geo_idx(lat_station, lat_dg)
#     lng_cooper_near = geo_idx(lng_station, lng_mc)
#     lat_cooper_near = geo_idx(lat_station, lat_mc)
#     # Sample nearest TROPOMI retrieval
#     if (lng_tropomi_near is None) or (lat_tropomi_near is None):
#         tropomi_no2_atstations.append(np.nan)
#     else:
#         tropomi_no2_station = no2_2019_dg[lat_tropomi_near,lng_tropomi_near]
#         tropomi_no2_atstations.append(tropomi_no2_station)
#     # Sample nearest surface-level NO2 from Cooper et al. (2020)
#     if (lng_cooper_near is None) or (lng_cooper_near is None):
#         cooper_no2_atstations.append(np.nan)
#     else:
#         cooper_no2_station = no2_2019_mc[lng_cooper_near,lat_cooper_near]
#         cooper_no2_atstations.append(cooper_no2_station)        
# aqs_no2['TROPOMINO2_2019'] = tropomi_no2_atstations
# aqs_no2['CooperNO2_2019'] = cooper_no2_atstations
# # Site IDs for on-road monitoring sites 
# # from https://www3.epa.gov/ttnamti1/nearroad.html
# onroad = ['13-121-0056','13-089-0003','48-453-1068','06-029-2019',
#     '24-027-0006','24-005-0009','01-073-2059','16-001-0023','25-025-0044',
#     '25-017-0010','36-029-0023','37-119-0045','17-031-0218','17-031-0118',
#     '39-061-0048','39-035-0073','39-049-0038','48-113-1067','48-439-1053',
#     '08-031-0027','08-031-0028','19-153-6011','26-163-0093','26-163-0095',
#     '06-019-2016','09-003-0025','48-201-1066','48-201-1052','18-097-0087',
#     '12-031-0108','29-095-0042','32-003-1501','32-003-1502','06-059-0008',
#     '06-037-4008','21-111-0075','47-157-0100','12-011-0035','12-086-0035',
#     '55-079-0056','27-053-0962','27-037-0480','47-037-0040','22-071-0021',
#     '34-003-0010','36-081-0125','40-109-0097','12-095-0009','42-101-0075',
#     '42-101-0076','04-013-4019','04-013-4020','42-003-1376','41-067-0005',
#     '44-007-0030','37-183-0021','51-760-0025','06-071-0026','06-071-0027', 
#     '36-055-0015','06-067-0015','49-035-4002','48-029-1069','06-073-1017',
#     '06-001-0012','06-001-0013','06-001-0015','06-085-0006','72-061-0006',
#     '53-033-0030','53-053-0024','29-510-0094','29-189-0016','12-057-0113',
#     '12-057-1111','12-103-0027','51-059-0031','11-001-0051']
# # Select mobile sites vs other AQS sites
# aqs_onroad = aqs_no2.loc[aqs_no2.index.isin(onroad)]
# aqs_other = aqs_no2.loc[~aqs_no2.index.isin(onroad)]
# # Dan's figures appear to have an aspect ratio of 2100 x 500
# from matplotlib.figure import figaspect
# w, h = figaspect(1500/2100)
# fig, ax = plt.subplots(figsize=(w,h))
# # Axis titles
# color_white = '#0095A8'
# color_non = '#FF7043'
# # Plotting
# ax.plot(aqs_onroad['TROPOMINO2_2019'].values, 
#     aqs_onroad['Sample Measurement'].values, 'o', markersize=3, 
#     label='Near-road', color='lightgrey')
# ax.plot(aqs_other['TROPOMINO2_2019'].values, 
#     aqs_other['Sample Measurement'].values, 'ko', markersize=4, 
#     label='Not near-road')
# ax.set_xlabel('TROPOMI NO$_\mathregular{2}$/10$^\mathregular{16}$ '+
#     '[molec cm$^\mathregular{-2}$]', fontsize=14, fontproperties = font)
# ax.set_ylabel('AQS NO$_\mathregular{2}$* [ppbv]', fontsize=14, 
#     fontproperties=font)
# ax.set_xlim([0,1.25e16])
# ax.set_ylim([0,30])
# ax.set_xticks([0.0e16,0.2e16,0.4e16,0.6e16,0.8e16,1.0e16,1.2e16])
# ax.set_xticklabels(['0.0','0.2','0.4','0.6','0.8','1.0','1.2'], 
#     fontproperties=font)
# ax.set_yticks([0,5,10,15,20,25,30])
# ax.set_yticklabels(['0','5','10','15','20','25','30'], fontproperties=font)
# plt.setp(ax.get_xticklabels(), fontproperties=font, fontsize=14)
# plt.setp(ax.get_yticklabels(), fontproperties=font, fontsize=14)
# ax.xaxis.offsetText.set_visible(False)
# Calculate logarithmic fit
# # Force TROPOMI < 0.1e16 values to be AQS = 1 ppbv (just for the fit, not 
# # for the plot)
# aqs_other.loc[aqs_other['TROPOMINO2_2019']<1e15, 'Sample Measurement']=1.
# aqs_onroad.loc[aqs_onroad['TROPOMINO2_2019']<1e15, 'Sample Measurement']=1.
# mask = (~np.isnan(aqs_other['Sample Measurement'].values) & 
#     ~np.isnan(aqs_other['TROPOMINO2_2019'].values))
# # Logarithmic regression (from https://stackoverflow.com/questions/
# # 49944018/fit-a-logarithmic-curve-to-data-points-and-extrapolate
# # -out-in-numpy/49944478)
# def logFit(x,y):
#     # cache some frequently reused terms
#     sumy = np.sum(y)
#     sumlogx = np.sum(np.log(x))
#     b = (x.size*np.sum(y*np.log(x)) - sumy*sumlogx)/(
#         x.size*np.sum(np.log(x)**2) - sumlogx**2)
#     a = (sumy - b*sumlogx)/x.size
#     return a,b
# def logFunc(x, a, b):
#     return a + b*np.log(x)
# coeff = logFit(np.sort(aqs_other['TROPOMINO2_2019'].values[mask]),
#     np.sort(aqs_other['Sample Measurement'].values[mask]))
# ax.plot(np.sort(aqs_other['TROPOMINO2_2019'].values[mask])[38:], 
#     logFunc(np.sort(aqs_other['TROPOMINO2_2019'].values[mask]), 
#     *logFit(np.sort(aqs_other['TROPOMINO2_2019'].values[mask]),
#     np.sort(aqs_other['Sample Measurement'].values[mask])))[38:], 
#     color='crimson', label='a = %.1f, b = %.1f'%(
#     coeff[0],coeff[-1]))
# ax.plot(np.linspace(0,0.105e16,100),np.tile(1,100),color='crimson')
# # Calculate linear fit
# mask = (~np.isnan(aqs_other['Sample Measurement'].values) & 
#     ~np.isnan(aqs_other['TROPOMINO2_2019'].values))
# coeff = np.polyfit(aqs_other['TROPOMINO2_2019'].values[mask], 
#     aqs_other['Sample Measurement'].values[mask], 1)
# poly1d_fn = np.poly1d(coeff) 
# # poly1d_fn is now a function which takes in x and returns an estimate for y
# ax.plot(np.sort(aqs_other['TROPOMINO2_2019'].values[mask]),
#     poly1d_fn(np.sort(aqs_other['TROPOMINO2_2019'].values[mask])), ls='-', 
#     color='crimson', label='m = 1.7 x 10$^{\mathregular{-15}}$, b = 1.5')
# # Calculate R-squared value
# r2 = np.corrcoef(aqs_other['Sample Measurement'].values[mask],
#     coeff[0]+(coeff[1]*aqs_other['TROPOMINO2_2019'].values[mask]))**2
# tit = ax.set_title('R$^\mathregular{2}$ = 0.66', loc='right', 
#     fontproperties=font)
# tit.set_fontsize(12)
# leg = ax.legend(fontsize=12, ncol=1, frameon=True, loc=4, labelspacing=0.08, 
#     prop=font)
# plt.subplots_adjust(right=0.96, top=0.92, bottom=0.15)
# plt.savefig('/Users/ghkerr/Desktop/dan_figuresubplot_revised.eps', dpi=1000)

"""DAN MAPS OF MOST/LEAST WHITE TRACTS IN NYC, LA, AND DC"""
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.rcParams['hatch.linewidth'] = 0.3     
# import numpy as np
# import matplotlib as mpl
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.io import shapereader
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# # Initialize figure, axes
# fig = plt.figure(figsize=(12,3))
# proj = ccrs.PlateCarree(central_longitude=0.0)
# ax1 = plt.subplot2grid((1,3),(0,0), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# ax2 = plt.subplot2grid((1,3),(0,1), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# ax3 = plt.subplot2grid((1,3),(0,2), projection=ccrs.PlateCarree(
#     central_longitude=0.))
# fips = [['36','34'], ['06'], ['11','24','51']]
# nyc = ['36047','36081','36061','36005','36085','36119', '34003', 
#     '34017','34031','36079','36087','36103','36059','34023','34025',
#     '34029','34035','34013','34039','34027','34037'	,'34019']
# # Los Angeles-Long Beach-Anaheim, CA MSA
# losangeles = ['06037','06059']
# # Washington-Arlington-Alexandria, DC-VA-MD-WV MSA
# washington = ['11001','24009','24017','24021','24031','24033','51510',
#     '51013','51043','51047','51059','51600','51610','51061','51630',
#     '51107','51683','51685','51153','51157','51177','51179','51187']
# # Load demographics for each city 
# harmonized_city = tropomi_census_utils.subset_harmonized_bycountyfips(
#     harmonized, nyc)    
# frac_white = ((harmonized_city['AJWNE002'])/harmonized_city['AJWBE001'])
# mostwhite = harmonized_city.iloc[np.where(frac_white > 
#     np.nanpercentile(frac_white, ptile_upper))]
# leastwhite = harmonized_city.iloc[np.where(frac_white < 
#     np.nanpercentile(frac_white, ptile_lower))]
# sf = fips[0]
# records, tracts = [], []
# for sfi in sf:
#     shp = shapereader.Reader(DIR_GEO+'tigerline/'+
#         'tl_2019_%s_tract/tl_2019_%s_tract.shp'%(sfi,sfi))
#     recordsi = shp.records()
#     tractsi = shp.geometries()
#     recordsi = list(recordsi)
#     tractsi = list(tractsi)
#     tracts.append(tractsi)
#     records.append(recordsi)
#     geoids_records = [x.attributes['GEOID'] for x in np.hstack(records)]
#     geoids_records = np.where(np.in1d(np.array(geoids_records), 
#         harmonized_city.index)==True)[0]
# records = list(np.hstack(records)[geoids_records])
# tracts = list(np.hstack(tracts)[geoids_records])
# for geoid in harmonized_city.index:
#     where_geoid = np.where(np.array([x.attributes['GEOID'] for x in 
#         records])==geoid)[0][0]
#     tract = tracts[where_geoid]
#     harmonized_tract = harmonized_city.loc[harmonized_city.index.isin(
#         [geoid])]
#     if harmonized_tract.index[0] in mostwhite.index:
#         ax1.add_geometries([tract], proj, facecolor='b', edgecolor='None', 
#             alpha=1., zorder=10)                 
#     if harmonized_tract.index[0] in leastwhite.index:
#         ax1.add_geometries([tract], proj, facecolor='r', edgecolor='None', 
#             alpha=1., zorder=10)                         
# ax1.set_extent([-75.15, -71.8, 39.5, 41.9], proj)
# # Load demographics for each city 
# harmonized_city = tropomi_census_utils.subset_harmonized_bycountyfips(
#     harmonized, losangeles)    
# frac_white = ((harmonized_city['AJWNE002'])/harmonized_city['AJWBE001'])
# mostwhite = harmonized_city.iloc[np.where(frac_white > 
#     np.nanpercentile(frac_white, ptile_upper))]
# leastwhite = harmonized_city.iloc[np.where(frac_white < 
#     np.nanpercentile(frac_white, ptile_lower))]
# sf = fips[1]
# records, tracts = [], []
# for sfi in sf:
#     shp = shapereader.Reader(DIR_GEO+'tigerline/'+
#         'tl_2019_%s_tract/tl_2019_%s_tract.shp'%(sfi,sfi))
#     recordsi = shp.records()
#     tractsi = shp.geometries()
#     recordsi = list(recordsi)
#     tractsi = list(tractsi)
#     tracts.append(tractsi)
#     records.append(recordsi)
#     geoids_records = [x.attributes['GEOID'] for x in np.hstack(records)]
#     geoids_records = np.where(np.in1d(np.array(geoids_records), 
#         harmonized_city.index)==True)[0]
# records = list(np.hstack(records)[geoids_records])
# tracts = list(np.hstack(tracts)[geoids_records])
# for geoid in harmonized_city.index:
#     where_geoid = np.where(np.array([x.attributes['GEOID'] for x in 
#         records])==geoid)[0][0]
#     tract = tracts[where_geoid]
#     harmonized_tract = harmonized_city.loc[harmonized_city.index.isin(
#         [geoid])]
#     if harmonized_tract.index[0] in mostwhite.index:
#         ax2.add_geometries([tract], proj, facecolor='b', edgecolor='None', 
#             alpha=1., zorder=10)                 
#     if harmonized_tract.index[0] in leastwhite.index:
#         ax2.add_geometries([tract], proj, facecolor='r', edgecolor='None', 
#             alpha=1., zorder=10)                         
# ax2.set_extent([-118.9, -117.2, 33.1, 34.6], proj)
# # Load demographics for each city 
# harmonized_city = tropomi_census_utils.subset_harmonized_bycountyfips(
#     harmonized, washington)    
# frac_white = ((harmonized_city['AJWNE002'])/harmonized_city['AJWBE001'])
# mostwhite = harmonized_city.iloc[np.where(frac_white > 
#     np.nanpercentile(frac_white, ptile_upper))]
# leastwhite = harmonized_city.iloc[np.where(frac_white < 
#     np.nanpercentile(frac_white, ptile_lower))]
# sf = fips[2]
# records, tracts = [], []
# for sfi in sf:
#     shp = shapereader.Reader(DIR_GEO+'tigerline/'+
#         'tl_2019_%s_tract/tl_2019_%s_tract.shp'%(sfi,sfi))
#     recordsi = shp.records()
#     tractsi = shp.geometries()
#     recordsi = list(recordsi)
#     tractsi = list(tractsi)
#     tracts.append(tractsi)
#     records.append(recordsi)
#     geoids_records = [x.attributes['GEOID'] for x in np.hstack(records)]
#     geoids_records = np.where(np.in1d(np.array(geoids_records), 
#         harmonized_city.index)==True)[0]
# records = list(np.hstack(records)[geoids_records])
# tracts = list(np.hstack(tracts)[geoids_records])
# for geoid in harmonized_city.index:
#     where_geoid = np.where(np.array([x.attributes['GEOID'] for x in 
#         records])==geoid)[0][0]
#     tract = tracts[where_geoid]
#     harmonized_tract = harmonized_city.loc[harmonized_city.index.isin(
#         [geoid])]
#     if harmonized_tract.index[0] in mostwhite.index:
#         ax3.add_geometries([tract], proj, facecolor='b', edgecolor='None', 
#             alpha=1., zorder=10)                 
#     if harmonized_tract.index[0] in leastwhite.index:
#         ax3.add_geometries([tract], proj, facecolor='r', edgecolor='None', 
#             alpha=1., zorder=10)                         
# ax3.set_extent([-77.9, -76.6, 38.2, 39.5], proj)
# fname = DIR_GEO+'counties/tl_2019_us_county/'+'tl_2019_us_county.shp'
# shape_feature = cfeature.ShapelyFeature(shapereader.Reader(fname).geometries(),
#     proj, facecolor='none', edgecolor='k')
# for ax in [ax1, ax2, ax3]:
#     ax.add_feature(shape_feature)
# plt.savefig('/Users/ghkerr/Desktop/MSA_maps_fordan.png', dpi=300)

"""BRIAN MACDONALD EMISSION INVENTORIES"""
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# emiss = pd.read_excel('/Users/ghkerr/GW/data/emissions/'+
#     'FIVE Emission Trends (for Shobha) v4.xlsx', sheet_name='COVID',
#     usecols='A:Q', header=2)
# emiss.rename(columns={'NOx (t/d)':'LA Gas', 
#     'NOx (t/d).1':'LA Diesel', 
#     'NOx (t/d).2':'LA On-road', 
#     'NOx (t/d).3':'SF Gas', 
#     'NOx (t/d).4':'SF Diesel', 
#     'NOx (t/d).5':'SF On-road', 
#     'NOx (t/d).6':'NYC Gas', 
#     'NOx (t/d).7':'NYC Diesel', 
#     'NOx (t/d).8':'NYC On-road', 
#     'NOx (t/d).9':'ATL Gas', 
#     'NOx (t/d).10':'ATL Diesel', 
#     'NOx (t/d).11':'ATL On-road', 
#     'NOx (t/d).12':'SJV Gas', 
#     'NOx (t/d).13':'SJV Diesel',    
#     'NOx (t/d).14':'SJV On-road'}, inplace=True)
# city = 'SJV'
# fig = plt.figure(figsize=(9,3))
# ax = plt.subplot2grid((1,1),(0,0))
# ax.plot(emiss['Date'], emiss['%s Gas'%(city)], alpha=0.3, lw=0.5, 
#     color='#1b9e77')
# ax.plot(emiss['Date'], emiss['%s Diesel'%(city)], alpha=0.3, lw=0.5, 
#     color='#d95f02')
# # 7 day rolling average
# emiss['%s Gas'%(city)] = emiss['%s Gas'%(city)].rolling(window=7).mean()
# emiss['%s Diesel'%(city)] = emiss['%s Diesel'%(city)].rolling(window=7).mean()
# ax.plot(emiss['Date'], emiss['%s Gas'%(city)], color='#1b9e77', label='Gas')
# ax.plot(emiss['Date'], emiss['%s Diesel'%(city)], color='#d95f02', label='Diesel')
# plt.legend(frameon=False)
# ax.set_xlim([emiss['Date'].min(), emiss['Date'].max()])
# ax.set_title(city, loc='left', fontsize=16)
# ax.set_ylabel('NO$_{\mathregular{x}}$ [tonnes d$^{\mathregular{-1}}$]')
# plt.savefig(DIR_FIGS+'noxemiss_gasdiesel_%s.png'%city, dpi=400)
# plt.show()















# import numpy as np
# import pandas as pd
# from datetime import datetime
# import scipy.stats as st
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from scipy import stats
# import pandas as pd
# # Read in csv file for vehicle ownership/road density
# noxsource = pd.read_csv(DIR_HARM+'noxsourcedensity_us_v2.csv', delimiter=',', 
#     header=0, engine='python')
# # Leading 0 for state FIPS codes < 10 
# noxsource['GEOID'] = noxsource['GEOID'].map(lambda x: f'{x:0>11}')
# # Make GEOID a string and index row 
# noxsource = noxsource.set_index('GEOID')
# noxsource.index = noxsource.index.map(str)
# # Make other columns floats
# for col in noxsource.columns:
#     noxsource[col] = noxsource[col].astype(float)
# # Merge with harmonized census data
# harmonized_noxsource = harmonized_urban.merge(noxsource, left_index=True, 
#     right_index=True)
# cems_byrace = []
# cems_byincome = []
# cems_byeducation = []
# cems_byethnicity = []
# cems_byvehicle = []
# cemsp90_byrace, cemsp10_byrace = [], []
# cemsp90_byincome, cemsp10_byincome = [], []
# cemsp90_byeducation, cemsp10_byeducation = [], []
# cemsp90_byethnicity, cemsp10_byethnicity = [], []
# cemsp90_byvehicle, cemsp10_byvehicle = [], []

# income = harmonized_noxsource['AJZAE001']
# race = (harmonized_noxsource['AJWNE002']/harmonized_noxsource['AJWBE001'])
# education = (harmonized_noxsource.loc[:,'AJYPE019':'AJYPE025'
#     ].sum(axis=1)/harmonized_noxsource['AJYPE001'])
# ethnicity = (harmonized_noxsource['AJWWE002']/harmonized_noxsource['AJWWE001'])
# vehicle = 1-harmonized_noxsource['FracNoCar']
# for ptilel, ptileu in zip(np.arange(0,100,10), np.arange(10,110,10)):    
#     # By income
#     decile_income = harmonized_noxsource.loc[(income>
#         np.nanpercentile(income, ptilel)) & (
#         income<=np.nanpercentile(income, ptileu))]
#     decile_race = harmonized_noxsource.loc[(race>
#         np.nanpercentile(race, ptilel)) & (
#         race<=np.nanpercentile(race, ptileu))]             
#     cems_byincome.append(decile_income['CEMSwithin1'].mean()) 
#     cemsp10_byincome.append(decile_income['CEMSwithin1_p20'].mean()) 
#     cemsp90_byincome.append(decile_income['CEMSwithin1_p80'].mean())  
#     # By race            
#     decile_race = harmonized_noxsource.loc[(race>
#         np.nanpercentile(race, ptilel)) & (
#         race<=np.nanpercentile(race, ptileu))] 
#     cems_byrace.append(decile_race['CEMSwithin1'].mean()) 
#     cemsp10_byrace.append(decile_race['CEMSwithin1_p20'].mean()) 
#     cemsp90_byrace.append(decile_race['CEMSwithin1_p80'].mean()) 
#     # By education 
#     decile_education = harmonized_noxsource.loc[(education>np.nanpercentile(
#         education, ptilel)) & (education<=np.nanpercentile(education, 
#         ptileu))]
#     cems_byeducation.append(decile_education['CEMSwithin1'].mean())  
#     cemsp10_byeducation.append(decile_education['CEMSwithin1_p20'].mean())   
#     cemsp90_byeducation.append(decile_education['CEMSwithin1_p80'].mean())      
#     # By ethnicity
#     decile_ethnicity = harmonized_noxsource.loc[(ethnicity>
#         np.nanpercentile(ethnicity, ptilel)) & (
#         ethnicity<=np.nanpercentile(ethnicity, ptileu))]    
#     cems_byethnicity.append(decile_ethnicity['CEMSwithin1'].mean())  
#     cemsp10_byethnicity.append(decile_ethnicity['CEMSwithin1_p20'].mean())  
#     cemsp90_byethnicity.append(decile_ethnicity['CEMSwithin1_p80'].mean())  
#     # By vehicle ownership            
#     decile_vehicle = harmonized_noxsource.loc[(vehicle>
#         np.nanpercentile(vehicle, ptilel)) & (vehicle<=np.nanpercentile(
#         vehicle, ptileu))]
#     cems_byvehicle.append(decile_vehicle['CEMSwithin1'].mean())
#     cemsp10_byvehicle.append(decile_vehicle['CEMSwithin1_p20'].mean())  
#     cemsp90_byvehicle.append(decile_vehicle['CEMSwithin1_p80'].mean())  



# fig = plt.figure(figsize=(5,8))
# ax1 = plt.subplot2grid((5,1),(0,0))
# ax2 = plt.subplot2grid((5,1),(1,0))
# ax3 = plt.subplot2grid((5,1),(2,0))
# ax4 = plt.subplot2grid((5,1),(3,0))
# ax5 = plt.subplot2grid((5,1),(4,0))






# axes = [ax1, ax2, ax3, ax4, ax5]
# curves1 = [cemsp10_byrace, cemsp10_byincome, cemsp10_byeducation, 
#     cemsp10_byethnicity, cemsp10_byvehicle] 
# curves2 = [cemsp90_byrace, cemsp90_byincome, cemsp90_byeducation, 
#     cemsp90_byethnicity, cemsp90_byvehicle]
# # Loop through NOx sources
# for i in np.arange(0,5,1):
#      axes[i].plot(curves1[i], ls='-', lw=2, color='k', zorder=11)
#      axes[i].plot(curves2[i], ls='-.', lw=2, color='k', zorder=11)

# ax1.annotate('Higher',xy=(0.58,0.92),xytext=(0.78,0.92), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color='k'), 
#     fontsize=12, color='k', xycoords=ax1.transAxes)
# ax1.annotate('Lower',xy=(0.41,0.92),xytext=(0.1,0.92), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color='k'), 
#     fontsize=12, color='k', xycoords=ax1.transAxes)


# ax1.set_title('(a) White')
# ax2.set_title('(b) Income')
# ax3.set_title('(c) Educational attainment')
# ax4.set_title('(d) Non-Hispanic')
# ax5.set_title('(e) Vehicle ownership')
# ax1.text(0.5, 0.92, 'Income', fontsize=12, va='center',
#     color=color1, ha='center', transform=ax1.transAxes)

# # ax1.text(0.5, 0.84, 'Education', fontsize=12, color=color2, va='center', 
# #     ha='center', transform=ax1.transAxes)
# # ax1.annotate('More',xy=(0.61,0.84),xytext=(0.78,0.84), va='center', 
# #     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color2), fontsize=12,
# #     color=color2, xycoords=ax1.transAxes)
# # ax1.annotate('Less',xy=(0.38,0.84),xytext=(0.1,0.84), va='center',
# #     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color2), 
# #     fontsize=12, color=color2, xycoords=ax1.transAxes)
# # ax1.text(0.5, 0.76, 'White', fontsize=12, va='center',
# #     color=color3, ha='center', transform=ax1.transAxes)
# # ax1.annotate('More',xy=(0.57,0.76),xytext=(0.78,0.76), va='center',
# #     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color3), color=color3,
# #     fontsize=12, xycoords=ax1.transAxes)
# # ax1.annotate('Less',xy=(0.43,0.76),xytext=(0.1,0.76), va='center',
# #     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color3), fontsize=12, 
# #     color=color3, xycoords=ax1.transAxes)






# fig = plt.figure(figsize=(5,8))
# ax1 = plt.subplot2grid((5,1),(0,0))
# ax2 = plt.subplot2grid((5,1),(1,0))
# ax3 = plt.subplot2grid((5,1),(2,0))
# ax4 = plt.subplot2grid((5,1),(3,0))
# ax5 = plt.subplot2grid((5,1),(4,0))






# axes = [ax1, ax2, ax3, ax4, ax5]
# curves1 = [cemsp10_byrace, cemsp10_byincome, cemsp10_byeducation, 
#     cemsp10_byethnicity, cemsp10_byvehicle] 
# curves2 = [cemsp90_byrace, cemsp90_byincome, cemsp90_byeducation, 
#     cemsp90_byethnicity, cemsp90_byvehicle]
# # Loop through NOx sources
# for i in np.arange(0,5,1):
#      axes[i].plot(curves2[i], ls='-', lw=2, color='k', zorder=11, label='Large sources')
#      # axt = axes[i].twinx()
#      # axt.plot(curves2[i], ls='-.', lw=2, color='k', zorder=11, label='Large sources')

# ax1.annotate('Higher',xy=(0.58,0.92),xytext=(0.78,0.92), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color='k'), 
#     fontsize=12, color='k', xycoords=ax1.transAxes)
# ax1.annotate('Lower',xy=(0.41,0.92),xytext=(0.1,0.92), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color='k'), 
#     fontsize=12, color='k', xycoords=ax1.transAxes)
# ax5.legend()

# ax1.set_title('(a) White')
# ax2.set_title('(b) Income')
# ax3.set_title('(c) Educational attainment')
# ax4.set_title('(d) Non-Hispanic')
# ax5.set_title('(e) Vehicle ownership')

# for ax in axes:
#     ax.set_xlim([0,9])
#     ax.set_xticks(np.arange(0,10,1))
#     ax.set_xticklabels([])


# ax5.set_xticklabels(['First', 'Second', 'Third', 'Fourth', 'Fifth', 
#     'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth'], fontsize=10)
# ax5.set_xlabel('Decile', fontsize=12)

# plt.subplots_adjust(hspace=0.5, bottom=0.08, top=0.95)
# plt.savefig('/Users/ghkerr/Desktop/largeCEMS_fordan.png', dpi=600)




# fig = plt.figure(figsize=(11.5,5))
# ax1 = plt.subplot2grid((1,2),(0,0))
# ax2 = plt.subplot2grid((1,2),(0,1))
# # Colors for each demographic variable
# color1 = '#0095A8'
# color2 = '#FF7043'
# color3 = '#5D69B1'
# color4 = '#CC3A8E'
# color5 = '#4daf4a'
# axes = [ax1, ax2]
# curves = [[cemsp10_byrace, cemsp10_byincome, cemsp10_byeducation, 
#       cemsp10_byethnicity, cemsp10_byvehicle], 
#     [cemsp90_byrace, cemsp90_byincome, cemsp90_byeducation, 
#       cemsp90_byethnicity, cemsp90_byvehicle]]
# # Loop through NOx sources
# for i in np.arange(0,2,1):
#     # Plotting
#     axes[i].plot(curves[i][1], ls='-', lw=2, color=color1, zorder=11)
#     axes[i].plot(curves[i][2], ls='-', lw=2, color=color2, zorder=11)
#     axes[i].plot(curves[i][0], ls='-', lw=2, color=color3, zorder=11)
#     axes[i].plot(curves[i][3], ls='-', lw=2, color=color4, zorder=11)
#     axes[i].plot(curves[i][4], ls='-', lw=2, color=color5, zorder=11)
# # Legend
# ax1.text(0.5, 0.92, 'Income', fontsize=12, va='center',
#     color=color1, ha='center', transform=ax1.transAxes)
# ax1.annotate('Higher',xy=(0.58,0.92),xytext=(0.78,0.92), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color1), 
#     fontsize=12, color=color1, xycoords=ax1.transAxes)
# ax1.annotate('Lower',xy=(0.41,0.92),xytext=(0.1,0.92), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color1), 
#     fontsize=12, color=color1, xycoords=ax1.transAxes)
# ax1.text(0.5, 0.84, 'Education', fontsize=12, color=color2, va='center', 
#     ha='center', transform=ax1.transAxes)
# ax1.annotate('More',xy=(0.61,0.84),xytext=(0.78,0.84), va='center', 
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color2), fontsize=12,
#     color=color2, xycoords=ax1.transAxes)
# ax1.annotate('Less',xy=(0.38,0.84),xytext=(0.1,0.84), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color2), 
#     fontsize=12, color=color2, xycoords=ax1.transAxes)
# ax1.text(0.5, 0.76, 'White', fontsize=12, va='center',
#     color=color3, ha='center', transform=ax1.transAxes)
# ax1.annotate('More',xy=(0.57,0.76),xytext=(0.78,0.76), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color3), color=color3,
#     fontsize=12, xycoords=ax1.transAxes)
# ax1.annotate('Less',xy=(0.43,0.76),xytext=(0.1,0.76), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color3), fontsize=12, 
#     color=color3, xycoords=ax1.transAxes)
# ax1.text(0.5, 0.68, 'Hispanic', fontsize=12, 
#     color=color4, ha='center', va='center', transform=ax1.transAxes)
# ax1.annotate('Less',xy=(0.59,0.68),xytext=(0.78,0.68), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color4), fontsize=12, 
#     color=color4, xycoords=ax1.transAxes)
# ax1.annotate('More',xy=(0.4,0.68),xytext=(0.1,0.68), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color4), fontsize=12,
#     color=color4, xycoords=ax1.transAxes)
# ax1.text(0.5, 0.6, 'Vehicle ownership', fontsize=12, ha='center',
#     va='center', color=color5, transform=ax1.transAxes)
# ax1.annotate('More',xy=(0.65,0.6),xytext=(0.78,0.6), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color5), fontsize=12, 
#     color=color5, xycoords=ax1.transAxes)
# ax1.annotate('Less',xy=(0.34,0.6),xytext=(0.1,0.6), va='center',
#     arrowprops=dict(arrowstyle= '<|-', lw=1, color=color5), fontsize=12,
#     color=color5, xycoords=ax1.transAxes)
# for ax in axes:
#     ax.set_xlim([0,9])
#     ax.set_xticks(np.arange(0,10,1))
#     ax.set_xticklabels([])
# for ax in axes:
#     ax.set_xticklabels(['First', 'Second', 'Third', 'Fourth', 'Fifth', 
#         'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth'], fontsize=10)
#     ax.set_xlabel('Decile', fontsize=12)
# # ax1.set_ylim([0,0.01])
# # ax1.set_yticks(np.linspace(0,0.01,9))
# # ax1.set_yticklabels(['0','', '2.5', '', '5.0', '', '7.5', '',
# #     '10$\:$x$\:$10$^{\mathregular{-2}}$'])
# # ax2.set_ylim([0.0, 0.04])
# # ax2.set_yticks(np.linspace(0,0.04,9))
# # ax2.set_yticklabels(['0', '', '1', '', '2', '', '3', '', 
# #     '4$\:$x$\:$10$^{\mathregular{-2}}$'])
# # ax3.set_ylim([0,0.006])
# # ax3.set_yticks(np.linspace(0,0.006,9))
# # ax3.set_yticklabels(['0', '', '1.5' , '', '3.0' , '', '4.5' , '', 
# #     '6.0$\:$x$\:$10$^{\mathregular{-3}}$'])
# # ax4.set_ylim([0,4])
# # ax4.set_yticks(np.linspace(0,4,9))
# # ax4.set_yticklabels(['0','','1','','2','','3','','4'])
# ax1.set_title('(a) Small industry (< 20th percentile) density', 
#     fontsize=12, loc='left')              
# ax1.set_ylabel('[industries (1 km radius)$^{-1}$]', fontsize=12)              
# ax2.set_title('(b) Large industry (> 80th percentile) density', 
#     fontsize=12, loc='left')
# # plt.subplots_adjust(left=0.08, right=0.95)
# plt.savefig(DIR_FIGS+'figS6_largesmall_reviewercomments.png', dpi=1000)