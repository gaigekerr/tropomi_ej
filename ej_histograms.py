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

def fig1(harmonized):
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
    ax3.text(decreaseno2['NO2_ABS'].mean()+2e13, 2, '-2.9', color='black', 
        va='center')
    print('Mean gains for NO2 = %.2E'%Decimal(harmonized['NO2_ABS'].mean()))
    ax3.text(harmonized['NO2_ABS'].mean()+2e13, 1, '-0.85', color='black', 
        va='center')
    print('Smallest gains for NO2 = %.2E'%Decimal(increaseno2['NO2_ABS'].mean()))
    ax3.text(increaseno2['NO2_ABS'].mean()+2e13, 0, '0.08', color='black', 
        va='center')
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
    # Aesthetics    
    ax3.set_xlim([-3e15, 1e14])
    ax3.set_xticks([])    
    ax5.set_xlim([60000,80000])
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
    ax5.set_title('(e) Household income [$]', loc='left', fontsize=10)
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
    plt.savefig(DIR_FIGS+'fig1.pdf', dpi=1000)
    return 

def fig2(harmonized, harmonized_rural, harmonized_urban):
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
    fig = plt.figure(figsize=(12,7))
    ax1 = plt.subplot2grid((2,3),(0,0),rowspan=2)
    ax2 = plt.subplot2grid((2,3),(0,1),rowspan=2)
    ax3 = plt.subplot2grid((2,3),(0,2),rowspan=2)
    i = 0 
    yticks = []
    # For all tracts
    frac_white = (harmonized['AJWNE002']/harmonized['AJWBE001'])
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
    # Lockdown NO2  or white and non-white populations
    ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
    ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
    ax1.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()]), 
        np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()])+
        np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    # Historic NO2 for white and non-white populations
    ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    # Draw connection lines
    ax1.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
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
    ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, 
        zorder=12)
    ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, 
        zorder=12)
    ax1.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()]), 
        np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()])+
        np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax1.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
        np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
        np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))[0]), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7  
    # For rural tracts
    frac_white = (harmonized_urban['AJWNE002']/harmonized_urban['AJWBE001'])
    mostwhite = harmonized_urban.iloc[np.where(frac_white > 
        np.nanpercentile(frac_white, ptile_upper))]
    leastwhite = harmonized_urban.iloc[np.where(frac_white < 
        np.nanpercentile(frac_white, ptile_lower))]
    ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, 
        zorder=12)
    ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, 
        zorder=12)
    ax1.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()]), 
        np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()])+
        np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax1.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
        np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
        np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))[0]), 
        [i+os,i+os], color='k', ls='--', zorder=10)
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
        harmonized_city = subset_harmonized_bycountyfips(harmonized, city)
        # Find particular demographic for each city
        frac_white = (harmonized_city['AJWNE002']/harmonized_city['AJWBE001'])
        mostwhite = harmonized_city.iloc[np.where(frac_white > 
            np.nanpercentile(frac_white, ptile_upper))]
        leastwhite = harmonized_city.iloc[np.where(frac_white < 
            np.nanpercentile(frac_white, ptile_lower))]
        # Lockdown NO2  or white and non-white populations
        ax1.plot(mostwhite['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
        ax1.plot(leastwhite['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
        ax1.plot((np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()]), 
            np.min([mostwhite['PRENO2'].mean(),leastwhite['PRENO2'].mean()])+
            np.abs(np.diff([mostwhite['PRENO2'].mean(), leastwhite['PRENO2'].mean()]))[0]), 
            [i-os,i-os], color='k', ls='-', zorder=10)    
        # i = i+2
        # Historic NO2 for white and non-white populations
        ax1.plot(mostwhite['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
        ax1.plot(leastwhite['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
        # Draw connection lines
        ax1.plot((np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()]), 
            np.min([mostwhite['POSTNO2'].mean(),leastwhite['POSTNO2'].mean()])+
            np.abs(np.diff([mostwhite['POSTNO2'].mean(), leastwhite['POSTNO2'].mean()]))[0]), 
            [i+os,i+os], color='k', ls='--', zorder=10)
        yticks.append(np.nanmean([i]))
        ratio_pre.append(leastwhite['PRENO2'].mean()/mostwhite['PRENO2'].mean())    
        ratio_post.append(leastwhite['POSTNO2'].mean()/mostwhite['POSTNO2'].mean())
        i = i+7
    print('Disparities in 15 largest MSAs for race:')
    print('Baseline %.3f +/- %.3f'%(np.nanmean(ratio_pre),np.nanstd(ratio_pre)))
    print('Lockdown %.3f +/- %.3f'%(np.nanmean(ratio_post),np.nanstd(ratio_post)))
    print('\n')
    # Aesthetics 
    ax1.set_xlim([0.5e15,10e15])
    ax1.set_xticks(np.arange(1e15,10e15,1e15))
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
    ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
    ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
    ax2.plot((np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()]), 
        np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+
        np.abs(np.diff([mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    ax2.plot((np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()]), 
        np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
        np.abs(np.diff([mostwealthy['POSTNO2'].mean(), leastwealthy['POSTNO2'].mean()]))[0]), 
        [i+os,i+os], color='k', ls='--', zorder=10)
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
    ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, 
        zorder=12)
    ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, 
        zorder=12)
    ax2.plot((np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()]), 
        np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+
        np.abs(np.diff([mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax2.plot((np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()]), 
        np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
        np.abs(np.diff([mostwealthy['POSTNO2'].mean(), leastwealthy['POSTNO2'].mean()]))[0]), 
        [i+os,i+os], color='k', ls='--', zorder=10)
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
    ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, 
        zorder=12)
    ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, 
        zorder=12)
    ax2.plot((np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()]), 
        np.min([mostwealthy['PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+
        np.abs(np.diff([mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, 
        zorder=12)
    ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, 
        zorder=12)
    ax2.plot((np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()]), 
        np.min([mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
        np.abs(np.diff([mostwealthy['POSTNO2'].mean(), leastwealthy['POSTNO2'].mean()]))[0]), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7      
    ratio_pre = []
    ratio_post = []
    for city in [newyork, losangeles, chicago, dallas, houston, washington,
        miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
        riverside, detroit, seattle]:
        harmonized_city = subset_harmonized_bycountyfips(harmonized, city)
        mostwealthy = harmonized_city.loc[harmonized_city['AJZAE001'] > 
            np.nanpercentile(harmonized_city['AJZAE001'], 90)]
        leastwealthy = harmonized_city.loc[harmonized_city['AJZAE001'] < 
            np.nanpercentile(harmonized_city['AJZAE001'], 10)]
        ax2.plot(mostwealthy['PRENO2'].mean(), i-os, 'o', color=color_white, 
            zorder=12)
        ax2.plot(leastwealthy['PRENO2'].mean(), i-os, 'o', color=color_non, 
            zorder=12)
        ax2.plot((np.min([mostwealthy['PRENO2'].mean(),
            leastwealthy['PRENO2'].mean()]), np.min([mostwealthy[
            'PRENO2'].mean(),leastwealthy['PRENO2'].mean()])+np.abs(np.diff(
            [mostwealthy['PRENO2'].mean(), leastwealthy['PRENO2'].mean()]))[0]), 
            [i-os,i-os], color='k', ls='-', zorder=10)    
        ax2.plot(mostwealthy['POSTNO2'].mean(), i+os, 'o', color=color_white, 
            zorder=12)
        ax2.plot(leastwealthy['POSTNO2'].mean(), i+os, 'o', color=color_non, 
            zorder=12)
        ax2.plot((np.min([mostwealthy['POSTNO2'].mean(),
            leastwealthy['POSTNO2'].mean()]), np.min([
            mostwealthy['POSTNO2'].mean(),leastwealthy['POSTNO2'].mean()])+
            np.abs(np.diff([mostwealthy['POSTNO2'].mean(), 
            leastwealthy['POSTNO2'].mean()]))[0]), [i+os,i+os], color='k', 
            ls='--', zorder=10)
        yticks.append(np.nanmean([i]))
        ratio_pre.append(leastwealthy['PRENO2'].mean()/mostwealthy['PRENO2'].mean())
        ratio_post.append(leastwealthy['POSTNO2'].mean()/mostwealthy['POSTNO2'].mean())
        i = i+7
    print('Disparities in 15 largest MSAs for income:')
    print('Baseline %.3f +/- %.3f'%(np.nanmean(ratio_pre),np.nanstd(ratio_pre)))
    print('Lockdown %.3f +/- %.3f'%(np.nanmean(ratio_post),np.nanstd(ratio_post)))
    print('\n')
    # Aesthetics 
    ax2.set_xlim([0.5e15,10e15])
    ax2.set_xticks(np.arange(1e15,10e15,1e15))
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
    ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
    ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
    ax3.plot((np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()]), 
        np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()])+
        np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    ax3.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
        np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
        np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
        [i+os,i+os], color='k', ls='--', zorder=10)
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
    ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
    ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
    ax3.plot((np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()]), 
        np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()])+
        np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    ax3.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
        np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
        np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
        [i+os,i+os], color='k', ls='--', zorder=10)
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
    ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, zorder=12)
    ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, zorder=12)
    ax3.plot((np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()]), 
        np.min([mosteducated['PRENO2'].mean(),leasteducated['PRENO2'].mean()])+
        np.abs(np.diff([mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
        [i-os,i-os], color='k', ls='-', zorder=10)    
    ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, zorder=12)
    ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, zorder=12)
    ax3.plot((np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()]), 
        np.min([mosteducated['POSTNO2'].mean(),leasteducated['POSTNO2'].mean()])+
        np.abs(np.diff([mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
        [i+os,i+os], color='k', ls='--', zorder=10)
    yticks.append(np.nanmean([i]))
    i = i+7    
    ratio_pre = []
    ratio_post = []
    for city in [newyork, losangeles, chicago, dallas, houston, washington,
        miami, philadelphia, atlanta, phoenix, boston, sanfrancisco, 
        riverside, detroit, seattle]:
        harmonized_city = subset_harmonized_bycountyfips(harmonized, city)
        frac_educated = (harmonized_city.loc[:,'AJYPE019':'AJYPE025'].sum(axis=1)/
            harmonized_city['AJYPE001'])
        mosteducated = harmonized_city.iloc[np.where(frac_educated > 
            np.nanpercentile(frac_educated, 90))]
        leasteducated = harmonized_city.iloc[np.where(frac_educated < 
            np.nanpercentile(frac_educated, 10))]
        ax3.plot(mosteducated['PRENO2'].mean(), i-os, 'o', color=color_white, 
            zorder=12)
        ax3.plot(leasteducated['PRENO2'].mean(), i-os, 'o', color=color_non, 
            zorder=12)
        ax3.plot((np.min([mosteducated['PRENO2'].mean(),
            leasteducated['PRENO2'].mean()]), 
            np.min([mosteducated['PRENO2'].mean(),
            leasteducated['PRENO2'].mean()])+np.abs(np.diff(
            [mosteducated['PRENO2'].mean(), leasteducated['PRENO2'].mean()]))[0]), 
            [i-os,i-os], color='k', ls='-', zorder=10)    
        ax3.plot(mosteducated['POSTNO2'].mean(), i+os, 'o', color=color_white, 
            zorder=12)
        ax3.plot(leasteducated['POSTNO2'].mean(), i+os, 'o', color=color_non, 
            zorder=12)
        ax3.plot((np.min([mosteducated['POSTNO2'].mean(), leasteducated[
            'POSTNO2'].mean()]), np.min([mosteducated['POSTNO2'].mean(), 
            leasteducated['POSTNO2'].mean()])+np.abs(np.diff(
            [mosteducated['POSTNO2'].mean(), leasteducated['POSTNO2'].mean()]))[0]), 
            [i+os,i+os], color='k', ls='--', zorder=10)
        yticks.append(np.nanmean([i]))
        ratio_pre.append(leasteducated['PRENO2'].mean()/mosteducated['PRENO2'].mean())
        ratio_post.append(leasteducated['POSTNO2'].mean()/mosteducated['POSTNO2'].mean())    
        i = i+7
    print('Disparities in 15 largest MSAs for education:')
    print('Baseline %.3f +/- %.3f'%(np.nanmean(ratio_pre),np.nanstd(ratio_pre)))
    print('Lockdown %.3f +/- %.3f'%(np.nanmean(ratio_post),np.nanstd(ratio_post)))
    print('\n')
    ax3.set_xlim([0.5e15,10e15])
    ax3.set_xticks(np.arange(1e15,10e15,1e15))
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
    ax2.set_title('(b) Household income', loc='left', fontsize=10)
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
    plt.savefig(DIR_FIGS+'fig2.pdf', dpi=1000)
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
        harmonized_city = subset_harmonized_bycountyfips(harmonized, city)
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
        color=color_density, clip_on=False)
    ax1.fill_between(np.arange(0,10,1),ci_low, ci_high, facecolor=color_density,
        alpha=0.2)
    ax1.plot(priroad_byincome, ls='-', lw=2, color=color1, zorder=11)
    ax1.plot(priroad_byeducation, ls='-', lw=2, color=color2, zorder=11)
    ax1.plot(priroad_byrace, ls='-', lw=2, color=color3, zorder=11)
    ax1.plot(priroad_byethnicity, ls='-', lw=2, color=color4, zorder=11)
    ax1.plot(priroad_byvehicle, ls='-', lw=2, color=color5, zorder=11)
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
    plt.savefig(DIR_FIGS+'fig3.pdf',dpi=1000)
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
    ax1 = plt.subplot2grid((3,3),(0,0), projection=ccrs.PlateCarree(
        central_longitude=0.))
    ax2 = plt.subplot2grid((3,3),(0,1), projection=ccrs.PlateCarree(
        central_longitude=0.))
    ax3 = plt.subplot2grid((3,3),(0,2), projection=ccrs.PlateCarree(
        central_longitude=0.))
    ax4 = plt.subplot2grid((3,3),(1,0), projection=ccrs.PlateCarree(
        central_longitude=0.))
    ax5 = plt.subplot2grid((3,3),(1,1), projection=ccrs.PlateCarree(
        central_longitude=0.))
    ax6 = plt.subplot2grid((3,3),(1,2), projection=ccrs.PlateCarree(
        central_longitude=0.))
    ax7 = plt.subplot2grid((3,3),(2,0), projection=ccrs.PlateCarree(
        central_longitude=0.))
    ax8 = plt.subplot2grid((3,3),(2,1), projection=ccrs.PlateCarree(
        central_longitude=0.))
    ax9 = plt.subplot2grid((3,3),(2,2), projection=ccrs.PlateCarree(
        central_longitude=0.))
    axes = [ax1, ax4 , ax7, ax2, ax5, ax8, ax3, ax6, ax9]
    fips = [['36','34'], ['13'], ['26']] # NY, GA, MI
    extents = [[-74.05, -73.8, 40.68, 40.9], # New York City
        [-84.54, -84.2, 33.6, 33.95], # Atlanta
        [-83.35,-82.87,42.2,42.5]] # Detroit
    citycodes = [['36047','36081','36061','36005','36085','36119', '34003', 
        '34017','34031','36079','36087','36103','36059','34023','34025',
        '34029','34035','34013','34039','34027','34037'	,'34019'],
        # New York-Newark-Jersey City, NY-NJ-PA MSA
        ['13121','13135','13067','13089','13063','13057','13117', '13151'],
        # Atlanta-Sandy Springs-Alpharetta, GA MSA             
        ['26163','26125','26099','26093','26147','26087']]    
        # Detroit-Warren-Dearborn, MI MSA
    ticks = [np.linspace(-2e15,2e15,5),
        np.linspace(-0.8e15,0.8e15,5),
        np.linspace(-0.5e15,0.5e15,5)]
    ticklabels = [['%.2f'%x for x in np.linspace(-2,2,5)],
        ['%.2f'%x for x in np.linspace(-0.8,0.8,5)],
        ['%.2f'%x for x in np.linspace(-0.5,0.5,5)]]
    # Load U.S. counties
    reader = shapereader.Reader(DIR_GEO+'counties/tl_2019_us_county/'+
        'tl_2019_us_county')
    counties = list(reader.geometries())
    counties = cfeature.ShapelyFeature(np.array(counties, dtype=object), proj)
    # Colormaps    
    cmapno2 = plt.get_cmap('coolwarm', 16)
    normno2 = [matplotlib.colors.Normalize(vmin=-2e15, vmax=2e15),
        matplotlib.colors.Normalize(vmin=-0.8e15, vmax=0.8e15),
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
        harmonized_city = subset_harmonized_bycountyfips(harmonized, 
            citycodes[i])    
        # Find indicies of ~city
        down = (np.abs(lat_dg-extents[i][2])).argmin()
        up = (np.abs(lat_dg-extents[i][3])).argmin()
        left = (np.abs(lng_dg-extents[i][0])).argmin()
        right = (np.abs(lng_dg-extents[i][1])).argmin()
        # Calculate ~city average change in NO2 during lockdowns
        diff_cityavg = (np.nanmean(no2_post_dg[down:up+1,left:right+1])-
            np.nanmean(no2_pre_dg[down:up+1,left:right+1]))
        # Open shapefile for state; if more than one state, loop through 
        # through 
        sf = fips[i]
        records, tracts = [], []
        for sfi in sf:
            shp = shapereader.Reader(DIR_GEO+'tigerline/'+
                'tl_2019_%s_tract/tl_2019_%s_tract'%(sfi,sfi))
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
        # Plot oversampled NO2
        mba = axa.pcolormesh(lng_dg[left:right+1], lat_dg[down:up+1], 
            (no2_post_dg[down:up+1,left:right+1]-
            no2_pre_dg[down:up+1,left:right+1])-diff_cityavg, 
            cmap=cmapno2, norm=normno2[i], transform=proj)        
        # Add colorbars
        # Delta NO2
        divider = make_axes_locatable(axa)
        cax = divider.append_axes('right', size='5%', pad=0.1, 
            axes_class=plt.Axes)
        cbar = fig.colorbar(mba, cax=cax, orientation='vertical', ticks=ticks[i],
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
                'tl_2019_%s_prisecroads'%sfi)   
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
    ax1.text(-0.1, 0.5, '$\Delta$ NO$_{2\mathregular{,\:local}}$/10$'+
        '^\mathregular{15}$\n[molec cm$^{\mathregular{-2}}$]', 
        ha='center', rotation='vertical', rotation_mode='anchor',
        transform=ax1.transAxes, fontsize=12)
    ax4.text(-0.1, 0.5, 'Income [$]', ha='center', rotation='vertical', 
        rotation_mode='anchor', transform=ax4.transAxes, fontsize=12)
    ax7.text(-0.1, 0.5, 'White [%]', ha='center', rotation='vertical', 
        rotation_mode='anchor', transform=ax7.transAxes, fontsize=12)
    for ax in axes: 
        ax.set_aspect('auto')
        ax.outline_patch.set_zorder(20)
    plt.subplots_adjust(left=0.07, top=0.95, bottom=0.05, wspace=0.3)
    plt.savefig(DIR_FIGS+'fig4.png', dpi=1000)
    return

def figS1(harmonized, harmonized_rural):
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
    ax1.text(decreaseno2_all['NO2_ABS'].mean()+2e13, 2, '-2.07', color='k', 
        va='center')
    print('Mean gains for NO2 = %.2E'%Decimal(harmonized['NO2_ABS'].mean()))
    ax1.text(harmonized['NO2_ABS'].mean()+2e13, 1, '-0.43', color='k', 
        va='center')
    print('Smallest gains for NO2 = %.2E'%Decimal(
        increaseno2_all['NO2_ABS'].mean()))
    ax1.text(increaseno2_all['NO2_ABS'].mean()+2e13, 0, '0.23', color='k', 
        va='center')
    ax1b.barh([2,1,0], [decreaseno2_rural['NO2_ABS'].mean(), harmonized_rural[
        'NO2_ABS'].mean(), increaseno2_rural['NO2_ABS'].mean()], 
        color=colors[0])
    print('Largest gains for NO2 = %.2E'%Decimal(
        decreaseno2_rural['NO2_ABS'].mean()))
    ax1b.text(decreaseno2_rural['NO2_ABS'].mean()+10e13, 2, '-0.79', color='k', 
        va='center')
    print('Mean gains for NO2 = %.2E'%Decimal(
        harmonized_rural['NO2_ABS'].mean()))
    ax1b.text(harmonized_rural['NO2_ABS'].mean()-2.5e14, 1, '-0.15', color='k', 
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
        ax.set_xlim(-2.1e15,3e14)
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
    ax3.set_title('(c) Household income [$]', loc='left', fontsize=10)
    ax3b.set_title('(i) Household income [$]', loc='left', fontsize=10)
    ax4.set_title('(d) Educational attainment [%]',loc='left', fontsize=10)
    ax4b.set_title('(j) Educational attainment [%]',loc='left', fontsize=10)
    ax5.set_title('(e) Racial background [%]', loc='left', fontsize=10)
    ax5b.set_title('(k) Racial background [%]', loc='left', fontsize=10)
    ax6.set_title('(f) Household vehicle ownership [%]',loc='left', fontsize=10)
    ax6b.set_title('(l) Household vehicle ownership [%]',loc='left', fontsize=10)
    fig.text(0.5, 0.98, '$\mathbf{All}$', fontsize=14, ha='center')
    fig.text(0.5, 0.48, '$\mathbf{Rural}$', fontsize=14, ha='center')
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=3)
    plt.savefig(DIR_FIGS+'figS1.pdf', dpi=1000)
    return 


def figS2(harmonized, harmonized_rural, harmonized_urban):
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
    ax1.set_ylabel('Income', fontsize=12)
    ax2.set_ylabel('Racial background', fontsize=12)
    ax3.set_ylabel('Ethnic background', fontsize=12)
    ax4.set_ylabel('Educational\nattainment', fontsize=12)
    ax5.set_ylabel('Vehicle\nownership', fontsize=12)
    # Axes ticks        
    for ax in [ax1,ax2,ax3,ax4,ax5]:
        ax.set_xlim([0.95e15,4.1e15])
        ax.set_xticks([1e15,2e15,3e15,4e15])
        ax.set_xticklabels([])
    ax5.set_xticklabels(['1','2','3','4'])
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
    plt.savefig(DIR_FIGS+'figS2.pdf', dpi=1000)
    plt.show()
    return

def figS3(harmonized_urban):
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
        state_harm_i = pd.read_csv(DIR_HARM+'Tropomi_NO2_updated/'+
            'Tropomi_NO2_updated_%s_nhgis%s_tract.csv'%(FIPS_i, nhgis_version), 
            delimiter=',', header=0, engine='python')
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
        bpl = ax.boxplot([left['ALLNO2'].values[~np.isnan(left['ALLNO2'].values)], 
            left['PRENO2APR'].values[~np.isnan(left['PRENO2APR'].values)],
            left['PRENO2_x'][~np.isnan(left['PRENO2_x'])]], positions=[1,2,3],
            whis=[10,90], showfliers=False, patch_artist=True, showcaps=False)
        bpr = ax.boxplot([right['ALLNO2'].values[~np.isnan(right['ALLNO2'].values)], 
            right['PRENO2APR'].values[~np.isnan(right['PRENO2APR'].values)],
            right['PRENO2_x'][~np.isnan(right['PRENO2_x'])]], positions=[5,6,7],
            whis=[10,90], showfliers=False, patch_artist=True, showcaps=False)    
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
    ax1.set_title('(a) Household income', loc='left', fontsize=10)
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
    plt.savefig(DIR_FIGS+'figS3.pdf', dpi=1000)
    return
    
def figS4():
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
    import sys
    sys.path.append('/Users/ghkerr/phd/utils/')
    from geo_idx import geo_idx
    PATH_AQS = '/Users/ghkerr/GW/data/aqs/'
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
    ax.text(0.05e16, 20.5, 'm = 1.5 x 10$^{-15}$ ppbv (molec cm$^{-2}$)$^{-1}$'+
        '\nb = %.1f'%b, color=color_non, fontsize=12)
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
    plt.savefig(DIR_FIGS+'figS4.pdf', dpi=1000)
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
# # Add vehicle ownership/road density data
# harmonized = tropomi_census_utils.merge_harmonized_vehicleownership(harmonized)
# # Split into rural and urban tracts
# harmonized_urban, harmonized_rural = \
#     tropomi_census_utils.split_harmonized_byruralurban(harmonized)

# # Calculate percentage of tracts without co-located TROPOMI retrievals 
# print('%.1f of all tracts have NO2 retrievals'%(len(np.where(np.isnan(
#     harmonized['PRENO2'])==False)[0])/len(harmonized)*100.))
# print('%.1f of urban tracts have NO2 retrievals'%(len(np.where(np.isnan(
#     harmonized_urban['PRENO2'])==False)[0])/len(harmonized_urban)*100.))
# print('%.1f of rural tracts have NO2 retrievals'%(len(np.where(np.isnan(
#     harmonized_rural['PRENO2'])==False)[0])/len(harmonized_rural)*100.))

# # Figures
# fig1(harmonized)
# fig2(harmonized, harmonized_rural, harmonized_urban)
# fig3(harmonized_urban)
# fig4(harmonized, lat_dg, lng_dg, no2_post_dg, no2_pre_dg) 
# figS1(harmonized, harmonized_rural)
# figS2(harmonized, harmonized_rural, harmonized_urban)
# figS3(harmonized_urban)
# figS4()








