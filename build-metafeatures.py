#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python ver. 3.6.5
Created on Thu Oct  1 18:25:40 2020

@author: Emily Safron

For the purpose of filtering out signals from known short-period objects and still being left with those from potentially unknown long-period objects, we must score by the transit feature, rather than by the light curve.  Though this is computationally intensive, it will give us a very straightforward way to filter our results, in the end.

To score these features, we first need an iterable list of them.  For this, we go back to the user marks themselves.  We go through all the global xmin and xmax ranges, mark by mark, and quantify by measure of overlap fraction and closeness of marking midpoint, which chunks of x-range indicate markings which are "alike" enough to be consolidated together.  The consolidated markings, which we call metafeatures, are written and saved to a csv.

This code takes as input:
    mdwarf-classifs.csv

and produces as output:
    metafeatures.csv.

"""
# combining mfs w/ >75% overlap
# 717273 metafeatures
# 31148 seconds

# overlap req. = 0.75
# 942831 mfs
# 19830 sec

# overlap req. = 0.6, combining mfs w/ >75% overlap
# 675831 mfs
# 33344 sec


# This edited version of the metafeature list builder DOES allow a single user mark to contribute to multiple metafeatures.


import numpy as np
import pandas as pd
import os
from astropy.io import fits
import timeit

# For easy replacement
#user_directory = '/home/safron/Documents/PH/master/'    # sif
user_directory = '/home/esafron/Documents/PH/master/'    # masotan


start_time = timeit.default_timer()


''' Changeable pipeline parameters '''
tolerance = 1.0     # How close (in days) the midpoints of two user markings must be if indicating the same feature
single = 2.0        # Maximum duration (in days) of consolidatable marks; marks wider than this will automatically be their own metafeatures
cutoff = 6.0        # Duration (in days) of the longest user marking we're willing to assume was made intentionally



''' Read in data '''
# All M dwarf classifications
#db = pd.read_csv(user_directory+'mdwarf-classifs.csv')                 # sif
db = pd.read_csv('/home/esafron/Documents/PH/mdwarf-classifs.csv')      # masotan


# FOR TESTING:  ONLY KOI TARGETS
#kt = pd.read_csv('/home/esafron/Documents/PH/knowntransits-nea.csv')

print('Completed setting up list of data arrays.')



''' Set up the transit score table '''
# Get list of user transit markings for each non-synthetic light curve
setsubjectids = list(set(db['subject_id']))  # Set of all unique lightcurve ids INCLUDING SYNTHETICS

# FOR TESTING:
#setsubjectids = list(set(kt['curveid']))


transitids,lightcurves,kicids,xmins,xmaxs,durations,midpoints,synth_flag,dupe_flag = [],[],[],[],[],[],[],[],[]
transitid = 0
for i in range(len(setsubjectids)):
    subidclasses = db[db['subject_id']==setsubjectids[i]]                       # All classifications of lightcurve[i]
    subidclasses = subidclasses[subidclasses.xMinGlobal.notnull()]              # Filter out classifications where users made no mark
    
    if len(subidclasses)==0:                                                    # If this leaves no classification rows left
        continue                                                                # Move to next subject id
    else:
        # Test point to find right lightcurve file
        getrightcurve = subidclasses['start_time'].iloc[0] + 5.                 # Start time of PH subject, +5 days to account for possible shifting
        
        kepid = subidclasses['kepler_id'].iloc[0]
        dirname = '/home/esafron/Documents/PH/kep-data/'+str(kepid).zfill(9)                 # Get name of directory containing lightcurve data
        lcfilenames = [f for x in os.walk(dirname) for f in x[2] if f.endswith('llc.fits')]  # Make list of files in directory
        for z in range(len(lcfilenames)):                                                    # For each lightcurve file
            curvedata = fits.open(dirname+'/'+lcfilenames[z])                                # Extract lightcurve data
            img = curvedata[1].data
            flux = img['PDCSAP_FLUX']                                           # Extract flux
            time = img['TIME']                                                  # Extract times
            timerange = [time[0], time[-1]]                                     # Build time range of lightcurve
    
            if (timerange[0] < getrightcurve) & (getrightcurve < timerange[1]):     # If test point falls within time range of lightcurve
                break                                                               # Break from z loop, keep lightcurve parameters
            else:                                                                   # If midpoint still lies outside time range of lightcurve
                continue                                                            # Test next lightcurve file
                
        # Once we've got the right Kepler lightcurve file, start testing classification markings
        # Beware, if last time point is a nan
        timecopy = np.linspace(time[0], time[-1], len(time))

        # Find average time between points (to find numpoints that should be in each user mark)
        calcdelta = []
        for m in range(len(timecopy)-1):
            calcdelta.append(timecopy[m+1]-timecopy[m])
        delta = np.mean(calcdelta)
        
        done = []       # Record indices of marks that have already been consolidated
    
        indices = []    # Record which indices in new lists the following metafeatures correspond to
        
        for j in range(len(subidclasses)):
            # Mark [j] is the base mark we'll compare the other marks to for matches
            transitid += 1
            xmin = subidclasses['xMinGlobal'].iloc[j]
            xmax = subidclasses['xMaxGlobal'].iloc[j]
            midpoint = np.mean([xmin,xmax])
            duration = xmax - xmin
            
            numpoints = int(duration / delta)                                   # Count real values in kep lightcurve between xmin and xmax
            numdata = sum(not np.isnan(x) for x in time if xmin <= x <= xmax)   # Count real data in range of mark duration
            datafrac = float(numdata)/float(numpoints)                          # Calculate real data fraction
            
            matchidx = [j]                                                          
                    
            if duration >= cutoff:                                              # Filter out marks wider than cutoff
                continue
            elif j in done:                                                     # Skip marks that have already been consolidated
                continue
            elif datafrac < 0.5:
                continue                                                        # Filter out marks for which > 50% lie in data gaps
            elif duration > single:
                pass                                                            # Don't consolidate if duration > consolidation cutoff
            else:
                for k in range(len(subidclasses)):
                    testxmin = subidclasses['xMinGlobal'].iloc[k]
                    testxmax = subidclasses['xMaxGlobal'].iloc[k]
                    testmidpoint = np.mean([testxmin,testxmax])
                    testduration = testxmax - testxmin
                    
                    testnumpoints = testduration / delta
                    testnumdata = sum(not np.isnan(x) for x in time if testxmin <= x <= testxmax)
                    testdatafrac = float(testnumdata)/float(testnumpoints)
    
                    if k == j:                                                  # Don't compare a mark to itself
                        continue
                    elif k in done:                                             # Don't compare a mark to another mark that's already been used
                        continue
                    elif testduration > single:                                 # Ignore very wide marks
                        continue
                    elif testdatafrac < 0.5:                                    # Don't look at marks for which > 50% lie in data gaps
                        continue
                    else:
                        testmark = [testxmin,testxmax]
                        if np.abs(midpoint-testmidpoint) > tolerance:           # If matching criteria is not met
                            continue                                            # Move to next test index
                        elif np.abs(midpoint-testmidpoint) <= tolerance:        # If matching criteria is met
                            matchidx.append(k)                                  # Add subidclass index to matchlist
                
                # After we've found all the "matches" to the mark,
                # we have to make sure all marks match with each other as well,
                # and that none match with marks outside the current match set.
                #  |\\\|XX|///|XX|\\\|
                if len(matchidx)==1:                                            # If the original index is the only one in the match list
                    pass                                                        # Skip right to the metafeature tabulation
                else:
                    matchidx_new = list(matchidx)                               # Indices of bad usermarks will be popped from this copy
                    for y in range(len(matchidx)-1):
                        # Make y the base mark, rerun test
                        idx = matchidx[y+1]     # skip the original mark
                        xmin_y = subidclasses['xMinGlobal'].iloc[idx]
                        xmax_y = subidclasses['xMaxGlobal'].iloc[idx]
                        midpoint_y = np.mean([xmin_y,xmax_y])
                        duration_y = xmax_y - xmin_y
                        
                        # This mark has already been tested for datagap placement
                        # and duration
                            
                        matchidx_y = [idx]                                                          
                        
                        for yk in range(len(subidclasses)):
                            testxmin_y = subidclasses['xMinGlobal'].iloc[yk]
                            testxmax_y = subidclasses['xMaxGlobal'].iloc[yk]
                            testmidpoint_y = np.mean([testxmin_y,testxmax_y])
                            testduration_y = testxmax_y - testxmin_y
                            
                            testnumpoints_y = testduration_y / delta
                            testnumdata_y = sum(not np.isnan(x) for x in time if testxmin_y <= x <= testxmax_y)
                            testdatafrac_y = float(testnumdata_y)/float(testnumpoints_y)
            
                            # Same tests as before, except we DO want to allow it to check marks that are already "done"
                            
                            if yk == idx:                                       # Don't compare a mark to itself
                                continue
                            elif testduration_y >= cutoff:                      # Continue to ignore very wide marks
                                continue
                            elif testdatafrac_y < 0.5:                          # Don't look at marks for which > 50% lie in data gaps
                                continue
                            else:
                                testmark_y = [testxmin_y,testxmax_y]
                                if np.abs(midpoint_y-testmidpoint_y) > tolerance:   # If matching criteria is not met
                                    continue                                        # Move to next test index
                                else:                                               # If matching criteria is met
                                    matchidx_y.append(yk)                           # Add subidclass index to matchlist
                                    
                        # Match lists will not be in the same order
                        if set(matchidx) == set(matchidx_y):                                # If matchlists are the same
                            continue                                                        # Leave that usermark in the original match list
                        elif set(matchidx) != set(matchidx_y):                              # If any usermarks are added or missing
                            popidx = int(np.where(np.asarray(matchidx_new)==idx)[0][0])     # Find index of bad mark in match list copy
                            matchidx_new.pop(popidx)                                        # Remove that index
                    
                    matchidx = list(matchidx_new)                               # Redefine the match list
                    
            matchlist_xmin, matchlist_xmax = [], []
            for l in range(len(matchidx)):                                      # For all indices left in the matchlist, including the original
                idx_l = int(matchidx[l])                                        # Make sure index is an integer
                done.append(idx_l)                                              # Add index to done list
                matchlist_xmin.append(subidclasses['xMinGlobal'].iloc[idx_l])       # Get all xmin
                matchlist_xmax.append(subidclasses['xMaxGlobal'].iloc[idx_l])       # Get all xmax
                
            
            # Construct a consolidated mark from averages of the match xmin and xmax values
            avgxmin = np.mean(matchlist_xmin)
            avgxmax = np.mean(matchlist_xmax)
            # Record new metafeature properties in each appropriate list
            transitids.append(transitid)
            lightcurves.append(setsubjectids[i])
            kicids.append(subidclasses['kepler_id'].iloc[j])
            xmins.append(avgxmin)
            xmaxs.append(avgxmax)
            durations.append(avgxmax-avgxmin)
            midpoints.append(np.mean([avgxmin,avgxmax]))
            synth_flag.append(subidclasses['synthetic'].iloc[j])
            dupe_flag.append(' ')
            indices.append(len(transitids)-1)
            

    if i%10==0:
        print('{0:0.2f}'.format(100*float(i)/float(len(setsubjectids)))+'% completed.')


# Build dataframe
metafeatures = pd.DataFrame(np.column_stack([transitids, lightcurves, kicids, xmins, xmaxs, durations, midpoints, synth_flag, dupe_flag]), columns=['transitids', 'lightcurves', 'kicids', 'xmin', 'xmax', 'durations', 'midpoints', 'synth_flag', 'dupe_flag'])

# Export to csv
metafeatures.to_csv(user_directory+'metafeatures_new.csv', index=False)


print('Transits for scoring isolated.')
elapsed = timeit.default_timer() - start_time
print('Run time = '+str(elapsed)+' sec.')




''' End program. '''