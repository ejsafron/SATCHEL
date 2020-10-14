#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python ver. 3.6.5
Created on Tue May  5 10:13:28 2020

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
import timeit

# For easy replacement
user_directory = '/home/safron/Documents/PH/master/'    # sif
#user_directory = '/home/esafron/Documents/PH/'    # masotan


start_time = timeit.default_timer()


''' Changeable pipeline parameters '''
tolerance = 0.7     # How close (in days) the midpoints of two user markings must be if indicating the same feature
overlap = 0.75       # Fraction that two user markings must overlap to be considered the same
dupe_overlap = 0.75  # Fraction that two metafeatures must overlap to be considered duplicates
cutoff = 6.0        # Duration (in days) of the longest user marking we're willing to assume was made intentionally


''' Define functions '''
# This is slightly more forgiving than the true overlap
def fracoverlap(user1x, user2x):
    ''' Returns the fraction of overlapping areas '''
    dur1 = user1x[1] - user1x[0]
    dur2 = user2x[1] - user2x[0]
    return (max(0, min(user1x[1], user2x[1]) - max(user1x[0], user2x[0])))/max([dur1, dur2])


''' Read in data '''
# All M dwarf classifications
db = pd.read_csv(user_directory+'mdwarf-classifs.csv')

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
    subidclasses = db[db['subject_id']==setsubjectids[i]]  # All classifications of lightcurve[i]
    subidclasses = subidclasses[subidclasses.xMinGlobal.notnull()]  # Filter out classifications where users made no mark
    indices = []    # Record which indices in new lists the following metafeatures correspond to
    for j in range(len(subidclasses)):
        # Mark [j] is the base mark we'll compare the other marks to for matches
        transitid += 1
        xmin = subidclasses['xMinGlobal'].iloc[j]
        xmax = subidclasses['xMaxGlobal'].iloc[j]
        midpoint = np.mean([xmin,xmax])
        mark = [xmin,xmax]
        matchlist_xmin = [xmin]
        matchlist_xmax = [xmax]
        if (xmax-xmin) >= cutoff:   # Filter out very wide marks
            continue
        else:
            for k in range(len(subidclasses)):
                if k == j:
                    continue    # Don't compare a mark to itself
                elif (xmax-xmin) >= cutoff:   # Continue to ignore very wide marks
                    continue
                else:
                    testxmin = subidclasses['xMinGlobal'].iloc[k]
                    testxmax = subidclasses['xMaxGlobal'].iloc[k]
                    testmidpoint = np.mean([testxmin,testxmax])
                    testmark = [testxmin,testxmax]
                    if (np.isclose(midpoint, testmidpoint, rtol = tolerance)==True) & (fracoverlap(mark, testmark) >= overlap):    # If marks indicate same feature, add mark bounds to matchlist
                        matchlist_xmin.append(testxmin)
                        matchlist_xmax.append(testxmax)
                    elif (np.isclose(midpoint, testmidpoint, rtol = tolerance)==False) & (fracoverlap(mark, testmark) >= overlap):   # If marks pass overlap test but fail midpoint test
                        ''' Uncomment these to forgive failure of the midpoint test '''
#                        matchlist_xmin.append(testxmin)
#                        matchlist_xmax.append(testxmax)
                        continue
                    elif (np.isclose(midpoint, testmidpoint, rtol = tolerance)==False) & (fracoverlap(mark, testmark) < overlap):   # If marks fail both tests, do not match
                        continue
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
            
    # Once all j indices have been iterated to consolidate marks, we must test for duplicates
    # For this, we use the list indices that we've recorded
    # And we'll initiate a second consolidation process, to merge duplicate metafeatures
    # It's important to keep a very high overlap fraction requirement (dupe_overlap) for this process, so that the metafeature doesn't get shifted around a lot
    # It's okay if that means that there's a significantly higher number of metafeatures in the final list, since previous tests have indicated higher recovery rates in those cases anyway
    # We'll keep track of which ones have already been averaged and overwritten in a temp list
    done = []
    for index in indices:
        if np.isnan(midpoints[index])==True:    # True if index has already been marked as a dupe
            continue
        else:
            duplicates = [index]
            for l in range(len(indices)):   # For each index, cycle through other indices to compare marks
                if index == indices[l]:
                    continue    # Don't compare a mark to itself
                elif indices[l] in done:
                    continue    # Don't compare a mark to one that's already been checked for matches
                elif np.isnan(midpoints[indices[l]])==True:
                    continue    # Don't compare a mark to a dupe
                else:
                    basemark = [xmins[index], xmaxs[index]]
                    testmark = [xmins[indices[l]], xmaxs[indices[l]]]
                    if (np.isclose(midpoints[index], midpoints[indices[l]], rtol = tolerance)==True) & (fracoverlap(basemark,testmark) >= dupe_overlap):    # Only satisfaction of both tests indicate duplicate metafeatures
                        duplicates.append(indices[l])
                    else:   # Otherwise, do not add to duplicates list
                        continue
            done.append(index)
            
            if len(duplicates)>1:   # True if mark has duplicates
                # Replace row corresponding to first duplicate (index) with average values, then mark the other rows for popping later
                newxmin = np.mean(np.asarray(xmins)[duplicates])
                newxmax = np.mean(np.asarray(xmaxs)[duplicates])
                xmins[index] = newxmin
                xmaxs[index] = newxmax
                durations[index] = newxmax-newxmin
                midpoints[index] = np.mean([newxmin,newxmax])
                duplicates.pop(0)   # Remove original index from duplicates list
                for dupe_index in duplicates:    # Mark so loop knows to skip these
                    durations[dupe_index],midpoints[dupe_index] = np.nan,np.nan
                    dupe_flag[dupe_index] = 'dupe of '+str(transitids[index])
            else:
                continue            
    if i%10==0:
        print('{0:0.2f}'.format(100*float(i)/float(len(setsubjectids)))+'% completed.')


# Build dataframe
metafeatures = pd.DataFrame(np.column_stack([transitids, lightcurves, kicids, xmins, xmaxs, durations, midpoints, synth_flag, dupe_flag]), columns=['transitids', 'lightcurves', 'kicids', 'xmin', 'xmax', 'durations', 'midpoints', 'synth_flag', 'dupe_flag'])

# Export to csv
metafeatures.to_csv(user_directory+'metafeatures.csv', index=False)


print('Transits for scoring isolated.')
elapsed = timeit.default_timer() - start_time
print('Run time = '+str(elapsed)+' sec.')




''' End program. '''