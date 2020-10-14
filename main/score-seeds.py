#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python ver. 3.6.0
Created on Wed Sep 25 14:08:40 2019

@author: Emily Safron

This code does several things:
    - builds a table for scoring individual transit features from the metafeatures list (metafeatures.csv)
    - calculates a first run of transit scores based on userweights from simulations (with all other users given dummy weights, see build-uwtable.py)

We use asynchronous multiprocess mapping to speed up execution.  Fun!

Required inputs:
    - mdwarf-classifs.csv
    - metafeatures.csv
    - full_uwseeds.csv

Outputs:
    - transitscoreseeds.tsv


"""


import numpy as np
import pandas as pd
from astropy.table import Table
import multiprocessing as mp
import timeit


# For easy replacement
user_directory = '/home/safron/Documents/PH/master/'    # sif


''' Define constants and small functions '''

tolerance = 0.7     # How close (in days) the midpoints of two user markings must be if indicating the same feature
overlap = 0.5       # Fraction that two user markings must overlap to be considered the same
cutoff = 6.0        # Duration (in days) of the longest user marking we're willing to assume was made intentionally

def fracoverlap(user1x, user2x):
    ''' Returns the fraction of overlapping areas '''
    dur1 = user1x[1] - user1x[0]
    dur2 = user2x[1] - user2x[0]
    return (max(0, min(user1x[1], user2x[1]) - max(user1x[0], user2x[0])))/min([dur1, dur2])

def score(Wlist, wlist):
    # Wlist = list of weights of users who examined lightcurve
    # wlist = list of weights of users who agreed/disagreed (scoreyes/scoreno) that the transit occurred
    if len(wlist) == 0:
        return 0.0
    else:
        return 1.0/sum(Wlist) * sum(wlist)


''' Read in data '''

# All M dwarf classifications
db = pd.read_csv(user_directory+'mdwarf-classifs.csv')
setsubjectids = list(set(db['subject_id']))  # Set of all unique lightcurve ids

# Isolated metafeatures for scoring
metafeatures = pd.read_csv(user_directory+'metafeatures.csv')
#metafeatures_wdupes = metafeatures   # Not sure if needed later
metafeatures = metafeatures[np.isnan(metafeatures['midpoints'])==False]


# Userweight table
userweights = pd.read_csv(user_directory+'full_uwseeds.tsv', sep="\t")
allusers = list(userweights['username'])        # useful later



''' Create transit scoring table '''

# Build without usersyes, usersno, weightsyes, weightsno, scoreyes, scoreno, and numclasses columns, to be concatenated later
transit_scores = Table({'transitid':list(metafeatures['transitids']), 'lightcurve':list(metafeatures['lightcurves']), 'kicid':list(metafeatures['kicids']), 'xmin':list(metafeatures['xmin']), 'xmax':list(metafeatures['xmax']), 'duration':list(metafeatures['durations'])}, names=['transitid', 'lightcurve', 'kicid', 'xmin', 'xmax', 'duration'])
transit_scores = transit_scores.to_pandas()



''' Big Function '''

def calc_score_seed(curve_id):
    
    subtransits = transit_scores[transit_scores['lightcurve']==curve_id]
    numtransits = len(subtransits)
    subidclasses = db[db['subject_id']==curve_id]  # All classifications of the lightcurve
    numclasses = len(list(set(subidclasses['classification_id'])))  # Number of unique classifications of the lightcurve (and, therefore, the transit)
    subusers = list(set(subidclasses['user_name']))  # Users who classified the lightcurve
    subuserweights = []
    
    # Fill subuserweights with userweights of all users who classified the lightcurve
    for u in range(len(subusers)):
        subuserweights.append(float(userweights['normcombined'][userweights['username']==subusers[u]]))
    
    # Initialize score list for metafeatures
    scorelist = []
    
    # Big Loop
    for j in range(numtransits):  # For each unscored metafeature in the lightcurve
        usersyes = []       # Users who identified the feature
        weightsyes = []     # Weights of users who identified the feature
        usersno = []        # Users who did NOT identify the feature
        weightsno = []      # Weights of users who did NOT identify the feature
        transitid = int(subtransits['transitid'].iloc[j])
        markxmin = subtransits['xmin'].iloc[j]
        markxmax = subtransits['xmax'].iloc[j]
        markx = [markxmin,markxmax]
        markmidpoint = np.mean(markx)
        idx = transit_scores.loc[transit_scores['transitid']==transitid].index[0]   # Row index of transit_scores table corresponding to transit feature [j]
        for k in range(len(subusers)):  # For each user who classified the lightcurve
            usersubidclasses = subidclasses[subidclasses['user_name']==subusers[k]]    # Classifications of the lightcurve by user[k]
            match = False

            # For each classification[l] of the lightcurve by user[k], search for matches with transit[j]
            # If a match is found, change match = True and break out of l loop
            for l in range(len(usersubidclasses)):
                if np.isnan(usersubidclasses['xMinGlobal'].iloc[l])==True:
                    continue
                else:
                    testxmin = usersubidclasses['xMinGlobal'].iloc[l]
                    testxmax = usersubidclasses['xMaxGlobal'].iloc[l]
                    testx = [testxmin,testxmax]
                    testmidpoint = np.mean(testx)
                    if (np.isclose(markmidpoint, testmidpoint, rtol = tolerance)==True) & (fracoverlap(markx, testx) >= overlap):   # If classif mark matches the transit
                        match = True
                        break
                    else:
                        continue
            if match==True:  # If a user did make a matching mark
                # Add username to usersyes and userweight to weightsyes
                usersyes.append(subusers[k])
                weightsyes.append(float(userweights['normcombined'][userweights['username']==subusers[k]]))
            else:  # If a user did not make a matching mark
                # Add username to usersno and userweight to weightsno
                usersno.append(subusers[k])
                weightsno.append(float(userweights['normcombined'][userweights['username']==subusers[k]]))
    
        # After all users have been tested, build strings to insert into transit scoring table
        # For yes:
        if len(usersyes)!=0:
            usersyesstr = '['+str(usersyes[0])
            weightsyesstr = '['+str(weightsyes[0])
            for l in range(len(usersyes)):
                if l!=0:    # True if there is more than one user with a matching mark
                    usersyesstr = usersyesstr+','+str(usersyes[l])
                    weightsyesstr = weightsyesstr+','+str(weightsyes[l])
            usersyesstr = usersyesstr+']'
            weightsyesstr = weightsyesstr+']'
        else:
            usersyesstr = str(usersyes)
            weightsyesstr = str(weightsyes)
        
        # For no:
        if len(usersno)!=0:     # True if there is at least one user who did not make a matching mark
            usersnostr = '['+str(usersno[0])    # Build string to insert into transit scoring table
            weightsnostr = '['+str(weightsno[0])
            for n in range(len(usersno)):
                if n!=0:    # True if there is more than one user who did not make a matching mark
                    usersnostr = usersnostr+','+str(usersno[n])
                    weightsnostr = weightsnostr+','+str(weightsno[n])
            usersnostr = usersnostr+']'
            weightsnostr = weightsnostr+']'
        else:
            usersnostr = str(usersno)
            weightsnostr = str(weightsno)
            
        # Calculate scores
        scoreyes = score(subuserweights,weightsyes)
        scoreno = score(subuserweights,weightsno)
        
        # Append scores to scorelist
        scorelist.append([usersyesstr, usersnostr, weightsyesstr, weightsnostr, scoreyes, scoreno, numclasses, idx])
        
    return scorelist




''' Initialize multiprocessing pool '''

# Initialize timer
start_time = timeit.default_timer()


# Pool
pool = mp.Pool(mp.cpu_count())
results = []

# Asynchronous mapping
results = pool.map_async(calc_score_seed, [curve for curve in setsubjectids]).get()

# Close pool
pool.close()

# Build column lists of scores
usersyeslist = []
usersnolist = []
weightsyeslist = []
weightsnolist = []
scoreyeslist = []
scorenolist = []
numclasseslist = []
indexlist = []
for i in range(len(results)):
    for j in range(len(results[i])):
        usersyeslist.append(results[i][j][0])
        usersnolist.append(results[i][j][1])
        weightsyeslist.append(results[i][j][2])
        weightsnolist.append(results[i][j][3])
        scoreyeslist.append(results[i][j][4])
        scorenolist.append(results[i][j][5])
        numclasseslist.append(results[i][j][6])
        indexlist.append(results[i][j][7])
        
# Construct score dataframe from columns
score_df = pd.DataFrame(np.column_stack([usersyeslist, usersnolist, weightsyeslist, weightsnolist, scoreyeslist, scorenolist, numclasseslist, indexlist]), columns=['usersyes', 'usersno', 'weightsyes', 'weightsno', 'scoreyes', 'scoreno', 'numclasses', 'indx'])


# Concatenate scores dataframe to big dataframe along column axis
score_df['indx'] = score_df['indx'].astype('int')                   # Change type of indx column to integers for sorting
score_df = score_df.sort_values(by=['indx'], ascending=True)        # Sort by index, to match row index of transit_scores dataframe
score_df = score_df.set_index('indx')                               # Change index column to indx

transit_scores = pd.concat([transit_scores, score_df], axis=1)


# Save to a .tsv file
# NOTE:  MUST be a .tsv, not a .csv.  The usersyes, usersno, weightsyes, and weightsno fields all contain commas that will incorrectly trip the reading of a .csv file.
transit_scores.to_csv(user_directory+'transitscoreseeds.tsv', sep="\t", index=False)


# Get time results
elapsed = timeit.default_timer() - start_time
print('Run time = '+str(elapsed)+' sec.')







""" End program. """
