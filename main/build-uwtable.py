#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:36:55 2020
Python ver. 3.6.0

@author: Emily Safron

This code should run after building the metafeature list and the userweight seeds, but before the coupled iteration of scores and userweights.  It creates the userweight table for iteration.  In addition, this code refines the userweight table to include, for each user, a list of:
        - which metafeature ids they have classified,
        - how many total metafeatures they have classified,
        - and how many total subjects they have classified,
for ease of reference in the iteration code.

This code requires the following files as input:
    mdwarf-classifs.csv
    user-weighting.csv
    metafeatures.csv
    
and outputs the file:
    full-uwseeds.tsv

"""

import numpy as np
import pandas as pd
from astropy.table import Table
import timeit
import multiprocessing as mp



# For easy replacement
user_directory = '/home/safron/Documents/PH/master/11242020/'    # sif



start_time = timeit.default_timer()



''' Import data '''

# All M dwarf classifications
db = pd.read_csv(user_directory+'mdwarf-classifs.csv')
dbsynth = db[db['synthetic']==True]

# Userweights from synthetics
columns = ['username','upweight','downweight','combined','normupweight','normdownweight','normcombined','numclasses']
uw_seeds = pd.read_csv(user_directory+'user-weighting.csv',names=columns)

# The metafeatures
metafeatures = pd.read_csv(user_directory+'metafeatures.csv')
#  ||  feature_id  |  subject_id  |  kicids  |  xmin  |  xmax  |  durations  |  midpoints  |  synth_flag  ||



''' Set up the userweight table for iteration '''
# Setting up the user list and initializing the user weights
# The only things we'll need from the seed table are username, normcombined, and maybe numclasses
# But we need to distinguish the number of subject classifications from the number of metafeature classifications, which we also need
allusers = list(set(db['user_name']))
userweights = Table({'username':allusers,'normcombined':list(1. for i in range(len(allusers))),'numsubjectclasses':np.zeros(len(allusers)),'numfeatureclasses':np.zeros(len(allusers)),'featuresclassified':np.zeros(len(allusers))}, names=['username','normcombined','numsubjectclasses','numfeatureclasses','featuresclassified'])

userweights = userweights.to_pandas()


print("Completed setting up list of data arrays.")


# Fill the userweights table with seed values from synthetics, where applicable
for i in range(len(uw_seeds['username'])):
    user = uw_seeds['username'].iloc[i]         # All users who saw simulations
    if user not in np.asarray(userweights['username']):
        continue
    else:
        userweights['normcombined'][np.where(userweights['username']==user)[0][0]] = uw_seeds['normcombined'].iloc[i]
    if i%10==0:
        print('{0:0.2f}'.format(100*float(i)/float(len(uw_seeds['username'])))+'% completed.')
            
print('Userweight seeds copied.')

# Must make featuresclassified column dtype object so that we can insert lists there
userweights = userweights.astype({'featuresclassified': object})





print("Counting and recording metafeatures classified by users..")


''' Build the list of metafeature IDs classified by simply combining the list of subjects from the classifications db with the unscored metafeatures db. '''


# A function to find the transit IDs classified by a user
def which_features_classified(the_user):
    user_features = []                                                                          # Place to stick the transit IDs as we find them
    subjectids = list(set(db['subject_id'][db['user_name']==the_user]))                         # All lightcurves classified by the user in db
    for i in range(len(subjectids)):
        features_in_subject = list(metafeatures['feature_id'][metafeatures['subject_id']==subjectids[i]])  # List of all transit IDs corresponding to lightcurve
                                                                                                # in metafeature list
        for j in range(len(features_in_subject)):
            user_features.append(features_in_subject[j])                                        # Append each transit ID to the list
    user_features = list(set(user_features))                                                    # Shouldn't be any duplicates, but remove if so
    return [the_user, user_features, len(subjectids)]


# Initialize timer
start_time = timeit.default_timer()


# Initialize pool
pool = mp.Pool(mp.cpu_count())
which_feature_results = []

# Asynchronous mapping, for ALL users
which_feature_results = pool.map_async(which_features_classified, [user for user in allusers]).get()

# Close pool
pool.close()


# Dissect results into easy-to-organize lists, may not be in same order as before because we used async
user_check = []
featuresclassifiedlist = []
featurecountlist = []
subjectcountlist = []
for i in range(len(userweights['username'])):                                   # i is user index in userweights table (same length as allusers list)
    user = userweights['username'].iloc[i]                                      # username string
    idx = np.where(np.asarray(which_feature_results)[:,0]==user)[0][0]          # index location in user_check results list
    user_check.append(which_feature_results[idx][0])                            # username from results list
    featuresclassifiedlist.append(which_feature_results[idx][1])                # list of metafeatures classified
    featurecountlist.append(int(len(which_feature_results[idx][1])))            # number of metafeatures classifed
    subjectcountlist.append(which_feature_results[idx][2])                      # number of lightcurves classifed



# Get time results
elapsed = timeit.default_timer() - start_time
print('Run time = '+str(elapsed)+' sec.')




''' We finally have all the lists we need! '''

# Fold into full userweight seed dataframe
full_uwseeds = pd.DataFrame(np.column_stack([allusers, user_check, list(userweights['normcombined']), subjectcountlist, featurecountlist, featuresclassifiedlist]), columns=['username','user_check','normcombined', 'numsubjectclasses', 'numfeatureclasses', 'featuresclassified'])

# Make numsubjectclasses column into integers
full_uwseeds['numsubjectclasses'] = full_uwseeds.numsubjectclasses.astype(int)

# Save as .tsv file
full_uwseeds.to_csv(user_directory+'full_uwseeds.tsv', sep="\t", index=False)






''' End program. '''
