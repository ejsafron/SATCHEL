#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python ver. 3.6.5
Created on Tue Oct 20 18:16:16 2020

@author: Emily Safron

This pipeline piece calculates preliminary user weight "seeds," which are used to calculate the first round of feature scores.  This is accomplished in two major parts:  Part I consists of matching user markings to synthetic signals in simulations, and Part II is comprised of upweighting, downweighting, and normalization based on the matches from Part I.

A user mark is considered a match to a synthetic signal if it has non-zero overlap with the synthetic signal AND its midpoint is close to that of the synthetic signal within a specified tolerance.  User marks with a duration longer than a specified cutoff are ignored.  These two hyperparameters (tolerance and cutoff) are adjustable near the beginning of the code.

This code takes as input:
    mdwarf-classifs.csv
    syntheticdata.csv

and produces as output:
    match-user-synthetics.csv
    user-weighting.csv

"""


import numpy as np
import pandas as pd
import timeit
from astropy.table import Table


# For easy replacement
user_directory = '/home/safron/Documents/PH/master/11242020/'    # sif


''' Set adjustable pipeline parameters '''
tolerance = 0.6      # How close (in days) the midpoints of two user markings must be if indicating the same feature
cutoff = 3.5         # Duration (in days) of the longest user marking we're willing to assume was made intentionally



start_time = timeit.default_timer()



''' Define functions '''
def funcoverlap(synx, userx):
	'''Returns the total length of the overlapping areas'''
    # synx is a list of the form [beginning of transit, end of transit]
    # userx is a list of the form [beginning of user marking, end of user marking]
	return max(0, min(synx[1], userx[1]) - max(synx[0], userx[0]))



''' Read in data '''
# All M dwarf classifications
db = pd.read_csv(user_directory+'mdwarf-classifs.csv')
db = db[db['synthetic']==True]                                                  # We only need simulations for this

# Get set of all simulation lightcurve ids from db
setsubjectids = list(set(db['subject_id'][db['synthetic']==True]))


''' Read in the synthetic file '''
synthetics = pd.read_csv(user_directory+'syntheticdata.csv')


print('Completed setting up list of data arrays.')



''' Set up the output file '''

matchfilename = user_directory+'match-user-synthetics.csv'
matchfilewrite = open(matchfilename,'w')

# Write header containing column names into line 0
matchfilewrite.write('username,classid,subjectid,signalid,synoverlap,syntheticid,userxmin,userxmax\n')


print("Matching user marks to synthetic signals...")

for i in range(len(setsubjectids)):
    subclasses = db[db['subject_id']==setsubjectids[i]]                         # All classifications of lightcurve[i]
    setsubclassids = list(set(subclasses['classification_id']))                 # List of relevant classification ids
    synthid = subclasses['synthetic_id'].iloc[0]                                # Synthetic ID
    synthsubjectdata = synthetics[synthetics['syntheticid']==synthid]                # Info on all synthetic transits in light curve
        
    for j in range(len(setsubclassids)):
        done = []       # Record indices of marks that have already been consolidated
        
        subclassid = subclasses[subclasses['classification_id']==setsubclassids[j]]     # Only marks from one classification
        
        # Extract user & classification information
        username = subclassid['user_name'].iloc[0]
        classid = subclassid['classification_id'].iloc[0]
        subjectid = subclassid['subject_id'].iloc[0]
        
        if np.isnan(subclassid['xMinGlobal'].iloc[0])==True:                    # If user made no marks on light curve
            # Assign np.nans to everything else
            synoverlap,userxmin,userxmax = np.nan,np.nan,np.nan
            
            for k in range(len(synthsubjectdata)):
                # Extract information about each synthetic
                signalid = synthsubjectdata['signalid'].iloc[k]
                synxmin,synxmax = synthsubjectdata['synxmin'].iloc[k],synthsubjectdata['synxmax'].iloc[k]
                synduration = synxmax-synxmin
                synmidpoint = np.average([synxmin,synxmax])
                
                # Write an "empty" line into the output file
                line = []
                for value in [username,classid,subjectid,signalid,synoverlap,synthid,userxmin,userxmax]:
                    line.append(value)
                for l in range(len(line)):
                    if l < len(line)-1:
                        matchfilewrite.write(str(line[l])+',')
                    else:
                        matchfilewrite.write(str(line[l])+'\n')
        
        else:                                                                   # If the user made at least one mark on the light curve
            for k in range(len(synthsubjectdata)):
                # Extract information about each synthetic
                synxmin,synxmax = synthsubjectdata['synxmin'].iloc[k],synthsubjectdata['synxmax'].iloc[k]
                synduration = synxmax-synxmin
                synmidpoint = np.average([synxmin,synxmax])
                
                idx,alloverlap,allmidpointdiff,alluserxmin,alluserxmax,alluserduration = [],[],[],[],[],[]

                for m in range(len(subclassid)):
                    tmpuserxmin,tmpuserxmax = float(subclassid['xMinGlobal'].iloc[m]),float(subclassid['xMaxGlobal'].iloc[m])
                    tmpuserduration = tmpuserxmax-tmpuserxmin
                    tmpusermidpoint = np.average([tmpuserxmin,tmpuserxmax])
                    idx.append(m)
                    alluserxmin.append(tmpuserxmin)
                    alluserxmax.append(tmpuserxmax)
                    alluserduration.append(tmpuserxmax-tmpuserxmin)
                    alloverlap.append(funcoverlap([tmpuserxmin,tmpuserxmax],[synxmin,synxmax])/synduration)
                    allmidpointdiff.append(np.abs(synmidpoint-tmpusermidpoint))
            
                # Still inside j loop, for specific classification id
                # And k loop, for specific row of synthetic id
                
                usermarksdf = pd.DataFrame(np.column_stack([idx,alluserxmin,alluserxmax,alluserduration,alloverlap,allmidpointdiff]), columns=['idx','userxmin','userxmax','userduration','overlap','midpointdiff'])
                                
                # Find all marks with midpoints within the specified tolerance of the synthetic's midpoint
                matches = usermarksdf[(usermarksdf['overlap']>0) | (usermarksdf['userduration']<tolerance)]
                matches = matches[matches['midpointdiff']<=tolerance]
                
                match = False
                if len(matches)>0:  # True if there is a matching mark
                    # Find index of the closest unused match
                    matches = matches.sort_values(by='midpointdiff')
                    matches = matches.reset_index(drop=True)
                    
                    for n in range(len(matches)):
                        if int(matches['idx'].iloc[n]) in done:
                            continue
                        else:
                            match = True
                            break
                
                if match==False:                                                # If no match or no UNUSED match was found
                    synoverlap,userxmin,userxmax = np.nan,np.nan,np.nan
                    
                else:
                    synoverlap = matches['overlap'].iloc[n]
                    userxmin = matches['userxmin'].iloc[n]
                    userxmax = matches['userxmax'].iloc[n]
                    
                    # Add subclassid index of match to done list
                    done.append(int(matches['idx'].iloc[n]))
                
                # Write line into the output file
                line = []
                for value in [username,classid,subjectid,signalid,synoverlap,synthid,userxmin,userxmax]:
                    line.append(value)
                for l in range(len(line)):
                    if l < len(line)-1:
                        matchfilewrite.write(str(line[l])+',')
                    else:
                        matchfilewrite.write(str(line[l])+'\n')
                
                # Proceed to next k index (next synthetic transit in light curve)
                
        # Once done with all synthetic transits, move to next classification id for that light curve
        # Done list will be dumped and reinitiated

    if i%10==0:
        print('{0:0.2f}'.format(100*float(i)/float(len(setsubjectids)))+'% completed.')

print('Completed matching synthetics to user markings.')
matchfilewrite.close()





""" PART 2:  USER WEIGHTING """


# Setting up the maximum amount a single correct classification can increase your score
# Max increase in score = 1.0/completenesscutoff
completenesscutoff = 0.1


""" Define decay function for relative upweights of consecutive transits in single lightcurves """

# Decay function with exponential tail
def decay(k):
    if k <= 3:
        return 1.03-0.03*np.exp(1.07*k)
    if k > 3:
        return ( 1.03-0.03*np.exp(1.07*3) )*np.exp(-1.5*(k-3))


''' Define the file name to which we'll write the userweight table: '''
weightfilename = user_directory+'user-weighting.csv'

''' Read in matching data, to check whether users correctly marked synthetics, and classifications database '''
matching = pd.read_csv(user_directory+'match-user-synthetics.csv')

# For later:
dbnonsynth = db[db['synthetic']==False]     # subset of non-synthetic classifications
dbsynth = db[db['synthetic']==True]     # subset of synthetic classifications



''' Set up the table '''
# Setting up the user list and initializing the user weights
allusers = list(set(matching['username']))
userweights = Table({'username':allusers,'upweight':np.ones(len(allusers)),'downweight':np.ones(len(allusers)),'combined':np.ones(len(allusers)),'normupweight':np.zeros(len(allusers)),'normdownweight':np.zeros(len(allusers)),'normcombined':np.zeros(len(allusers)),'numclasses':np.zeros(len(allusers))}, names=['username','upweight','downweight','combined','normupweight','normdownweight','normcombined','numclasses'])


print("Beginning upweighting...")

''' Starting the upweighting portion '''

featureidlist = []
sidclist = []
for i in range(len(allusers)):
    # Find all classifications of specific user
    userclasses = matching[matching['username']==allusers[i]]
    setuserclassids = list(set(matching['classid'][matching['username']==allusers[i]]))
    # For each classification done by user, get list of transit ids
    for j in range(len(setuserclassids)):
        signalids = list(set(userclasses['signalid'][userclasses['classid']==setuserclassids[j]]))
        userclassesclassid = userclasses[userclasses['classid']==setuserclassids[j]]
        weightlist = []     # For storing upweight values prior to applying decay function
        # Go through all classifications of each signalid and count the total number detected
        for k in range(len(signalids)):
            # If user correctly classified a synthetic, then we increase their weight
            if userclassesclassid['synoverlap'].iloc[k]>0.5: 
                signalidclasses = matching[matching['signalid']==signalids[k]]
                numidentified = signalidclasses[signalidclasses['synoverlap']>0.5]
                # Find increase in user weight, maximum increase = (1.0/completenesscutoff)-1
                signalidcompleteness = float(len(numidentified))/float(len(signalidclasses))
                signalidcompleteness = max([signalidcompleteness,completenesscutoff])
                weightlist.append(1.0/signalidcompleteness-1.0)
                featureidlist.append(signalids[k])
                sidclist.append(signalidcompleteness)
            userweights[i]['numclasses'] += 1
        weightlist.sort()
        # Apply decay function
        for b in range(len(weightlist)):
            weightlist[b] = weightlist[b]*decay(b)
            userweights[i]['upweight'] += weightlist[b]
            
    if i%10==0:
        print('{0:0.2f}'.format(100*float(i)/float(len(allusers)))+'% completed.')

print("Completed upweighting.")



# Save SIDC info for later analysis
sidcs = pd.DataFrame(np.column_stack([featureidlist,sidclist]), columns=['signalid', 'sidc'])

featureset = list(set(featureidlist))
# Rewrite old lists, don't need new variables
featureidlist = []
sidclist = []
for i in range(len(featureset)):
    featureidlist.append(featureset[i])
    sidclist.append(sidcs['sidc'][sidcs['signalid']==featureset[i]].iloc[0])   # May only be one entry
    
# Again, new dataframe, old variable
sidcs = pd.DataFrame(np.column_stack([featureidlist,sidclist]), columns=['signalid', 'sidc'])
sidcs.to_csv(user_directory+'sidcs.csv', index=False)




''' Starting the downweighting portion '''

# Extracting necessary information on all classifications of synthetics
# These are REDEFINED from the regular db definitions in PART 1
# Ah, the woes of merging two once separate codes into one
classid,usernames,syntheticid,xminglobal,xmaxglobal = np.asarray(dbsynth['classification_id']), np.asarray(dbsynth['user_name']), np.asarray(dbsynth['synthetic_id']), np.asarray(dbsynth['xMinGlobal']), np.asarray(dbsynth['xMaxGlobal'])

# Getting a list of unique classids
setclassid = list(set(classid))


print("Beginning downeighting...")

for i in range(len(setclassid)):
    # Getting list of markings for each unique classid
    classxmin = xminglobal[np.where(classid==setclassid[i])]
    classusername = usernames[np.where(classid==setclassid[i])][0]
    countwrong = 0
    # If user didn't marking anything, don't do anything.  User is not penalized.
    if len(classxmin) == 1 and np.isnan(classxmin[0])==True:
        pass
    # If user did mark something:
    else:
        classxmax = xmaxglobal[np.where(classid==setclassid[i])]
        syntheticmarkings = matching[matching['classid']==setclassid[i]]
        # For each userxmin, check for a corresponding correctly marked synthetic
        # If it exists, break out of for loop
        # If it doesn't exist, increase countwrong and user's downweight
        for j in range(len(classxmin)):
            isitcorrect = False
            for k in range(len(syntheticmarkings)):
                if np.abs(syntheticmarkings['userxmin'].iloc[k]-float(classxmin[j])) < 0.000001 and np.abs(syntheticmarkings['userxmax'].iloc[k]-float(classxmax[j])) < 0.000001 and syntheticmarkings['synoverlap'].iloc[k] > 0.5:
                    isitcorrect = True
                    break   # Breaks out of k-loop, continues through j-loop
            if isitcorrect == False and classusername in userweights['username']:
                countwrong += 1
                userweights[np.where(userweights['username']==classusername)[0][0]]['downweight'] += 1

    if i%10==0:
        print('{0:0.2f}'.format(100*float(i)/float(len(setclassid)))+'% completed.')

print("Downweighting complete.")


# Finding the highest weight possible if users marked every single transit and no false positives
syntheticids = list(set(matching['syntheticid']))
highestweight = 1
for i in range(len(syntheticids)):
    signalids = list(set(matching['signalid'][matching['syntheticid']==syntheticids[i]]))
    weightlist = []
    for j in range(len(signalids)):
        synthetic = matching.iloc[np.where(matching['signalid']==signalids[j])]
        numidentified = synthetic.iloc[np.where(synthetic['synoverlap']>0.5)]
        signalidcompleteness = float(len(numidentified))/float(len(synthetic))
        signalidcompleteness = max([signalidcompleteness,completenesscutoff])
        weightlist.append(1.0/signalidcompleteness-1.0)
    weightlist.sort()
    for k in range(len(weightlist)):
        weightlist[k] = weightlist[k]*decay(k)
        highestweight += weightlist[k]

print('Highest weight possible = '+str(highestweight)+'.')


# Normalizing upweights and downweights to 1
userweights['normupweight'] = userweights['upweight']/np.average(userweights['upweight'])
userweights['normdownweight'] = userweights['downweight']/np.average(userweights['downweight'])
# Combining the upweights and downweights
userweights['combined'] = userweights['normupweight']/userweights['normdownweight']
userweights['normcombined'] = userweights['combined']/np.average(userweights['combined'])

print("Normalization complete.")


''' Writing output '''

weightfilewrite = open(weightfilename,'w')
for i in range(len(userweights)):
    weightfilewrite.write(userweights[i][0]+','+str(userweights[i][1])+','+str(userweights[i][2])+','+str(userweights[i][3])+','+str(userweights[i][4])+','+str(userweights[i][5])+','+str(userweights[i][6])+','+str(userweights[i][7])+'\n')

weightfilewrite.close()




elapsed = timeit.default_timer() - start_time
print('Run time = '+str(elapsed)+' sec.')



""" End program. """
