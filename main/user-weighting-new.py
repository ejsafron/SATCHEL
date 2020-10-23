#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python ver. 3.6.5
Created on Tue Oct 20 18:16:16 2020

@author: Emily Safron

For the purpose of filtering out signals from known short-period objects and still being left with those from potentially unknown long-period objects, we must score by the transit feature, rather than by the light curve.  Though this is computationally intensive, it will give us a very straightforward way to filter our results, in the end.

To score these features, we first need an iterable list of them.  For this, we go back to the user marks themselves.  We go through all the global xmin and xmax ranges, mark by mark, and quantify by measure of overlap fraction and closeness of marking midpoint, which chunks of x-range indicate markings which are "alike" enough to be consolidated together.  The consolidated markings, which we call metafeatures, are written and saved to a csv.

This code takes as input:
    mdwarf-classifs.csv

and produces as output:
    metafeatures.csv.

"""


import numpy as np
import pandas as pd
import timeit
from astropy.io import ascii
from astropy.table import Table


# For easy replacement
user_directory = '/home/safron/Documents/PH/master/new/'    # sif
#user_directory = '/home/esafron/Documents/PH/master/test/'    # masotan


''' Set adjustable pipeline parameters '''
tolerance = 1.0      # How close (in days) the midpoints of two user markings must be if indicating the same feature
cutoff = 2.5         # Duration (in days) of the longest user marking we're willing to assume was made intentionally





start_time = timeit.default_timer()



''' Define functions '''
def funcoverlap(synx, userx):
	'''Returns the total length of the overlapping areas'''
    # EMILY NOTE:  synx is a list of the form [beginning of transit, end of transit] and userx is a list of the form [beginning of user marking, end of user marking].
	return max(0, min(synx[1], userx[1]) - max(synx[0], userx[0]))



''' Read in data '''
# All M dwarf classifications
db = pd.read_csv(user_directory+'mdwarf-classifs.csv')
db = db[db['synthetic']==True]
db = db[db['synthetic_id']>1412]

# Get set of all simulation lightcurve ids from db
setsubjectids = list(set(db['subject_id'][db['synthetic']==True]))



''' Read in the synthetic file '''
# Cut out entries with syntheticid < 1412, because those synthetic transits are made by "small planets in hard-to-see stars." 
synthetics = ascii.read(user_directory+'allsynthetics.dat', header_start=0, data_start=1)
synthetics = synthetics[np.where(synthetics['syntheticid']>1412)]
synthetics = synthetics.to_pandas()

print('Completed setting up list of data arrays.')



''' Set up the output file '''

matchfilename = user_directory+'match-user-synthetics.csv'
matchfilewrite = open(matchfilename,'w')

# Write header containing column names into line 0
matchfilewrite.write('kepid,fits,i,j,k,l,period,prad,srad,kepmag,activity,transitid,syntheticid,plphase,synpixmin,synpixmax,synxmin,synxmax,synmidpoint,synduration,userxmin,userxmax,usermidpoint,userduration,synoverlap,useroverlap,midpointdiff,username,quarter,classid,createdat,datalocation,starttime,subjectid,syntheticbool,keplertype,xminrelative,xmaxrelative\n')




for i in range(len(setsubjectids)):
    subclasses = db[db['subject_id']==setsubjectids[i]]                         # All classifications of lightcurve[i]
#    subclasses = db[db['subject_id']=='54171a408841e106190012f4']
    setsubclassids = list(set(subclasses['classification_id']))                 # List of relevant classification ids
    synthid = subclasses['synthetic_id'].iloc[0]                                # Synthetic ID
    curvesynths = synthetics[synthetics['syntheticid']==synthid]                # Info on all synthetic transits in light curve
        
    for j in range(len(setsubclassids)):
        done = []       # Record indices of marks that have already been consolidated
        
        subclassid = subclasses[subclasses['classification_id']==setsubclassids[j]]     # Only marks from one classification
        
        # Extract user & classification information
        username = subclassid['user_name'].iloc[0]
        quarter = subclassid['quarter'].iloc[0]
        classid = subclassid['classification_id'].iloc[0]
        createdat = subclassid['created_at'].iloc[0]
        datalocation = subclassid['data_location'].iloc[0]
        starttime = subclassid['start_time'].iloc[0]
        subjectid = subclassid['subject_id'].iloc[0]
        syntheticbool = subclassid['synthetic'].iloc[0]
        keplertype = subclassid['kepler_type'].iloc[0]
        
        if np.isnan(subclassid['xMinGlobal'].iloc[0])==True:                    # If user made no marks on light curve
            # Assign np.nans to everything else
            synoverlap,useroverlap,midpointdiff,userxmin,userxmax,userduration,usermidpoint,xminrelative,xmaxrelative = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
            
            for k in range(len(curvesynths)):
                # Extract information about each synthetic
                synxmin,synxmax = curvesynths['synxmin'].iloc[k],curvesynths['synxmax'].iloc[k]
                synduration = synxmax-synxmin
                synmidpoint = np.average([synxmin,synxmax])
                
                # Write an "empty" line into the output file
                line = []
                for value in synthetics.iloc[k]:
                    line.append(value)
                for value in [synmidpoint,synduration,userxmin,userxmax,usermidpoint,userduration,synoverlap,useroverlap,midpointdiff,username,quarter,classid,createdat,datalocation,starttime,subjectid,syntheticbool,keplertype,xminrelative,xmaxrelative]:
                    line.append(value)
                for l in range(len(line)):
                    if l < len(line)-1:
                        matchfilewrite.write(str(line[l])+',')
                    else:
                        matchfilewrite.write(str(line[l])+'\n')
        
        else:                                                                   # If the user made at least one mark on the light curve
            for k in range(len(curvesynths)):
                # Extract information about each synthetic
                synxmin,synxmax = curvesynths['synxmin'].iloc[k],curvesynths['synxmax'].iloc[k]
                synduration = synxmax-synxmin
                synmidpoint = np.average([synxmin,synxmax])
                
                idx,alloverlap,alluseroverlap,allmidpointdiff,alluserxmin,alluserxmax,alluserduration,allusermidpoint,alluserxminrel,alluserxmaxrel = [],[],[],[],[],[],[],[],[],[]

                for m in range(len(subclassid)):
                    tmpuserxmin,tmpuserxmax = float(subclassid['xMinGlobal'].iloc[m]),float(subclassid['xMaxGlobal'].iloc[m])
                    tmpuserduration = tmpuserxmax-tmpuserxmin
                    tmpusermidpoint = np.average([tmpuserxmin,tmpuserxmax])
                    tmpuserxminrel = float(subclassid['xMinRelative'].iloc[m])
                    tmpuserxmaxrel = float(subclassid['xMaxRelative'].iloc[m])
                    idx.append(m)
                    alluserxmin.append(tmpuserxmin)
                    alluserxmax.append(tmpuserxmax)
                    alluserxminrel.append(tmpuserxminrel)
                    alluserxmaxrel.append(tmpuserxmaxrel)
                    alluserduration.append(tmpuserxmax-tmpuserxmin)
                    alloverlap.append(funcoverlap([tmpuserxmin,tmpuserxmax],[synxmin,synxmax])/synduration)
                    alluseroverlap.append(funcoverlap([tmpuserxmin,tmpuserxmax],[synxmin,synxmax])/tmpuserduration)
                    allusermidpoint.append(tmpusermidpoint)
                    allmidpointdiff.append(np.abs(synmidpoint-tmpusermidpoint))
            
                # Still inside j loop, for specific classification id
                # And k loop, for specific row of synthetic id
                
                usermarksdf = pd.DataFrame(np.column_stack([idx,alluserxmin,alluserxmax,alluserxminrel,alluserxmaxrel,alluserduration,alloverlap,alluseroverlap,allusermidpoint,allmidpointdiff]),columns=['idx','userxmin','userxmax','userxminrel','userxmaxrel','userduration','overlap','useroverlap','usermidpoint','midpointdiff'])
                
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
                    synoverlap,useroverlap,midpointdiff,userxmin,userxmax,userduration,usermidpoint,xminrelative,xmaxrelative = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
                    
                else:
                    synoverlap = matches['overlap'].iloc[n]
                    useroverlap = matches['useroverlap'].iloc[n]
                    midpointdiff = matches['midpointdiff'].iloc[n]
                    userxmin = matches['userxmin'].iloc[n]
                    userxmax = matches['userxmax'].iloc[n]
                    userduration = matches['userduration'].iloc[n]
                    usermidpoint = matches['usermidpoint'].iloc[n]
                    xminrelative = matches['userxminrel'].iloc[n]
                    xmaxrelative = matches['userxmaxrel'].iloc[n]
                    
                    # Add subclassid index of match to done list
                    done.append(int(matches['idx'].iloc[n]))
                
                # Write line into the output file
                line = []
                for value in synthetics.iloc[k]:
                    line.append(value)
                for value in [synmidpoint,synduration,userxmin,userxmax,usermidpoint,userduration,synoverlap,useroverlap,midpointdiff,username,quarter,classid,createdat,datalocation,starttime,subjectid,syntheticbool,keplertype,xminrelative,xmaxrelative]:
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

# Exclude the 761 classifcations of synthetics with synthetic_id < 1412, as in matching code
dbsynth = dbsynth[dbsynth['synthetic_id']>1412]


''' Set up the table '''
# Setting up the user list and initializing the user weights
allusers = list(set(matching['username']))
userweights = Table({'username':allusers,'upweight':np.ones(len(allusers)),'downweight':np.ones(len(allusers)),'combined':np.ones(len(allusers)),'normupweight':np.zeros(len(allusers)),'normdownweight':np.zeros(len(allusers)),'normcombined':np.zeros(len(allusers)),'numclasses':np.zeros(len(allusers))}, names=['username','upweight','downweight','combined','normupweight','normdownweight','normcombined','numclasses'])


''' Starting the upweighting portion '''

for i in range(len(allusers)):
    # Find all classifications of specific user
    userclasses = matching[matching['username']==allusers[i]]
    setuserclassids = list(set(matching['classid'][matching['username']==allusers[i]]))
    # For each classification done by user, get list of transit ids
    for j in range(len(setuserclassids)):
        transitids = list(set(userclasses['transitid'][userclasses['classid']==setuserclassids[j]]))
        userclassesclassid = userclasses[userclasses['classid']==setuserclassids[j]]
        weightlist = []     # For storing upweight values prior to applying decay function
        # Go through all classifications of each transitid and count the total number detected
        for k in range(len(transitids)):
            # If user correctly classified a synthetic, then we increase their weight
            if userclassesclassid['synoverlap'].iloc[k]>0.5: 
                transitidclasses = matching[matching['transitid']==transitids[k]]
                numidentified = transitidclasses[transitidclasses['synoverlap']>0.5]
                # Find increase in user weight, maximum increase = (1.0/completenesscutoff)-1
                transitidcompleteness = float(len(numidentified))/float(len(transitidclasses))
                transitidcompleteness = max([transitidcompleteness,completenesscutoff])
                weightlist.append(1.0/transitidcompleteness-1.0)
            userweights[i]['numclasses'] += 1
        weightlist.sort()
        # Apply decay function
        for b in range(len(weightlist)):
            weightlist[b] = weightlist[b]*decay(b)
            userweights[i]['upweight'] += weightlist[b]
            
    print('{0:0.2f}'.format(100.0*float(i+1)/float(len(allusers)))+'% completed.  '+allusers[i]+' weight = '+str(userweights[i]['upweight'])+'. Numclasses = '+str(int(userweights[i]['numclasses']))+'. i = '+str(i)+'.')


''' Starting the downweighting portion '''

# Extracting necessary information on all classifications of synthetics
# These are REDEFINED from the regular db definitions in PART 1
# Ah, the woes of merging two once separate codes into one
classid,usernames,syntheticid,xminglobal,xmaxglobal = np.asarray(dbsynth['classification_id']), np.asarray(dbsynth['user_name']), np.asarray(dbsynth['synthetic_id']), np.asarray(dbsynth['xMinGlobal']), np.asarray(dbsynth['xMaxGlobal'])

# Getting a list of unique classids
setclassid = list(set(classid))


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
    print('{0:0.2f}'.format(100.0*float(i+1)/float(len(setclassid)))+'% completed.  '+classusername+' countwrong = '+str(countwrong)+'. i = '+str(i)+'.')



# Finding the highest weight possible if users marked every single transit and no false positives
syntheticids = list(set(matching['syntheticid']))
highestweight = 1
for i in range(len(syntheticids)):
    transitids = list(set(matching['transitid'][matching['syntheticid']==syntheticids[i]]))
    weightlist = []
    for j in range(len(transitids)):
        synthetic = matching.iloc[np.where(matching['transitid']==transitids[j])]
        numidentified = synthetic.iloc[np.where(synthetic['synoverlap']>0.5)]
        transitidcompleteness = float(len(numidentified))/float(len(synthetic))
        transitidcompleteness = max([transitidcompleteness,completenesscutoff])
        weightlist.append(1.0/transitidcompleteness-1.0)
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


# Writing output
weightfilewrite = open(weightfilename,'w')
for i in range(len(userweights)):
    weightfilewrite.write(userweights[i][0]+','+str(userweights[i][1])+','+str(userweights[i][2])+','+str(userweights[i][3])+','+str(userweights[i][4])+','+str(userweights[i][5])+','+str(userweights[i][6])+','+str(userweights[i][7])+'\n')

weightfilewrite.close()




elapsed = timeit.default_timer() - start_time
print('Run time = '+str(elapsed)+' sec.')



""" End program. """