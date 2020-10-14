# -*- coding: utf-8 -*-
"""
Python ver. 3.6.0

Created on Fri Jun 22 10:16:38 2018

@author: Emily Safron

This is an edited version of matchusermarkssynthetics.py written by Joseph Schmitt for the purpose of finding overlaps between user-made marks on Planet Hunters lightcurves and synthetic transits inserted into Kepler lightcurves.

For each synthetic transit (some of which occur in the same light curve), some number of classifications were done by users.  For each transit, and each classification therein, any user marking overlapping the transit is recorded or, if the user made no overlapping marking, 'nan's are recorded.

The purpose of the program is to calculate user weights based on how correctly PH users classify lightcurves containing synthetic transits.  Users are "upweighted" for correct markings, and "downweighted" for incorrect markings.  Downweights are not given for failure to mark transits.  The exact amount of each upweight and downweght is impacted by a "transit ID completeness," which is a measure of how difficult a transit is to find based on the percentage of users that find it, and by a decay function that decreases the relative upweight for multiple transits found in a single lightcurve.

This code takes as input:
    allsynthetics.dat
    mdwarf-classifs.csv

and produces as output:
    match-user-synthetics.csv
    user-weighting.csv.

"""

# Goes through each transitid and finds all classids of that syntheticid
# For each unique classid:
	# If no user markings overlap transitid, outputs np.nan (failure to mark transitid) 
	# If user marking(s) does overlap transitid, outputs closest user markings to transitid as measured by midpoint difference (successful marking of transitid)


import pandas as pd
import numpy as np
from astropy.io import ascii
from astropy.table import Table
import timeit

# For easy replacement
user_directory = '/home/safron/Documents/PH/master/'    # sif


start_time = timeit.default_timer()


""" PART 1:  MATCH USER SYNTHETICS """

''' Define functions '''
def funcoverlap(synx, userx):
	'''Returns the total length of the overlapping areas'''
    # EMILY NOTE:  synx is a list of the form [beginning of transit, end of transit] and userx is a list of the form [beginning of user marking, end of user marking].
	return max(0, min(synx[1], userx[1]) - max(synx[0], userx[0]))


''' Read in the synthetic file '''
# Cut out entries with syntheticid < 1412, because those synthetic transits are made by "small planets in hard-to-see stars." 
synthetics = ascii.read(user_directory+'allsynthetics.dat', header_start=0, data_start=1)
synthetics = synthetics[np.where(synthetics['syntheticid']>1412)]
#synthetics = synthetics.to_pandas()


''' Create output data file '''

matchfilename = user_directory+'match-user-synthetics.csv'
matchfilewrite = open(matchfilename,'w')

# Write header containing column names into line 0
matchfilewrite.write('kepid,fits,i,j,k,l,period,prad,srad,kepmag,activity,transitid,syntheticid,plphase,synpixmin,synpixmax,synxmin,synxmax,synmidpoint,synduration,userxmin,userxmax,usermidpoint,userduration,synoverlap,useroverlap,midpointdiff,username,quarter,classid,createdat,datalocation,starttime,subjectid,syntheticbool,keplertype,xminrelative,xmaxrelative\n')


''' Load database '''
db = pd.read_csv(user_directory+'mdwarf-classifs.csv')

# Cut db down to only k1 and k1_synthetic rows
db = db[(db['kepler_type']=='k1_synthetic') | (db['kepler_type']=='k1')]

# Create individual arrays from each column
classid,createdat,datalocation,quarters,starttime,subjectid,usernames,kic,syntheticbool,syntheticid,keplertype,xminrelative,xmaxrelative,xminglobal,xmaxglobal = np.asarray(db['classification_id']), np.asarray(db['created_at']), np.asarray(db['data_location']), np.asarray(db['quarter']), np.asarray(db['start_time']), np.asarray(db['subject_id']), np.asarray(db['user_name']), np.asarray(db['kepler_id']), np.asarray(db['synthetic']), np.asarray(db['synthetic_id']), np.asarray(db['kepler_type']), np.asarray(db['xMinRelative']), np.asarray(db['xMaxRelative']), np.asarray(db['xMinGlobal']), np.asarray(db['xMaxGlobal'])

# Make list of arrays from classification data
data = [usernames,classid,quarters,kic,syntheticid,xminglobal,xmaxglobal,createdat,datalocation,starttime,subjectid,syntheticbool,keplertype,xminrelative,xmaxrelative]

print('Completed setting up list of data arrays.')


############################### data index to variable ###############################
#             0             1             2           3              4              5
#     usernames       classid      quarters         kic    syntheticid     xminglobal
#             6             7             8           9             10             11
#    xmaxglobal     createdat  datalocation   starttime      subjectid  syntheticbool
#            12            13            14
#    keplertype  xminrelative  xmaxrelative


for i in range(len(synthetics)):
    # Extracting the information about the synthetic
    synxmin,synxmax = synthetics['synxmin'][i],synthetics['synxmax'][i]
    synduration = synxmax-synxmin
    synmidpoint = np.average([synxmin,synxmax])
    
    onedata = data[:]  # Copying to an independent list
    
    # Keeping only the classifications of the current synthetic_id
    # 4 is the last index to prevent it from screwing up the np.where
    for j in [0,1,2,3,5,6,7,8,9,10,11,12,13,14,4]:
        onedata[j] = onedata[j][np.where(onedata[4]==synthetics['syntheticid'][i])]
    
    setclassid = list(set(onedata[1]))  # Creating a unique list of classid's
    for j in range(len(setclassid)):  # Iterating through the unique list of classid's
        classdata = onedata[:]  # Copying to an independent list
        # Keeping only the classifications of the current classid
        # 1 is the last index to prevent it from screwing up the np.where
        for k in [0,2,3,4,5,6,7,8,9,10,11,12,13,14,1]:
            classdata[k] = classdata[k][np.where(classdata[1]==setclassid[j])]
        numclassifications = len(classdata[1]) # Number of user markings in current classid
        # Extracting all unchanging information
        username = classdata[0][0]
        quarter = ' '+classdata[2][0]  # Space added in front so Excel doesn't read in a date
        classid = classdata[1][0]
        createdat = classdata[7][0]
        datalocation = classdata[8][0]
        starttime = classdata[9][0]
        subjectid = classdata[10][0]
        syntheticbool = classdata[11][0]
        keplertype = classdata[12][0]
        # Setting up overlap and midpoint lists
        alloverlap,alluseroverlap,allmidpointdiff,alluserxmin,alluserxmax,alluserduration,allusermidpoint,alluserxminrel,alluserxmaxrel = [],[],[],[],[],[],[],[],[]
        # Iterating over number of user markings made on current classid
        for k in range(numclassifications):  
            if classdata[5][k]>0:  # True if user made a marking
                # Computing user markings, overlap, and midpointdiff
                tmpuserxmin,tmpuserxmax = float(classdata[5][k]),float(classdata[6][k])
                tmpuserduration = tmpuserxmax-tmpuserxmin
                alluserxmin.append(tmpuserxmin)
                alluserxmax.append(tmpuserxmax)
                tmpuserxminrel = float(classdata[13][k])
                tmpuserxmaxrel = float(classdata[14][k])
                alluserxminrel.append(tmpuserxminrel)
                alluserxmaxrel.append(tmpuserxmaxrel)
                alluserduration.append(tmpuserxmax-tmpuserxmin)
                alloverlap.append(funcoverlap([tmpuserxmin,tmpuserxmax],[synxmin,synxmax])/synduration)
                alluseroverlap.append(funcoverlap([tmpuserxmin,tmpuserxmax],[synxmin,synxmax])/tmpuserduration)
                tmpusermidpoint = np.average([tmpuserxmin,tmpuserxmax])
                allusermidpoint.append(tmpusermidpoint)
                allmidpointdiff.append(np.abs(synmidpoint-tmpusermidpoint))
            else:  # In case user made no marking
                pass
        
        # Setting up arrays to use for np.where
        # Still for j in range(len(setclassid)):
        alloverlaparray,allmidpointdiffarray = np.asarray(alloverlap),np.asarray(allmidpointdiff)
        allmidpointwithoverlap = allmidpointdiffarray[np.where(alloverlaparray>0)]
        if len(allmidpointwithoverlap)>0:  # True if there is a marking with overlap
            # Finding index of the closest user marked midpoint with overlap
            usermarkindex = allmidpointdiff.index(np.min(allmidpointwithoverlap))
            synoverlap = alloverlap[usermarkindex]
            useroverlap = alluseroverlap[usermarkindex]
            midpointdiff = allmidpointdiff[usermarkindex]
            userxmin = alluserxmin[usermarkindex]
            userxmax = alluserxmax[usermarkindex]
            userduration = alluserduration[usermarkindex]
            usermidpoint = allusermidpoint[usermarkindex]
            xminrelative = alluserxminrel[usermarkindex]
            xmaxrelative = alluserxmaxrel[usermarkindex]
        else:  # In case of no marking or no overlap
            synoverlap,useroverlap,midpointdiff,userxmin,userxmax,userduration,usermidpoint,usermarkindex,xminrelative,xmaxrelative = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        
        line = []
        for value in synthetics[i]:
            line.append(value)
        for value in [synmidpoint,synduration,userxmin,userxmax,usermidpoint,userduration,synoverlap,useroverlap,midpointdiff,username,quarter,classid,createdat,datalocation,starttime,subjectid,syntheticbool,keplertype,xminrelative,xmaxrelative]:
            line.append(value)
        for k in range(len(line)):
            if k < len(line)-1:
                matchfilewrite.write(str(line[k])+',')
            else:
                matchfilewrite.write(str(line[k])+'\n')
    if i%10==0:
        print('{0:0.2f}'.format(100*float(i)/float(len(synthetics)))+'% completed.')

print('Completed matching synthetics to user markings.')
matchfilewrite.close()



""" PART 2:  USER WEIGHTING """


# Setting up the maximum amount a single correct classification can increase your score
# Max increase in score = 1.0/completenesscutoff
completenesscutoff = 0.1


""" Define decay function for relative upweights of consecutive transits in single lightcurves """

# Decay function with cutoff
#def decay(k):
#    a = np.log(1.03/0.03)/1.07
#    if k <= a:
#        return 1.03-0.03*np.exp(1.07*k)
#    if k > a:
#        return 0


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