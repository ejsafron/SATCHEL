#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python ver. 3.6.0
Created on Tue Aug 14 12:59:12 2018
@author: Emily Safron

This code loads in the Planet Hunters classification database from February 25th, 2018 and pares the rows down to just those classifications with Kepler IDs matching those of the M-dwarf list published in Gaidos et al. 2016, specified in the file mdwarfs.dat.  This routine also expands some info from the location strings in the PH classification data for easy parsing in later parts of the pipeline.

This code takes as input:
    2018-02-25_planet_hunter_classifications.csv
    mdwarfs.dat

and produces as output:
    mdwarf-classifs.csv.

"""

import pandas as pd
import time

# For easy replacement
user_directory = '/home/safron/Documents/PH/master/'    # sif

start_time = time.time()


''' Load databases '''
mdwarfs = pd.read_csv(user_directory+'mdwarfs.dat', sep='\s+', header=None)
classifs = pd.read_csv(user_directory+'2018-02-25_planet_hunter_classifications.csv')


print('Databases loaded.')


''' Initialize new csv '''
filename = user_directory+'mdwarf-classifs.csv'

filewrite = open(filename,'w')
filewrite.write('classification_id,created_at,data_location,quarter,start_time,subject_id,user_name,kepler_id,synthetic,synthetic_id,kepler_type,xMinRelative,xMaxRelative,xMinGlobal,xMaxGlobal\n')
filewrite.close()


# Count rows of classification database
numrows = len(classifs['classification_id'])
kep_ids = list(mdwarfs[0])

# Main loop
for i in range(numrows):
    row = classifs.loc[i,:]
    location = str(row['data_location'])
    if 'C' in location:
        pass
    elif '-k0' in location:
        pass
    elif 'kdwarf' in location:
        pass
    elif 'sim' in location:
        pass
    else:
        kep_id = int(location.split('_')[0].split('/')[-1])
        if kep_id in kep_ids:
            locsplit = location.split('_')
            if 'synth' in location:  # For synthetics
                synthsplit = locsplit[1].split('-')
                synthetic = 'true'
                synthetic_id = synthsplit[-1].split('.')[0]
                kepler_id = locsplit[0].split('/')[-1]
                kepler_type = 'k1_synthetic'
            else:  #For K1 non-synthetics
                synthetic = 'false'
                synthetic_id = ''
                kepler_id = locsplit[0].split('/')[-1]
                kepler_type = 'k1'
        
            classid,createdat,datalocation,quarters,starttime,subjectid,username,xminrelative,xmaxrelative,xminglobal,xmaxglobal = row['classification_id'], row['created_at'], row['data_location'], row['quarter'], row['start_time'], row['subject_id'], row['user_name'], row['xMinRelative'], row['xMaxRelative'], row['xMinGlobal'], row['xMaxGlobal']
            
            if ',' in username:
                namesplit = username.split(',')
                username = '(comma)'.join(namesplit)
            else:
                pass
            
            printlist = f"{classid}, {createdat},{datalocation},{quarters},{starttime},{subjectid},{username},{kepler_id},{synthetic},{synthetic_id},{kepler_type},{xminrelative},{xmaxrelative},{xminglobal},{xmaxglobal}\n"
            with open(filename, 'a') as filewrite:
                filewrite.write(printlist)
        else:
            pass

    
    if i%1000==0:
        print('{0:0.3f}'.format(100*float(i)/float(numrows))+'% completed.')


print('File written successfully.')


run_time = (time.time() - start_time)/60.0

print('Run time:  %0.2f minutes' %run_time)



''' End program '''