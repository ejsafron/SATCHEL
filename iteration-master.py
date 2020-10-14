#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python ver. 3.6.5
Created on Mon May 11 13:36:44 2020

@author: Emily Safron

EDIT, 20 March 2020:  Changing the convergence condition so that the loop
breaks when the MEAN change in zero (within a tolerance of 1e-3).  This is to
avoid resonant increase/decrease cycles across iterations that prevent the loop
from converging.  Over all, convergence should happen quicker.  Additionally,
I will also add a couple of extra checks, and attempt to implement a way to
preserve the state of the transit scores and user weights in the case that the
loop is aborted prior to convergence.

This code defines a while loop that uses transit scores to recalculate and
adjust user weights, renormalize the user weight distribution, and then check
the stability of the normalization based on whether or not any user weights
have changed (within a tolerance of 1e-3) from the beginning of the loop to the
end.  If no user weights have changed, the loop breaks, and the final transit
scores and user weights are saved.

To run this code, we will need:
    
    Input:   metafeatures.csv
             transitscoreseeds.csv
             full_uwseeds.csv
            
    Output:  final_userweights.tsv
             final_transitscores.tsv
             outputlog.csv
             weight-change-rate.pdf *
             weight-distrib-change.pdf *
             weight-diff-distrib-change.pdf *
             score-distrib-change.pdf *
             meanmagdiffs.pdf *
             weight-distrib-change-<N>.png **
             weight-diff-distrib-change-<N>.png **
             score-distrib-change-<N>.png **
             scores.gif *
             weights.gif *
             weight-diffs.gif *
             
* (single plot, updated every iteration)
** (individual png and gif saved for every iteration <N>)
             

"""

''' Import packages '''
import numpy as np
import pandas as pd
import ast
import multiprocessing as mp
import timeit
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import itertools
import os
import csv
import math
from PIL import Image



# For easy replacement
user_directory = '/home/safron/Documents/PH/master/'    # sif
#user_directory = '/home/esafron/Documents/PH/master/'    # masotan

# Create directory for plots
os.mkdir(user_directory+'plots')

# Create subdirectories for non-diagnostic plots
os.mkdir(user_directory+'plots/weight-distrib-changes')
os.mkdir(user_directory+'plots/weight-diff-distrib-changes')
os.mkdir(user_directory+'plots/score-distrib-changes')
os.mkdir(user_directory+'plots/weight-distrib-changes/anim')
os.mkdir(user_directory+'plots/weight-diff-distrib-changes/anim')
os.mkdir(user_directory+'plots/score-distrib-changes/anim')


''' Initiate output log file '''

file = open(user_directory+'outputlog.csv', 'w', newline='')
filewrite = csv.writer(file, delimiter=',')
header = ['iteration_count','changing_count','largeflux_count','mean_magweightdiff','mean_weight','mean_score','time_elapsed']  # Header
filewrite.writerow(header)






''' Import data '''

# The metafeatures
allfeatures = pd.read_csv(user_directory+'metafeatures.csv')
#  ||  transitids  |  lightcurves |  kicids  |  xmin  |  xmax  |  durations  |  midpoints  |  synth_flag  |  dupe_flag  ||

# TEST
print("# of metafeatures:  "+str(len(allfeatures)))

# The transit score seeds
score_seeds = pd.read_csv(user_directory+'transitscoreseeds.tsv', sep="\t", names=['transitid', 'lightcurve', 'kicid', 'xmin', 'xmax', 'duration', 'usersyes', 'usersno', 'weightsyes', 'weightsno', 'scoreyes', 'scoreno', 'numclasses'], skiprows=1)
#  ||  transitid  |  lightcurve  |  kicid  |  userxmin  |  userxmax  |  duration  |  usersyes  |  usersno  |  weightsyes  |  weightsno  |  scoreyes  |  scoreno  |  numclasses  ||


# The userweight seeds
userweights = pd.read_csv(user_directory+'full_uwseeds.tsv', sep="\t")
#  ||  username  |  username_check  |  normcombined  |  numLCclasses  |  numtransitclasses  |  transitsclassified  ||



# These things can be defined outside the loop
# The only thing that will update with each iteration is the past_weightlist
featurelist = list(score_seeds['transitid'])    # List of all transit IDs
userlist = list(userweights['username'])        # List of all users
past_weightlist = list(userweights['normcombined'])
Nlist = list(userweights['numtransitclasses'])

print("# of features for analysis:  "+str(len(featurelist)))


''' Define score calculation function '''

def score(Wlist, wlist):
    # Wlist = list of weights of users who examined lightcurve
    # wlist = list of weights of users who agreed/disagreed (scoreyes/scoreno) that the transit occurred
    if len(wlist) == 0:
        return 0.0
    else:
        return 1.0/sum(Wlist) * sum(wlist)





''' Iteration loop '''
# We have to write a while loop to iterate transit scoring til the mean normalized combined userweight is approximately 1.0 (within tolerance).

do_continue = True
iteration_count = 0

# For measuring progress in the loop later
iteration_counts = []
changing_counts = []
weight_distribs = [list(userweights['normcombined'])]
diff_distribs = []
score_distribs = [list(score_seeds['scoreyes'])]
meanmagdiffs = []

# For plotting later
labels_wstart = ["start"]
Ncolors = 8
colormap = plt.cm.viridis   # LinearSegmentedColormap
Ncolors = min(colormap.N, Ncolors)
mapcolors = [colormap(int(x*colormap.N/Ncolors)) for x in range(Ncolors)]
l_styles = ['-','--','-.',':',(0,(1,10)),(0,(5,10)),(0,(3,5,1,5)),(0,(3,1,1,1)),(0,(3,5,1,5,1,5))]

for i,(linestyle, color) in zip(range(1), itertools.product(l_styles, mapcolors)):
    linestyle_zero = linestyle
    color_zero = color


# Plot and save starting states of scores and weights

# Userweights
bins = np.logspace(-4,2,41)
fig,ax = plt.subplots()
ax.hist(weight_distribs[-1], bins=bins, log=True, histtype='step', edgecolor=color_zero, stacked=True, fill=False, linewidth=1.6, linestyle=linestyle_zero, label=labels_wstart[-1])
ax.legend(loc=1, prop={'size': 10})
ax.set_xscale('log')
ax.set_xlim(5*10**(-4),50)
ax.set_ylim(ymax=10**5)
ax.set_yticks([10**0,10**1,10**2,10**3,10**4,10**5])
ax.set_xlabel("User weight")
ax.set_ylabel("Count")
fig.suptitle("User weight distribution as iteration progresses", y=0.99)
plt.subplots_adjust(left=0.10, right=0.95, top=0.94)
plt.savefig(user_directory+'plots/weight-distrib-changes/weight-distrib-change-'+str(iteration_count)+'.png')     # New file
plt.close()

# Scores
bins = np.linspace(0,1,31)
fig,ax = plt.subplots()
ax.hist(score_distribs[-1], bins=bins, log=True, histtype='step', edgecolor=color_zero, stacked=True, fill=False, linewidth=1.6, linestyle=linestyle_zero, label=labels_wstart[-1])
ax.legend(loc=1, prop={'size': 10})
ax.set_xlim(-0.1,1.1)
ax.set_ylim(10, 5*10**5)
ax.set_yticks([10**1,10**2,10**3,10**4,10**5])
ax.set_xlabel("Transit score")
ax.set_ylabel("Count")
fig.suptitle("Transit 'yes' score distribution as iteration progresses", y=0.99)
plt.subplots_adjust(left=0.10, right=0.95, top=0.94)
plt.savefig(user_directory+'plots/score-distrib-changes/score-distrib-change-'+str(iteration_count)+'.png')   # New file
plt.close()


# Record loop start time
loop_start_time = timeit.default_timer()


# First line of output file
filewrite.writerow([str(iteration_count)]+['N/A','N/A','N/A']+[str(np.mean(weight_distribs[0]))]+[str(np.mean(score_distribs[0]))]+[str(timeit.default_timer() - loop_start_time)])
file.close()

# MAIN LOOP
while do_continue == True:
    
    iteration_count += 1
        
    # Userweight adjustment function
    def adjust_weight(the_user):
        idx = int(np.where(np.asarray(userlist)==the_user)[0][0])
        transits_classified = ast.literal_eval(userweights['transitsclassified'][userweights['username']==the_user].iloc[0])
        N = int(Nlist[idx])
        past_weight = float(past_weightlist[idx])
        scores = []
        error_flag = False
    
        if len(transits_classified)==0:
            new_weight = past_weight
        
        else:
            for i in range(len(transits_classified)):       # may be as few as 1
                usersyes = score_seeds['usersyes'][score_seeds['transitid']==transits_classified[i]]
                usersyes = usersyes.iloc[0][1:-1]
                usersyes = usersyes.split(',')
                
                usersno = score_seeds['usersno'][score_seeds['transitid']==transits_classified[i]]
                usersno = usersno.iloc[0][1:-1]
                usersno = usersno.split(',')
                
                if the_user in usersyes:
                    scores.append(float(score_seeds['scoreyes'][score_seeds['transitid']==transits_classified[i]]))
                    continue
                
                elif the_user in usersno:
                    scores.append(float(score_seeds['scoreno'][score_seeds['transitid']==transits_classified[i]]))
                    continue
                
                else:
                    print("Weight adjustment error:  User "+the_user+" did not classify transit metafeature "+str(transits_classified[i])+".")
                    # Change error flag to discontinue while loop
                    error_flag = True
                    break
            new_weight = float(1.0/N * np.sum(scores))
    
        return [the_user, new_weight, error_flag]
    
    # Initialize timer
    start_time = timeit.default_timer()        
    
    
#    new_weight_results = []
#    for user in userlist:
#        new_weight_results.append(adjust_weight(user))
    
    
    # Pool
    pool = mp.Pool(mp.cpu_count())
    new_weight_results = []
    # Asynchronous mapping
    new_weight_results = pool.map_async(adjust_weight, [user for user in userlist]).get()
    # Close pool
    pool.close()
    
    # Get time results
    elapsed = timeit.default_timer() - start_time
#    print("Test, elapsed time = "+str(elapsed)+" sec.")                         # TEST ONLY
    total_elapsed = timeit.default_timer() - loop_start_time
    print("User weights adjusted, iteration # "+str(iteration_count)+":  "+str(elapsed)+" sec.")
    print("Total elapsed time = "+str(total_elapsed)+" sec.")
    
    # Dissect results into easy-to-organize lists
    user_check = []
    new_weightlist = []
    flaglist = []
    for x in range(len(new_weight_results)):
        user_check.append(new_weight_results[x][0])
        new_weightlist.append(new_weight_results[x][1])
        flaglist.append(new_weight_results[x][2])
    
    # Check to see if an error was raised during weight adjustment
    if True in flaglist:
        do_continue = False
    
    # Normalization:
    for y in range(len(new_weightlist)):
        new_weightlist[y] = float(new_weightlist[y])/np.mean(new_weightlist)
    
    
    # Transit scoring function
    def calc_score(transit_id):
        scorelist = []
        
        usersyes = score_seeds['usersyes'][score_seeds['transitid']==transit_id]        # Users who identified the feature
        usersyes = usersyes.iloc[0][1:-1]
        usersyes = usersyes.split(',')
        
        usersno = score_seeds['usersno'][score_seeds['transitid']==transit_id]          # Users who did NOT identify the feature
        usersno = usersno.iloc[0][1:-1]
        usersno = usersno.split(',')
        
        weightsyes = []
        weightsno = []
        
        # For yes:
        if len(usersyes)==0:
            scoreyes = 0.0
            scoreno = 1.0
        
        elif (usersno == ['']):
            scoreno = 0.0
            scoreyes = 1.0
        
        else:
            for m in range(len(usersyes)):
                temp_idx = int(np.where(np.asarray(user_check)==usersyes[m])[0][0])     # Index corresponding to user weight in
                weightsyes.append(new_weightlist[temp_idx])                             # user_check and new_weightlist
        
            for n in range(len(usersno)):
                temp_idx = int(np.where(np.asarray(user_check)==usersno[n])[0][0])
                weightsno.append(new_weightlist[temp_idx])
                
            allweights = weightsyes + weightsno
            
            # Calculate scores
            scoreyes = score(allweights, weightsyes)
            scoreno = score(allweights, weightsno)
        
        # Append scores to scorelist
        scorelist.append([transit_id, scoreyes, scoreno])
        
        return scorelist
    
    # Initialize timer
    start_time = timeit.default_timer()
    # Pool
    pool = mp.Pool(mp.cpu_count())
    score_results = []
    # Asynchronous mapping
    score_results = pool.map_async(calc_score, [transit_id for transit_id in featurelist]).get()
    # Close pool
    pool.close()
    
    # Get time results
    elapsed = timeit.default_timer() - start_time
#    print("Test, elapsed time = "+str(elapsed)+" sec.")                         # TEST ONLY
    total_elapsed = timeit.default_timer() - loop_start_time
    print("Transit scores calculated, iteration # "+str(iteration_count)+":  "+str(elapsed)+" sec.")
    print("Total elapsed time = "+str(total_elapsed)+" sec.")
    
    # Dissect results into easy-to-organize lists
    transitid_check = []
    scoreyeslist = []
    scorenolist = []
    for x in range(len(score_results)):
        transitid_check.append(score_results[x][0][0])
        scoreyeslist.append(score_results[x][0][1])
        scorenolist.append(score_results[x][0][2])
    
    # JUST GOTTA SORT THESE LISTS AND USE THEM TO RECONSTRUCT THE SCORE_SEEDS DF SO THE FUNCTION CALLS EVERYTHING CORRECTLY
    score_update = pd.DataFrame(np.column_stack([transitid_check, scoreyeslist, scorenolist]), columns=['transitid_check', 'new_scoreyes', 'new_scoreno'])
    score_update['transitid_check'] = score_update['transitid_check'].astype('int')
    score_update = score_update.sort_values(by=['transitid_check'], ascending=True)
    
    # Check to make sure transit id matching is correct, just once!
    if iteration_count==1:
        for q in range(len(score_update)):
            old_id = int(score_seeds['transitid'].iloc[q])
            new_id = score_update['transitid_check'].iloc[q]
            if old_id != new_id:
                print("Post-scoring error:  Transit ID mismatch")
                do_continue = False
                break
            else:
                continue
    
    # Replace old score columns in score_seeds dataframe
    score_seeds['scoreyes'] = score_update['new_scoreyes']
    score_seeds['scoreno'] = score_update['new_scoreno']
    
    
    # Userweight difference function
    def weight_diff_check(the_user):
        idx1 = int(np.where(np.asarray(userlist)==the_user)[0][0])              # idx1 corresponds to index in userlist and past_weightlist
        idx2 = int(np.where(np.asarray(user_check)==the_user)[0][0])            # idx2 corresponds to index in user_check and new_weightlist
        past_weight = float(past_weightlist[idx1])
        new_weight = float(new_weightlist[idx2])
        return [the_user, float(new_weight-past_weight), np.abs(new_weight-past_weight), np.abs(new_weight-past_weight)<0.1, np.isclose(past_weight, new_weight, rtol=1e-03)]    # [user, weightdiff, abs_weightdiff, largeflux_flag, smallflux_flag]
    
    # Map weight difference function to all users
    # Don't need to time this part; should be quick!
    pool = mp.Pool(mp.cpu_count())
    weight_diff_results = []
    # Asynchronous mapping
    weight_diff_results = pool.map_async(weight_diff_check, [user for user in userlist]).get()
    # Close pool
    pool.close()
    
    
    # weight_diff_results should be a list of lists, the fourth and fifth columns of which are lists of Booleans
    # We want to know if there are instances of False in the fourth column, which signify big fluctuations of user weights
    if False not in np.array(weight_diff_results, dtype='O')[:,3]:
        largeflux_flag = False
        largeflux_count = 0
        print("No user weight fluctuations > 0.1 unit.")
    else:
        largeflux_flag = True
        largeflux_count = list(np.array(weight_diff_results, dtype='O')[:,3]).count(False)
        print("# of weight fluctuations >= 0.1 unit:  "+str(largeflux_count))
    
    # Count the number of individual weights still changing at all (the smallflux_flag)
    changing_count = list(np.array(weight_diff_results, dtype='O')[:,4]).count(False)
    print("Number of user weights still changing = "+str(changing_count))   # and print it to give us an idea of how long we still have to go
    
    # From the third column of weight_diff_results, which are the absolute values of the weight differences, we want the mean

    meanmagdiff = np.mean(np.array(weight_diff_results)[:,2].astype(np.float))
    
    # We may still run into a problem with resonant looping
    # In which the changes in weight sway back and forth over iterations
    # Alternating forever, not necessarily with magnitudes less than our threshhold
    # We'll check for this occurring, and if it is, we'll change the loop flag
    # And add some small, random perturbations to the transit scores
    # In an effort to try and break the resonance
    
    meanweight = np.mean(new_weightlist)
    if iteration_count >= 5:
        old_meanweight = np.mean(weight_distribs[-2])
        older_meanweight = np.mean(weight_distribs[-4])
    else:
        old_meanweight = 10000.     # Dummy values so that the loop flag isn't tripped
        older_meanweight = 10000.
    
    meanscore = np.mean(scoreyeslist)
    if iteration_count >= 5:
        old_meanscore = np.mean(score_distribs[-2])
        older_meanscore = np.mean(score_distribs[-4])
    else:
        old_meanscore = 10000.     # Dummy values so that the loop flag isn't tripped
        older_meanscore = 10000.


    # Add points and lists to quantities to be plotted, to observe progress
    iteration_counts.append(iteration_count)
    changing_counts.append(changing_count)
    weight_distribs.append(new_weightlist)   # Note:  this list includes a "start" point corresponding to a "zero'th" iteration
    diff_distribs.append(list(np.array(weight_diff_results)[:,1].astype(np.float)))
    meanmagdiffs.append(meanmagdiff)
    score_distribs.append(scoreyeslist)
    labels_wstart.append("iter. "+str(iteration_count))
    N = len(iteration_counts)+1
    
    
    ''' Pause to plot stuff '''
    
    linestyle_list = []
    color_list = []
    for i,(linestyle, color) in zip(range(N), itertools.product(l_styles, mapcolors)):
        linestyle_list.append(linestyle)
        color_list.append(color)
    
    if iteration_count > 19:
        numcols = 2
        gridwidth = 0.68
    else:
        numcols = 1
        gridwidth = 0.8
    
    # Plot how many weights are changing per iteration
    fig, ax = plt.subplots()
    ax.scatter(iteration_counts, changing_counts)
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Number of weights still changing")
    fig.suptitle("Number of user weights still changing per iteration", y=0.93, x=0.53)
    plt.subplots_adjust(left=0.12, right=0.95)
    plt.savefig(user_directory+'plots/weight-change-rate.pdf')   # Overwrite file to save during every iteration
    plt.close()
    
    
    # Also plot how userweight distribution changes with iteration progress
    # DIAGNOSTIC
    distmin_list = []     # Find min and max weights from all distributions
    distmax_list = []
    for i in range(N):
        distmin_list.append(min(weight_distribs[i]))
        distmax_list.append(max(weight_distribs[i]))
    minexp = math.floor(math.log10(min(distmin_list)))
    maxexp = math.ceil(math.log10(max(distmax_list)))
    bins = np.logspace(minexp,maxexp,7*(maxexp-minexp))     # Seven bins per logspace
    bins0 = list(np.linspace(0,10**minexp,8))
    bins0.pop(-1)
    bins = bins0 + list(bins)
    fig,ax = plt.subplots(gridspec_kw=dict(right=gridwidth))
    for i in range(N):
        ax.hist(weight_distribs[i], bins=bins, log=True, histtype='step', edgecolor=color_list[i], stacked=True, fill=False, linewidth=1.6, linestyle=linestyle_list[i], label=labels_wstart[i])
    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., ncol=numcols, prop={'size': 10})
    ax.set_xscale('symlog', linthreshx=10**minexp)
    ax.set_xlim(0,5*10**maxexp)
    ax.set_ylim(ymax=10**5)
    ax.set_yticks([10**0,10**1,10**2,10**3,10**4,10**5])
    ax.set_xlabel("User weight")
    ax.set_ylabel("Count")
    fig.suptitle("User weight distribution as iteration progresses", y=0.99)
    plt.subplots_adjust(left=0.10, right=0.95, top=0.94)
    plt.savefig(user_directory+'plots/weight-distrib-change.pdf')   # Overwrite file to save during every iteration
    plt.close()


    # Individual plots for animation
    fig,ax = plt.subplots()
    ax.hist(weight_distribs[-1], bins=bins, log=True, histtype='step', edgecolor=color_list[-1], stacked=True, fill=False, linewidth=1.6, linestyle=linestyle_list[-1], label=labels_wstart[-1])
    ax.legend(loc=1, prop={'size': 10})
    ax.set_xscale('symlog', linthreshx=10**minexp)
    ax.set_xlim(0,5*10**maxexp)
    ax.set_ylim(ymax=10**5)
    ax.set_yticks([10**0,10**1,10**2,10**3,10**4,10**5])
    ax.set_xlabel("User weight")
    ax.set_ylabel("Count")
    fig.suptitle("User weight distribution as iteration progresses", y=0.99)
    plt.subplots_adjust(left=0.10, right=0.95, top=0.94)
    plt.savefig(user_directory+'plots/weight-distrib-changes/weight-distrib-change-'+str(iteration_count)+'.png')     # New file
    plt.close()



    
    # And the userweight difference distribution with iteration progress
    # DIAGNOSTIC
    Nbins = 7
    bins1 = np.zeros(shape=int(Nbins*5))                # For negative side
    bins2 = np.linspace(-10**(-3),10**(-3),Nbins)       # For near zero
    bins3 = np.logspace(-3,2,int(Nbins*5))              # For positive side
    for i in range(len(bins1)):
        bins1[i] = bins1[i]-bins3[-(i+1)]               # Mirror positive side into negativve
    bins2 = list(bins2)                                 # Make middle into a proper list
    bins2.pop(0)                                        # Pop duplicates
    bins2.pop(-1)
    bins = list(bins1)+bins2+list(bins3)                # Append all into single list of bin bounds
    
    fig,ax = plt.subplots(gridspec_kw=dict(right=gridwidth))
    for i,(linestyle, color) in zip(range(N-1), itertools.product(l_styles, mapcolors)):
        ax.hist(diff_distribs[i], bins=bins, log=True, histtype='step', edgecolor=color, stacked=True, fill=False, linewidth=1.6, linestyle=linestyle, label="iter. "+str(i+1))
    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., ncol=numcols, prop={'size': 10})
    ax.set_xscale('symlog', linthreshx=10**(-3))
    ax.set_xlim(xmax=50)
    ax.set_ylim(ymax=3*10**5)
    ax.set_xticks([-10**2,-10**1,-10**0,0,10**0,10**1])
    ax.set_yticks([10**0,10**1,10**2,10**3,10**4])
    ax.set_xlabel("User weight difference")
    ax.set_ylabel("Count")
    fig.suptitle("User weight difference distribution as iteration progresses", y=0.99)
    plt.subplots_adjust(left=0.10, right=0.95, top=0.94)
    plt.savefig(user_directory+'plots/weight-diff-distrib-change.pdf')  # Overwrite file to save during every iteration
    plt.close()
    
    
    # Individual plots for animation
    fig,ax = plt.subplots()
    ax.hist(diff_distribs[-1], bins=bins, log=True, histtype='step', edgecolor=color_list[-1], stacked=True, fill=False, linewidth=1.6, linestyle=linestyle_list[-1], label="iter. "+str(iteration_count))
    ax.legend(loc=1, prop={'size': 10})
    ax.set_xscale('symlog', linthreshx=10**(-3))
    ax.set_xlim(xmax=50)
    ax.set_ylim(ymax=3*10**5)
    ax.set_xticks([-10**2,-10**1,-10**0,-10**(-1),-10**(-2),0,10**(-2),10**(-1),10**0,10**1])
    ax.set_yticks([10**0,10**1,10**2,10**3,10**4])
    ax.set_xlabel("User weight difference")
    ax.set_ylabel("Count")
    fig.suptitle("User weight difference distribution as iteration progresses", y=0.99)
    plt.subplots_adjust(left=0.10, right=0.95, top=0.94)
    plt.savefig(user_directory+'plots/weight-diff-distrib-changes/weight-diff-distrib-change-'+str(iteration_count)+'.png')  # New file
    plt.close()
    

    
    # And the time series of mean magnitude weight change
    fig, ax = plt.subplots()
    ax.scatter(iteration_counts, meanmagdiffs)
    ax.set_yscale('log')
    ax.set_ylim(ymin=10**math.floor(math.log10(meanmagdiffs[-1])))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Mean change in user weight")
    fig.suptitle("Mean magnitude of the change in user weight as iteration progresses", y=0.93, x=0.53)
    plt.subplots_adjust(left=0.12, right=0.95)
    plt.savefig(user_directory+'plots/meanmagdiffs.pdf')   # Overwrite file to save during every iteration
    plt.close()

    
    
    # And transit score distribution changes with iteration progress
    # DIAGNOSTIC
    bins = np.linspace(0,1,31)
    fig,ax = plt.subplots(gridspec_kw=dict(right=gridwidth))
    for i in range(N):
        ax.hist(score_distribs[i], bins=bins, log=True, histtype='step', edgecolor=color_list[i], stacked=True, fill=False, linewidth=1.6, linestyle=linestyle_list[i], label=labels_wstart[i])
    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., ncol=numcols, prop={'size': 10})
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(10,5*10**5)
    ax.set_yticks([10**1,10**2,10**3,10**4,10**5])
    ax.set_xlabel("Transit score")
    ax.set_ylabel("Count")
    fig.suptitle("Transit 'yes' score distribution as iteration progresses", y=0.99)
    plt.subplots_adjust(left=0.10, right=0.95, top=0.94)
    plt.savefig(user_directory+'plots/score-distrib-change.pdf')   # Overwrite file to save during every iteration
    plt.close()
    
    
    # Individual plots for animation
    bins = np.linspace(0,1,31)
    fig,ax = plt.subplots()
    ax.hist(score_distribs[-1], bins=bins, log=True, histtype='step', edgecolor=color_list[-1], stacked=True, fill=False, linewidth=1.6, linestyle=linestyle_list[-1], label=labels_wstart[-1])
    ax.legend(loc=1, prop={'size': 10})
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(10,5*10**5)
    ax.set_yticks([10**1,10**2,10**3,10**4,10**5])
    ax.set_xlabel("Transit score")
    ax.set_ylabel("Count")
    fig.suptitle("Transit 'yes' score distribution as iteration progresses", y=0.99)
    plt.subplots_adjust(left=0.10, right=0.95, top=0.94)
    plt.savefig(user_directory+'plots/score-distrib-changes/score-distrib-change-'+str(iteration_count)+'.png')   # New file
    plt.close()
    
    
    
    
    
    ''' Make animations '''
    
    for i in range(iteration_count+1):
        im = Image.open(user_directory+'plots/score-distrib-changes/score-distrib-change-'+str(i)+'.png')
        im.save(user_directory+'plots/score-distrib-changes/anim/score-distrib-change-'+str(i)+'.gif')
        im = Image.open(user_directory+'plots/weight-distrib-changes/weight-distrib-change-'+str(i)+'.png')
        im.save(user_directory+'plots/weight-distrib-changes/anim/weight-distrib-change-'+str(i)+'.gif')
    
    for i in range(1,iteration_count+1):
        im = Image.open(user_directory+'plots/weight-diff-distrib-changes/weight-diff-distrib-change-'+str(i)+'.png')
        im.save(user_directory+'plots/weight-diff-distrib-changes/anim/weight-diff-distrib-change-'+str(i)+'.gif')
    
    
    
    # Score distribution change over iterations
    # Open all the frames
    images = []
    for i in range(iteration_count+1):
        frame = Image.open(user_directory+'plots/score-distrib-changes/anim/score-distrib-change-'+str(i)+'.gif')
        images.append(frame)
    
    # Save the frames as an animated GIF
    images[0].save(user_directory+'plots/score-distrib-changes/anim/scores.gif',
                   save_all=True,
                   append_images=images[1:],
                   duration=700,
                   loop=0)                                                      # Overwrite gif from previous iteration
    
    
    # Weight distribution change over iterations
    # Open all the frames
    images = []
    for i in range(iteration_count+1):
        frame = Image.open(user_directory+'plots/weight-distrib-changes/anim/weight-distrib-change-'+str(i)+'.gif')
        images.append(frame)
    
    # Save the frames as an animated GIF
    images[0].save(user_directory+'plots/weight-distrib-changes/anim/weights.gif',
                   save_all=True,
                   append_images=images[1:],
                   duration=700,
                   loop=0)                                                      # Overwrite gif from previous iteration
    
    
    # Weight diff distribution change over iterations
    # Open all the frames
    images = []
    for i in range(1,iteration_count+1):
        frame = Image.open(user_directory+'plots/weight-diff-distrib-changes/anim/weight-diff-distrib-change-'+str(i)+'.gif')
        images.append(frame)
    
    # Save the frames as an animated GIF
    images[0].save(user_directory+'plots/weight-diff-distrib-changes/anim/weight-diffs.gif',
                   save_all=True,
                   append_images=images[1:],
                   duration=700,
                   loop=0)                                                      # Overwrite gif from previous iteration
    
    
    
    
    
    # Don't forget to update past_weightlist!
    # Crosscheck indices using user_check and userlist
    for z in range(len(user_check)):                                            # z corresponds to index in user_check and new_weightlist
        idx = int(np.where(np.asarray(userlist)==user_check[z])[0][0])          # idx corresponds to index in userlist and past_weightlist
        past_weightlist[idx] = float(new_weightlist[z])                         # Replace past weights with new ones for next iteration
    
    
    # Pause to save state of weights and scores just in case next iteration is interrupted
    # We just need to store the new userweights in the established dataframe and re-save both the score frame and the weights frame
    # Use past_weightlist, since its indices are the same as userlist, and therefore the userweights dataframe
    weight_update = pd.DataFrame(np.column_stack([past_weightlist]), columns=['new_weight'])
    userweights['normcombined'] = weight_update['new_weight']
    
    userweights.to_csv(user_directory+'temp_userweights.tsv', sep="\t", index=False)
    score_seeds.to_csv(user_directory+'temp_transitscores.tsv', sep="\t", index=False)
    
    
    
    # Write values for this iteration to csv
    file = open(user_directory+'outputlog.csv', 'a', newline='')
    filewrite = csv.writer(file, delimiter=',')
#    line = [str(iteration_count)]+[str(changing_count)]+[str(largeflux_count)]+[str(meanmagdiff)]+[str(meanweight)]+[str(meanscore)]+[str(timeit.default_timer() - loop_start_time)]
    line = [str(34)]+[str(changing_count)]+[str(largeflux_count)]+[str(meanmagdiff)]+[str(meanweight)]+[str(meanscore)]+['N/A']
    filewrite.writerow(line)
    file.close()


    # Final convergence check, and perturbing transit scores if necessary
    if np.isclose(meanmagdiff, 0.0, atol=1e-03) == True:
        do_continue = False
        print("Mean magnitude weight change is sufficiently close to zero; convergence achieved.")
    elif ((np.isclose(meanweight, old_meanweight, rtol=1e-03)==True) and (np.isclose(meanweight, older_meanweight, rtol=1e-03)==True)) and ((np.isclose(meanscore, old_meanscore, rtol=1e-03)==True) and (np.isclose(meanscore, older_meanscore, rtol=1e-03)==True)):
        print("Resonant loop suspected.")
        perturb = np.random.uniform(low=-0.04, high=0.04, size=(len(scoreyeslist),))
        
        scoreyeslist = [x + y for x, y in zip(scoreyeslist, perturb)]
        scorenolist = [x - y for x, y in zip(scorenolist, perturb)]

        # Reupdate transit scores
        score_update = pd.DataFrame(np.column_stack([scoreyeslist, scorenolist]), columns=['new_scoreyes', 'new_scoreno'])        
        # Replace old score columns in score_seeds dataframe
        score_seeds['scoreyes'] = score_update['new_scoreyes']
        score_seeds['scoreno'] = score_update['new_scoreno']       
        print("Transit scores perturbed.")
    else:
        pass
    


    # So, at this point, we have done three things in parallel processes:
    # - Adjusted userweights
    # - Recomputed transit scores
    # - Checked userweight differences from before and after adjustment
    # The first two of these processes are timed, while the last is not.
    
    # The 'scoreyes' column of the score_seeds df has been updated
    # The 'scoreno' column of the score_seeds df has been updated
    # The past_weightlist list has been updated
    # The plot of how much the weights changed has been updated and saved
    
    # There are three ways that the do_continue Boolean can be changed to False:
    # - an error is raised during weight adjustment
    # - an error is raised while replacing recalculated transit scores
    # - either the new weights are sufficiently similar to their corresponding old weights (on average) or the mean of the weight difference distribution is sufficiently close to zero

    # Otherwise, do_continue remains True, and the while loop continues to the next iteration.
    
    
    ''' End while loop '''

''' By the end of the iteration, we should have our final userweights, stored as both new_userweights and past_userweights, and the score_seeds dataframe should already be updated with the final transit scores.  We don't even need to recalculate those!  Furthermore, the userweights and the transit scores are already overwritten into their proper frames, from the last step of the loop. '''


# Rewrite to final files
userweights.to_csv(user_directory+'final_userweights.tsv', sep="\t", index=False)
score_seeds.to_csv(user_directory+'final_transitscores.tsv', sep="\t", index=False)







""" End program. """