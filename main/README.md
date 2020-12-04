The main _____ pipeline is comprised of four pieces, which must be run in sequential order.  The default packages required are:
- `numpy` (handling array-like objects, indexing, number generation, and some math operations)
- `pandas` (handling dataframes)
- `ast` (parsing fields from some input files)
- `multiprocessing` (parallel processing)
- `timeit` (recording runtimes)
- `matplotlib` (plotting)
- `itertools` (plotting options)
- `os` (multiprocessing setup)
- `csv` (file-writing)
- `math` (some math operations)
- `PIL` (output animations)
- `astropy` (table construction)

The order in which the pieces must be run is
1. `user-weighting.py`
2. `build-uwtable.py`
3. `score-seeds.py`
4. `iteration.py`

Each of the first three produce at least one output file that is necessary input the next piece, and possibly later pieces.  Each piece is described below, including input format, run commands, default output, etc.  Some processes are parallelized, so runtime will vary with both input and machine.  All reported runtimes are for data involving ~30k users, ~500k metafeatures, and ~1.25m classifications, on a 16-thread machine with 132GB RAM.

## 1.  `user-weighting.py`

The first piece of the main pipeline is `user-weighting.py`, which requires as input a csv-readable file containing information about simulations (`syntheticdata.csv`) and another csv-readable file containing user classifications (`mdwarf-classifs.csv`).  <div class="text-red mb-2">[NOTE:  Make this callable through user input.]</div>  The following fields are **necessary** in each input file:

**in simulations file: **
- `syntheticid`:  integer; unique identifier for a simulation subject
- `synmin`:  float; early edge of synthetic signal in simulation subject
- `synmax`:  float; late edge of synthetic signal in simulation subject
- `signalid`:  string; unique identifier for each signal of simulation subjects

**in classifications file: **
- `synthetic`:  Boolean; True if classified subject was a simulation, False otherwise
- `subject_id`:  string; unique identifier for each subject
- `synthetic_id`:  integer or NaN; identifier that matches `syntheticid` in simulations file, if subject was a simulation, else NaN
- `user_name`:  string; name of user
- `classification_id`:  string; unique identifier corresponding to an instance of a user seeing a subject
- `xMinGlobal`:  float; early edge of user mark on subject, in global time, or NaN if no mark was made
- `xMaxGlobal`:  float; late edge of user mark on subject, in global time, or NaN if no mark was made

This first piece gauges user performance on simulated signals.  This is done in two parts:

First, the user-made marks must be matched to the synthetic signals (PART 1: MARK MATCHING).  For each synthetic signal (some of which occur in the same subject), some number of classifications were done by users.  For each signal, user marks with non-zero overlap and a midpoint within user-specified tolerance of the synthetic signal's midpoint are found and recorded.  If a single user made more than one matching mark, only the mark with the closest midpoint is recorded as the match.  No user mark counts as a match for more than one synthetic signal.  If a user made no matching marks on a simulation, 'NaN's are recorded.  These records are preserved in `match-user-synthetics.csv`.

Second, user weights are calculated based on how correctly simulations were classified (PART 2: USER WEIGHTING).  Users are "upweighted" for correct markings, and "downweighted" for incorrect markings.  Downweights are not given for failure to mark signals.  The exact amount of each upweight given for correctly identifying a synthetic signal is impacted by the "signal ID completeness," which is a measure of how difficult a signal is to identify based on the percentage of users that find it, and by a decay function that decreases the relative upweight for multiple signals found in a single subject.  These values, for all users who saw simulations, are recorded in `user-weighting.csv`.

To run in command line, execute
```
$> python user-weighting.py
```
in the directory containing the input files.

Default output:
- `match-user-synthetics.csv`
- `user-weighting.csv`.

Runtime:  ~3 hours.

<div class="text-red mb-2">[NOTE:  Plot of resulting user weight distribution?]</div>

## 2.  `build-uwtable.py`

This piece of the pipeline constructs the full set of user weight seeds for all users (including any who did not see simulations in the project interface).  In addition to copying in values for users who have weights based on simulation performance and assigning dummy weights to all others, this code also sifts through the classifications file to find which metafeatures were classified by each user.  The unique metafeature IDs for each user are stored in a field of the output file, for quicker access during the main loop of the later pipeline piece `iteration.py`.

The input required to run this piece is
- csv-readable classifications file
- `user-weighting.csv`
- csv-readable metafeatures file

The fields necessary in the metafeatures file are:
- `feature_id`:  integer; unique identifier for each metafeature
- `subject_id`:  string; unique identifier for each subject, corresponding to `subject_id` from classifications file
- `xmin`:  float; early edge of metafeature on subject, in global time
- `xmax`:  float; late edge of metafeature on subject, in global time
- `duration`:  float; duration of metafeature on subject, equivalent to `xmax` - `xmin`
- `midpoint`:  float; midpoint of metafeature on subject, in global time, equivalent to the mean of `xmin` and `xmax`
- `synth_flag`:  Boolean; True if metafeature is in a simulation subject, False otherwise

The dummy weight is easily adjustable near the beginning of the code, under the unsurprising variable name `dummy`.

To run in command line, execute
```
$> python build-uwtable.py
```
in the directory containing the input files.

Default output:
- `full_uwseeds.tsv`

Runtime:  ~9.4 hours

## 3.  `score-seeds.py`

This piece calculates preliminary scores for each metafeature based on the user weights from the previous step, including both simulation performance and dummy weights.

Input required:
- csv-readable classifications file
- `full_uwseeds.tsv`
- csv-readable metafeatures file

The scores are weighted averages of "yes" and "no" votes, where a user mark indicates a "yes" and a lack of mark indicates a "no," rationalized by the weight of all users who saw the subject containing the feature.  For more detais, please see Safron et al. (in prep).

During the process, the users who saw each metafeature are identified and stored as a field in the output file, for quicker access later.

To run in command line, execute
```
$> python score-seeds.py
```

Default output:
- `transitscoreseeds.tsv`

Runtime:  ~1.2 hours.

## 4.  `iteration.py`

This last pipeline piece is particularly important in the case where a significant fraction of real classifications are done by users who have never seen simulations.  It iteratively adjusts both the user weights and the metafeature scores, using user agreement with majority votes on highly-scoring features as a metric.  After each adjustment of the weights, the distribution is renormalized such that the mean is unity.

Input required:
- csv-readable metafeatures file
- `full_uwseeds.tsv`
- `transitscoreseeds.tsv`

This code generates an output log file, to which a new line is added after every iteration.  Information contained in the log file includes, by default:
- iteration number
- number of users whose weights are still changing by more than 10<sup>-3</sup> units
- number of users whose weights are still changing by more than 0.1 units
- the numerical difference between the mean of the current weight distribution compared to that of the last iteration
- the mean of the current weight distribution
- the mean of the current score distribution
- the total time elapsed

Many of these quantities are also plotted, by default.  The iteration ceases when the difference between the mean of the current weight distribution compared to that of the last iteration is zero, within a tolerance of 10<sup>-3</sup>, indicating that the mean of the user weight distribution has been successfully converged to unity.  If the weights of any users are still changing by relatively large amounts (an adjustable parameter, 0.1 by default), iteration will continue until those weights have further stabilized.

<img src="https://github.com/ejsafron/PH-pipeline/blob/master/img/meanmagdiffs.png" alt="please work" style="max-width:100%;">

If a resonant cycle occurs in which, for example, a subset of weights are adjusted back and forth each iteration, preventing convergence, a small, random perturbation is added to the score set to attempt to knock the process out of the resonant loop.

For further details, see Safron et al. (in prep).

The state of each distribution is saved (and overwritten) after each successful iteration.  Once convergence has been reached, an informative message is printed and the final distributions are saved.  Below are comparison plots, showing the initial states of the user weight and feature score distributions (from `full_uwseeds.tsv` and `transitscoreseeds.tsv`, respectively) and the final weight and score distributions (from `final_userweights.tsv` and `final_transitscores.tsv`, respectively) for the PH data.

<img src="https://github.com/ejsafron/PH-pipeline/blob/master/img/initial-states.png" alt="please work" style="max-width:100%;">

<img src="https://github.com/ejsafron/PH-pipeline/blob/master/img/final-states.png" alt="please work" style="max-width:100%;">

The way these distributions changed as iterations progressed is illustrated in the animations below.

| | |
|:-------------------------:|:-------------------------:|
|<img width="1604" alt="pleeeeeease work" src="https://github.com/ejsafron/PH-pipeline/blob/master/img/weights.gif">|  <img width="1604" alt="pleeeeeease" src="https://github.com/ejsafron/PH-pipeline/blob/master/img/scores.gif">|

To run in command line, execute:
```
$> python iteration.py
```

Default output:
- `outputlog.csv`
- `final_userweights.tsv`
- `final_transitscores.tsv`
- `temp_userweights.tsv`
- `temp_transitscores.tsv`
- `weight-change-rate.pdf` *
- `weight-distrib-change.pdf` *
- `weight-diff-distrib-change.pdf` *
- `score-distrib-change.pdf` *
- `meanmagdiffs.pdf` *
- `weight-distrib-change-<N>.png` **
- `weight-diff-distrib-change-<N>.png` **
- `score-distrib-change-<N>.png` **
- `scores.gif` *
- `weights.gif` *
- `weight-diffs.gif` *

\* (single plot, updated every iteration)<br/>
\** (individual png and gif saved for every iteration `<N>`)

Runtime:  ~2.5 days (25 iterations)
