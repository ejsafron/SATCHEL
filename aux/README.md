## Metafeatures

In any crowdsourced analysis of one-dimensional time-series data, we assume that multiple users will classify each subject, and that each user may make multiple marks on the subject.  An obvious feature may be marked many times, while a less obvious one may be marked by only one user.

The pipeline requires a list of features to score, but we do not want to score the same feature more than once.  This code is a prescription for one method of combining user markings into "metafeatures," for scoring.  This code will not work <em>tantum et tale</em> for data from other projects, as it both requires and generates auxiliary information specific to the <em>Kepler</em> light curves.  It may, nevertheless, be useful for those interested in adapting its methodology.

The only input that is strictly required is a csv-readable classifications file.  This file must contain the following, for each classification:
- `subject_id`:  string; unique identifier for the subject that was classified
- `xMinGlobal`:  float; early edge of marking made by user, or `NaN` if no marking was made
- `xMaxGlobal`:  float; late edge of marking made by user, or `NaN` if no marking was made
- `synthetic`:  Boolean; `True` if the subject was a simulation, `False` otherwise.

For each subject, all classifications for which users made no markings are filtered out.  If any classification rows are left, the mark from each classification is checked for matches in other rows.  The mark is ignored even before matchining if:
1. It is wider than the adjustable `cutoff` variable (very wide marks are both unrealistic within the context of the <em>Kepler</em> data and problematic when averaging endpoints later)
2. It been consolidated with other marks already
3. More than half the datapoints comprising the mark lie within a data gap (the <em>Kepler</em> data contains many such gaps, corresponding to safety shut downs and scheduled orientation changes, etc.)

Another mark is considered a match if:
1. It satisfies all of the same above criteria
2. It is not the same mark as the original (by index)
3. If its midpoint lies within an adjustable `tolerance` of the original mark's midpoint.

Once all matches to the original mark have been found, each matching mark is itself checked for matches.  All marks with identical matchlists are put into a final match list, their endpoints averaged and recorded, and their indices placed in the `done` list.  A `transitid` is assigned to the consolidated metafeature, and the following information is written to the output file:
- `transitid`
- `subject_id`
- `kepler_id`
- `xmin` (float, averaged `xMinGlobal` from final match list)
- `xmax` (float, averaged `xMaxGlobal` from final match list)
- `duration` (float, `xmax` - `xmin`)
- `midpoint` (`np.mean([xmin,xmax])`)
- `synthetic`
- `dupe_flag` (deprecated)
- `indices` (deprecated)

The `kepler_id` is not necessary for the main pipeline, and may be commented out or removed.  Similarly, the chunk of code from `# Test point to find right lightcurve file` to `delta = np.mean(calcdelta)` may be commented out or removed, as long as the later mentions of `numpoints`, `numdata`, `datafrac`, `testnumpoints`, `testnumdata`, and `testdatafrac` are also removed or commented out.  These are only for the purpose of checking the data gap issue, so they're meaningless for data from other projects where gaps were not a problem.

Otherwise, the code may be used generally.

To run in command line:
```
$> python build-metafeatures.py
```
Default output:
- `metafeatures.csv`

Approximate runtime (for ~1.25m classifications, on a 16-thread machine with 132GB RAM):  ~12.5 hours.
