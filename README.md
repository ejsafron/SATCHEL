# PH-pipeline
Pipeline to search for long-period exoplanet signals in <i>Kepler</i> time-series data classified on the Planet Hunters website.

This pipeline is a work in progress, currently in development as a Ph.D. project.  The first piece of the pipeline (userweighting.py) was largely contributed by Joey Schmitt, from work done in 2015.  That piece was heavily edited, the rest were written, and all are subject to editing by Emily Safron, Ph.D. candidate under Dr. Tabetha Boyajian at Louisiana State University.  Email:  ejsafron@gmail.com

<h3>Background</h3>

The Planet Hunters (PH) is a citizen science project designed as a novel method for detecting exoplanet signals in time-series photometry, called <i>light curves</i>, taken by space telescopes.  From 2010 to 2018, PH hosted light curves from the <i>Kepler</i> space telescope, whose main mission consisted of observing about 156,000 stars in a relatively small field of the sky for about four years.  While this dataset is yet incomplete on PH, there have been several billion classifications done by several hundred thousand users.  Starting in 2018, PH began hosting light curves from the Transiting Exoplanet Survey Satellite (<i>TESS</i>), which is in the process of surveying almost the entire sky for transit signals in 28 day intervals.  Classification of <i>TESS</i> light curves by PH users is complete out to the current availability of data, comprising about 320,000 classifications of 20,000 light curves by 18,000 users.  A similar pipeline to this one is currently in use by Nora Eisner for the search of PH classifications to identify notable planet candidates in the <i>TESS</i> data.

The <i>Kepler</i> stellar catalog contained main sequence stars of many spectral types, but the PH science team prioritized the full classification of the M-type stars after that of the entirety of Quarters 0, 1, and 2.  This decision was motivated by studies done early that decade (e.g., Howard et al. 2010) that showed higher occurrence rates of planets around cool stars.  The team selected M-dwarfs from the <i>Kepler</i> catalog using two conservative cutoffs, which they felt could be further refined later:  T<sub>eff</sub> < 4,200 K, and log(<i>g</i>) > 4.5.  This yielded 5,625 stars.  Into 40 of these light curves, chosen such that they did not contain obvious exoplanet, eclipsing binary, or brown dwarf signals, synthetic transit signals constructed at random from uniform distributions of planetary radius, period, and other properties were injected to create several thousand simulation light curves.  These simulations were shown to users in the same anonymous manner as the real, unmodified <i>Kepler</i> data, to provide an unbiased tool for measuring both each individual user's ability to distinguish signals in the light curves as well as the recovery efficiency of the system as a whole.

The PH back end exports all classifications in a .csv file.  After further filtering of the M-dwarf sample based on more recent temperature and surface gravity constraints (specifically, Gaidos et al. 2016), I filtered these down to about 1.85 million classifications of 3,267 stars, divided into 111,876 light curves, by 31,797 users.  The .csv file has also been expanded to read several things from the "data_location" field into more convenient data columns.  It is the primary input for all further pieces of the pipeline.  (While I don't consider the code that I used to modify this .csv file to be part of the pipeline itself, I will include it in this repository as an auxiliary file.)

<h3>Repository structure</h3>

The file structure of this repository is as follows:
<ul>
  <li>README.md</li>
  <li>Inputs
    <ul>
      <li>Files necessary to begin the process, plus auxiliary files</li>
    </ul>
  </li>
  <li>Main
    <ul>
      <li>The .py files that make up the body of the pipeline itself</li>
      <li>Auxiliary input/output
        <ul>
          <li>Output files generated by pipeline pieces that become input for later processes</li>
        </ul>
      </li>
    </ul>
  </li>
  <li>Outputs
    <ul>
      <li>Main output files, including plots</li>
    </ul>
  <li>Testing & optimization
    <ul>
      <li>Codes that have performed various tests and comparisons on the pipeline</li>
      <li>Auxiliary files necessary to perform tests</li>
      <li>Output</li>
    </ul>
  </li>
</ul>

<h3>Process structure</h3>

Below is a visual representation of how the pipeline works, from the user interaction with the data all the way through the end of the pipeline (where human interaction begins).  The end goal, in my case, is to get a list of long-period Planet Hunters Objects of Interest (PHOIs).  This involves a couple of stages of vetting post-pipeline.

<img src="https://drive.google.com/thumbnail?id=10b38qTVmNyk3WTgyX0ROiCYtjY5Yuvr7" alt="please work" height=330>

Each individual process is explained in more detail in README.txt form.  For more detailed information about the vetting processes, please see [paper, in prep.].

<h3>To run or adapt this pipeline:</h3>

<h3>To report bugs:</h3>

<h3>To cite this work:</h3>