Copyright (c) 2016, Maxim Imakaev, Jelena Chuklina
 
Code written by Maxim Imakaev <mimakaev@gmail.com>, Jelena Chuklina <chuklina.jelena@gmail.com>

Licenced under 3-clause BSD (see LICENCE file) 

This is a set of utilities used to calculate promoter sequences starting from a set of upstreams. 

Requirements
============

* Python 2.7 (2.6 may work) 
* numpy
* scipy
* matplotlib
* pandas 
* joblib 
* weblogolib (https://pypi.python.org/pypi/weblogo) 

If python and/or the above packages are not available on your machine, anaconda https://www.continuum.io/downloads will have everything included, except for weblogolib, which can be installed with "pip install weblogo". You do not need root access to use anaconda.

The software was tested on Ubuntu, and will probably work under MAC. It was not tested for Windows, but may work.  


Important general notes
=======================

Each of the utilities operates on a folder with data, which we call scoreFolder in the code. This folder would contain all the intermediate and final results in the different subfolders, and all the plots as well. In the example, this folder is called "scores_NOonly". 

The program works on a list of upstream sequences, which must all be the same length. If they are not, results may be unpredictable. It also assumes that all the upstreams end exactly at the TSS. 

A list of upstreams should be supplied as a **text file** (not fasta), with different sequences in different lines. 

Different parts of the program operate on one folder, and create sub-folders and files inside it. Different parts are to be run in the defined order. They will throw an error if the previous part was not run, and will silently override files if run again with different parameters. This folder is called scoreFolder throughout this text. 

1 Pattern scores 
================

input
-----
List of given upstreams 
Additional parameters defined in sampleLaunch.py

output
------
Matrix of scores in the scoreFolder/saved
Human-readable tables in scoreFolder/CSVs 


First part of the library evaluates all pairs of 6-mers at all positions in all upstreams. This is the most computationally intensive task, and takes up to a several CPU-days. However, it can be parallelized (see below). 

You can speed it up by separating all pairs of positions into groups. By default, the program just loops over all pairs of positions: 

..code: pairs = [(i, j) for i in range(0, seqLen - 3) for j in range(i + 5, seqLen)]

You can create this list beforehand and split it into N separate pieces, and then process each piece in a separate process. 

This part of the library is done by sampleLaunch.py program, which is being called by 00_computeScores.sh script. 

Note that the code of the sampleLaunch.py has several other parameters which you may want to change. 


2 Aggregating all scores
========================

input
-----
List of given upstreams
Files saved by part 1 in the scoreFolder

output
------
Plots in scoreFolder/analysis
"Cloud" plots and other aggregated tables in scoreFolder/sortedBy

Note that we use 02_sortBy_ScoreNew_8 part way of ranking all patterns, which slices the cloud of scores in the direction we chose (see supplement of the paper).

..note:: From step 2 you can go either to steps 3-4 or to step 5 directly. Those methods are independent. 

3 Run the PCA 
=============

input
-----
List of given upstreams
List of "all" upstreams to score in (provide the same list if you are running it on the full list)
Files saved by part 2 in scoreFolder
Positions of the left and right boxes for the "cloud" you want to explore  (see below) 

output
------

best patterns for each PC in scoreFolder/savedPats
best upstreams for each PC in scoreFolder/savedSeqs


To find positions of boxes in the "cloud", examine scoreFolder/sortedBy/02_sortBy_ScoreNew_8_best500_locations.pdf and look at the location of the most prominent cloud of points. Record x and y positions of the center of the cloud. 


4 Build PWMs from saved sequences
=================================

input
-----
List of given upstreams
List of "all" upstreams to score in (provide the same list if you are running it on the full list)
Files saved by the part 3 in scoreFolder
Positions of the left and right boxes for the "cloud" you want to explore (see below)

output
------
Logos in scoreFolder/savedLogos
Tables with scoring and promoter sequences for each PC in sourceFolder/savedTables

Note that you would usually subtract 1 from the positions of the "cloud" used previously, as the 8-mer used for the PWM extends by 1bp left and right beyound the start of the 6-mer. 


5 Detect promoter sequences
===========================

input
-----
List of given upstreams
list of "all" upstreams to build a table
Files saved by the part 2 

output
------
Table with promoters in sourceFolder/promoterTable.csv

