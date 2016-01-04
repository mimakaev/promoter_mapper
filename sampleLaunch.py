# (c) 2015 Massachusetts Institute of Technology. All Rights Reserved
# Code written by Maxim Imakaev <imakaev@mit.edu>

from getScores import PatternFinder
import sys
import cProfile


if len(sys.argv) != 3:
    print "Usage: python sampleLaunch sequenceFile outputFolder [positions]"
    print "positions should have form 15-25,15-26,15-27,15-28 to evaluate positions 15-25,15-26,etc. "

ALPHA = 1.4
BEST_SINGLE = 30  # number of best scores to select for a single 6-mer. Complexity is proportional to square of this number.
MAX_SCORE = 6
MAX_SCORE_SINGLE = 3
MAX_SUBS = 3
MAX_SHIFT = 3
NMER_LENGTH = 6
"""
This is where all the the other constants are defined for this part only.
ALPHA is the slope of the score for mismatches, defined in the paper

BEST_SINGLE is the number of best n-mers to use for each side of the pattern.
Then all pairwise combinations of BEST_SINGLE x BEST_SINGLE 6-mers will be evaluated at
every pari of positions in the upstream

MAX_SCORE is the maximum score (offset + extension + mismatches) for the whole pattern
MAX_SCORE_SINGLE is the maximum score for scoring separate n-mers (used for selection of BEST_SINGLE only)

MAX_SUBS and MAX_SHIFT are maximum number of subs and maximum offset of a pattern
"""



a = PatternFinder(6, sys.argv[2])
# create a patternFinder object bound to folder provided as a second command line argument

a.loadSequences(sys.argv[1])

# if third argument is provided, we interpret it as a set of locations where to evaluate pattern
if len(sys.argv) >= 4:
    locations = [map(int, j.split("-")) for j in sys.argv[3].split(",")]
    print locations

# If not provided, the program will automatically select it
else:
    locations = None



a.scorePatternsThroughBest(
                        bestNum=BEST_SINGLE,
                        maxScore=MAX_SCORE,
                        maxScoreSingle=MAX_SCORE_SINGLE,
                        maxShift=MAX_SHIFT,
                        maxSubs=MAX_SUBS,
                        penaltySlope=ALPHA,
                        coordinates=locations)

