# (c) 2015 Massachusetts Institute of Technology. All Rights Reserved
# Code written by Maxim Imakaev <imakaev@mit.edu>


import matplotlib
import matplotlib.pyplot as plt
import cPickle
import itertools
import sys, os
import random
import joblib
import numpy as np
import pandas as pd

def translate(word):
    return [{"A":0, "T":1, "G":2, "C":3}[i] for i in word]

def translateRaw(array):
    return "".join([{0:"A", 1:"T", 2:"G", 3:"C"}[i] for i in array])




def removeAxes(mode="normal", shift=0, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    for loc, spine in ax.spines.iteritems():
        if mode == "normal":
            if loc in ['left', 'bottom']:
                if shift != 0:
                    spine.set_position(('outward',
                                        shift))  # outward by 10 points
            elif loc in ['right', 'top']:
                spine.set_color('none')  # don't draw spine
            else:
                raise ValueError('unknown spine location: %s' % loc)
        else:
            if loc in ['left', 'bottom', 'right', 'top']:
                spine.set_color('none')  # don't draw spine
            else:
                raise ValueError('unknown spine location: %s' % loc)


def removeBorder(ax=None):
    removeAxes("all", 0, ax=ax)
    if ax is None:
        ax = plt.gca()
    for _, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
        line.set_visible(False)
    if ax is None:
        ax = plt.axes()
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def fixFormatter(ax=None):
    if ax is None:
        ax = plt.gca()
    matplotlib.rc("axes.formatter", limits=(-10, 10))
    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)
    x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(x_formatter)



def nicePlot(ax="gca", fs=8, show=True):
    """
    replaces obsolete "niceShow" command, packs it with new features
    """
    if ax == "gca":
        ax = plt.gca()
    matplotlib.rcParams.update({'font.size': fs})

    legend = ax.legend(loc=0, prop={"size": fs + 1})
    if legend is not None:
        legend.draw_frame(False)
    removeAxes(shift=0, ax=ax)

    plt.tight_layout(pad=0.3)
    if show:
        plt.show()

def cmap_map(function=lambda x: x, cmap="jet", mapRange=[0, 1]):
    """ Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous     points in a colormap.

    Also trims the "range[0]:range[1]" fragment from the colormap - use this to cut the part of the "jet" colormap!
    """
    if type(cmap) == str:
        cmap = plt.cm.get_cmap(cmap)

    cdict = cmap._segmentdata

    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = map(lambda x: x[0], cdict[key])

    step_list = sum(step_dict.values(), [])
    array = np.array
    step_list = array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step: array(cmap(step)[0:3])
    old_LUT = array(map(reduced_cmap, mapRange[0] + step_list * (
        mapRange[1] - mapRange[0])))
    new_LUT = array(map(function, old_LUT))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(('red', 'green', 'blue')):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = map(lambda x: x + (x[1],), this_cdict.items())
        colorvector.sort()
        cdict[key] = colorvector
    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)



class PatternFinder(object):
    '''
    class which generates patternIndex and substitution tables, \
    as well as scores upstreams and upstream pairs
    '''

    def __init__(self, patternLength, workingFolder, alphabet='ATGC', saveSubs="auto"):
        """
        Constructor of the class
        Parameters
        ----------
        patternLength : int
            Length of each box of the pattern. Use 6 for 6-mers.
        workingFolder : str
            Folder to save results
        saveSubs : "auto" or True or False
            Save a temporary file with a list of patterns with one substitution.
            Set to "True" if you changed an alphabet
            Otherwise "auto" will update files automatically
        """

        if not os.path.exists(workingFolder):
            os.mkdir(workingFolder)
        self.folder = workingFolder
        self.patternLength = patternLength
        self.alphabet = alphabet

        # initializing dictionaries which will convert index to pattern and back
        self.patternToIndex = {''.join(k):i for i, k in \
                               enumerate(itertools.product(alphabet, repeat=patternLength))}

        self.indexToPattern = {v:k for k, v in self.patternToIndex.iteritems()}

        self.patternNum = len(self.patternToIndex)
        L = len(alphabet)
        M = patternLength

        subsFile = os.path.join(workingFolder, "subsCache_pattLen_{0}".format(self.patternLength))
        if saveSubs.lower() == "auto":
            if os.path.exists(subsFile):
                saveSubs = False
            else:
                saveSubs = True

        if saveSubs:
            # actually calculating lists of patterns which can be obtained
            # with X substitutions from each possible given pattern
            self.sub0 = np.zeros(shape=(self.patternNum, 1), dtype=int)
            self.sub0[:, 0] = np.arange(self.patternNum, dtype=int)
            self.sub1 = np.zeros(shape=(self.patternNum, M * (L - 1)), dtype=int)
            self.sub2 = np.zeros(shape=(self.patternNum, M * (M - 1) * (L - 1) ** 2 / 2), dtype=int)
            self.sub3 = np.zeros(shape=(self.patternNum, M * (M - 1) * (M - 2) * (L - 1) ** 3 / 6), dtype=int)

            for pind in xrange(self.patternNum):
                counter = 0
                pattern = self.indexToPattern[pind]
                for pos in range(M):
                    for letter in alphabet:
                        if letter == pattern[pos]:
                            continue
                        newpat = pattern[:pos] + letter + pattern[pos + 1:]
                        newind = self.patternToIndex[newpat]
                        self.sub1[pind, counter] = newind
                        counter += 1
                assert counter == self.sub1.shape[1]
            for pind in range(self.patternNum):
                counter = 0
                pattern = self.indexToPattern[pind]
                for pos1 in range(M):
                    for letter1 in alphabet:
                        if letter1 == pattern[pos1]:
                            continue
                        for pos2 in xrange(pos1 + 1, M):
                            for letter2 in alphabet:
                                if letter2 == pattern[pos2]:
                                    continue
                                newpat = pattern[:pos1] + letter1 + pattern[pos1 + 1:pos2] + \
                                    letter2 + pattern[pos2 + 1:]
                                newind = self.patternToIndex[newpat]
                                self.sub2[pind, counter] = newind
                                counter += 1
                assert counter == self.sub2.shape[1]

            for pind in range(self.patternNum):
                counter = 0
                pattern = self.indexToPattern[pind]
                for pos1 in range(M):
                    for letter1 in alphabet:
                        if letter1 == pattern[pos1]:
                            continue
                        for pos2 in xrange(pos1 + 1, M):
                            for letter2 in alphabet:
                                if letter2 == pattern[pos2]:
                                    continue
                                for pos3 in xrange(pos2 + 1, M):
                                    for letter3 in alphabet:
                                        if letter3 == pattern[pos2]:
                                            continue
                                        newpat = pattern[:pos1] + letter1 + pattern[pos1 + 1:pos2] + \
                                            letter2 + pattern[pos2 + 1:pos3] + letter3 + pattern[pos3 + 1:]
                                        newind = self.patternToIndex[newpat]
                                        self.sub3[pind, counter] = newind
                                        counter += 1
                assert counter == self.sub3.shape[1]
            print "subs file dumped"
            joblib.dump((self.sub0, self.sub1, self.sub2, self.sub3), subsFile, compress=3)

        else:
            if not os.path.exists(subsFile):
                raise IOError("""Please run init with saveSubs=True or "auto" first!""")
            self.sub0, self.sub1, self.sub2, self.sub3 = joblib.load(subsFile)
            assert  len(self.sub0) == self.patternNum


        GC = [sum([s in ["G", "C"] for s in self.indexToPattern[i] ]) for i in xrange(self.patternNum)]
        self.GC = np.array(GC)
        self.GCOnly = np.nonzero(self.GC >= self.patternLength - 1)[0]
        self.singleCache = {}


    def deleteGCInPlace(self, matrix):
        """ Deletes GC-only patterns from matrix
        """
        matrix[self.GCOnly] = 0
        matrix[:, self.GCOnly] = 0


    def adjustByGC(self, matrix):
        """
        Accepts a symmetric matrix self.patternNum x self.patternNum
        Then it iterates over all GC contents (0,1,... patternLen),
        and finds average value over all rows of the matrix with this GC content.
        Then it creates a correction vector, and puts this average for each row.
        It then divides a correction vector by its mean (so it has a mean of 1),
        and divides each row by the corresponding correction value.
        This procedure almost doesn't change the mean of the matrix.

        Then it performs the same for all columns.
        This procedure disbalances rows a little bit,
        so we repeat the procedure one more time for rows and for columns.

        Before doing so, it additionaly deletes all GC-only motifs.

        The iterative procedure is similar to iterative correction (Imakaev, 2012)
        """


        matrix = np.array(matrix)
        self.deleteGCInPlace(matrix)

        while True:
            sumx = np.zeros(len(matrix), dtype=float)
            gcRange = [(0, 2)] + zip(range(2, self.patternLength - 1), range(3, self.patternLength)) + \
            [(self.patternLength - 1, self.patternLength + 1)]

            for curGC in gcRange:
                mask = (curGC[0] <= self.GC) * (self.GC < curGC[1])
                meanValue = np.mean(matrix[mask, :])
                if meanValue == 0:
                    sumx[mask] = -1
                else:
                    sumx[mask] = meanValue
            sumx[sumx == -1] = sumx[sumx != -1].mean()
            sumx /= sumx.mean()
            varx = sumx.var()

            matrix /= sumx[:, None]

            sumy = np.zeros(len(matrix), dtype=float)
            for curGC in gcRange:
                mask = (curGC[0] <= self.GC) * (self.GC < curGC[1])
                meanValue = np.mean(matrix[:, mask])
                if meanValue == 0:
                    sumy[mask] = -1
                else:
                    sumy[mask] = meanValue
            sumy[sumy == -1] = sumy[sumy != -1].mean()
            sumy /= sumy.mean()

            vary = sumy.var()

            matrix /= sumy[None, :]
            if (varx < 0.005) and (vary < 0.005):
                break
        return matrix

    def adjustVectorByGC(self, vector, minAT=2):
        """
        This function accepts a self.patternNum long vector
        and assumes each number corresponds to a pattern.
        It adjusts values corresponding to patterns with the same GC
        so that for each GC mean values of the vector are equal.

        It does it similarly to previous function.
        It additionally deletes all values which have
        AT content less than minAT

        Returns a corrected vector and a correction vector
        """
        vector = np.array(vector, dtype=float)
        erase = self.GC > self.patternLength - minAT
        vector[erase] = 0.
        corr = np.zeros_like(vector)
        corr[erase] = 0.00001
        mask = self.GC < 2
        corr[mask] = vector[mask].mean()
        for curGC in xrange(2, self.patternLength - minAT + 1):
            mask = self.GC == curGC
            corr[mask] = vector[mask].mean()
        corr /= corr.mean()
        return vector / corr, corr





    def scorePatternPairsSimple(self, startFrom=0, step=1,
                                maxSubs=3, maxShift=2,
                                maxExtension=2, maxPenalty=6,
                                maxPenaltySingle=3,
                                adjustByGC=False,
                                eachBestNum=10000):
        """
        #------------------ DEPRECATED ----------------------
        ## Note that this function is mostly deprecated now,
        but can be still used for some exact calculations ##


        This method scores all pattern pairs for up to 7-long patterns.
        It uses sparse logic to enumerate everything.

        Parameters
        ----------

        startFrom, step:int(optional)
            Used to separate work between different processes.
            If you want to split the work, launch step processes with values
            of startFrom from 0 to step-1

        maxScore:int (default=5)
            Maximum penaltu used by algorithm. Do not increase past 6.

        adjustByGC:bool (default=False)
            perform adjustment by GC.
            GC-only patterns will be removed anyways.
        eachBestNum:int (default=10000)
            Save this number of best pattern combinations for each pair of positions.
            Increasig this number does not change the complexity and runtime of algorithm,
            and changes only the HDD footprint and aggregation length.
            Setting this more than 100000 could cause errors during aggregation
            (and is probably impractical)

        """
        if self.patternLength > 7:
            raise ValueError("Pattern length too long. This will cause an out-of-memory error")

        chunkSize = 400
        bins = range(0, len(self.sequences), chunkSize) + [len(self.sequences)]
        bins = zip(bins[:-1], bins[1:])
        seqLen = len(self.sequences[0])
        nmerNum = self.patternNum
        nmerNum2 = nmerNum ** 2

        maxScore = maxPenalty + 1,

        pairs = [(i, j) for i in range(0, seqLen - 3) for j in range(i + 3, seqLen)]

        for i, j in pairs[startFrom::step]:
            "Main loop for a given pair [i,j]"

            allArrays = [np.zeros(nmerNum2, dtype=float) for s in range(maxScore)]
            for bin in bins:  # @ReservedAssignment
                seqs = self.sequences[bin[0]:bin[1]]
                positions = [np.array(seqs[:, s], order="C") for s in range(seqLen)]
                # print positions[i]
                # print positions[j]
                seqID = np.array(range(len(seqs)))

                allScores = []
                for o1 in range(-maxShift, maxShift + 1):
                    for o2 in range(-maxShift, maxShift + 1):
                        for m1 in range(maxSubs + 1):
                            for m2 in range(maxSubs + 1):
                                score = abs(o1 - o2) + min(abs(o1), abs(o2)) + m1 + m2
                                if np.abs(o1 - o2) > maxExtension:
                                    continue
                                if score < maxScore:
                                    cur = (i + o1, j + o2, m1, m2)
                                    if (cur[0] < 0) or (cur[1] < 0) or (cur[0] >= seqLen) or (cur[1] >= seqLen):
                                        continue
                                    allScores.append((cur, score))

                allResults = [[] for _ in xrange(maxScore)]
                subs = [self.sub0, self.sub1, self.sub2, self.sub3] [:maxSubs + 1]

                def scorePosition(positions1, positions2):
                    ind = positions1[:, :, None] + positions2[:, None, :] * nmerNum + seqID[:, None, None] * nmerNum2
                    return ind.flat

                for inds, score in allScores:
                    s1, s2, m1, m2 = inds
                    allResults[score].append(scorePosition(subs[m1][positions[s1]],
                                                           subs[m2][positions[s2]]))

                allResults = [np.unique(np.concatenate(s)) for s in allResults]
                for t in xrange(maxScore):
                    if t > 0:
                        subtract = np.unique(np.concatenate(allResults[:t]))
                        cur = np.setdiff1d(allResults[t], subtract, assume_unique=True)
                        # because we replace allResults[i] with cur, then
                        # overlap of allResults[0] and allResults[1] will be zero.
                        # So at any step we're concatenating non-overlapping datasets,
                        # and therefore we can set "assume_unique", which speeds up things
                        # quite a bit
                        allResults[t] = cur
                    else:
                        cur = allResults[t]
                    allArrays[t] += np.bincount(cur % nmerNum2, minlength=(nmerNum2))

            for t in xrange(maxScore):
                allArrays[t] = allArrays[t] / (2. ** t)
            combined = sum(allArrays)

            adjustByMean = True
            combined = np.reshape(combined, (nmerNum, nmerNum))
            self.deleteGCInPlace(combined)
            if adjustByGC:
                combined = self.adjustByGC(combined)
            combined = np.reshape(combined, (-1))
            args = np.argsort(combined)[-eachBestNum:]
            values = combined[args]
            for arg, value in zip(args[-20:], values[-20:]):
                print self.indexToPattern[arg % nmerNum], self.indexToPattern[arg / nmerNum], value
            # print args / nmerNum, args % nmerNum
            iall = np.zeros(len(args), dtype=np.int8) + i
            jall = np.zeros(len(args), dtype=np.int8) + j
            joblib.dump((iall, jall, args % self.patternNum, args / self.patternNum, values),
                        os.path.join(self.folder, "{0}_{1}".format(i, j)), compress=4)
            print "finished numbers", i, j


    def scorePatternsThroughBest(self, bestNum=30,
                                 maxScore=7, maxScoreSingle=3,
                                 maxShift=3, maxSubs=3,
                                 penaltySlope=2, coordinates=None):
        """
        Performs a scoring procedure when it first selects bestNum best motifs per location,
        and then looks at their combinations
        This method automatically adjust for GC

        Parameters
        ----------

        bestNum : int (optional)
            number of best motifs per locus to analyze.
            Time is simply quadratic with this number.
        startFrom, step: int(optional)
            Used to separate work between different processes.
            If you want to split the work, launch step processes with values
            of startFrom from 0 to step-1
        maxScore:int(default=7)
            Maximum total score for a pair
        maxScoreSingle:int(default=3)
            Maximum total score for selection of a single pattern.
        maxShift:int(default=3)
            Maximum shift and maximum extension of the pattern pair
            This includes maximum shift of each pattern component,
            so shift of the left pattern by 2 and extention by 2
            will count as 4 and will be rejected.
        maxSubs:int(default=3)
            Maximum number of substitutions to tolerate.
            Cannot be more than 3.
        penaltySlope(default=2.), int
            1/alpha, where alpha is described in the paper
        coordinates : None (default) or list of tuples
            Evaluate only a given set of coordinates (i,j) of the two 6-mers for a pattern
        """

        if bestNum > 200:
            raise ValueError("If you need to score all patterns, use different method")
        if maxSubs > 3:
            raise ValueError("More than 3 subs are not supported (and do you really need that?)")



        seqLen = len(self.sequences[0])
        nmerNum = self.patternNum

        subs = [self.sub0, self.sub1, self.sub2, self.sub3][:maxSubs + 1]

        seqs = np.array(self.sequences)
        seqID = np.array(range(len(seqs)))

        # a list of nmers at kth position in all upstream
        positions = [np.array(seqs[:, s], order="C") for s in range(seqLen)]


        def scoreSinglePosition(posOne):
            " a self-cached function to score patterns at one position"
            if posOne in self.singleCache:
                return self.singleCache[posOne]

            # a list of matches for each score
            allResults = [[] for _ in xrange(maxScoreSingle + 1)]


            # iterating over all offsets and mismatches
            for o1 in range(-maxShift, maxShift + 1):
                for m1 in range(len(subs)):
                    score = np.abs(o1) + m1
                    if score > maxScoreSingle:
                        continue
                    cur = posOne + o1
                    if (cur < 0) or (cur >= seqLen):
                        continue
                    # here we record an ID, which combines upstream and nmer information
                    IDs = subs[m1][positions[cur]] + seqID[:, None] * nmerNum
                    allResults[score].append(IDs.flat)

            allResults = [np.unique(np.concatenate(s)) for s in allResults]
            allArrays = []
            for t in xrange(maxScoreSingle + 1):
                if t > 0:
                    # we don't count all IDs which scored less than t
                    subtract = np.unique(np.concatenate(allResults[:t]))

                    # se only count those which did
                    cur = np.setdiff1d(allResults[t], subtract, assume_unique=True)
                    allResults[t] = cur
                else:
                    # perfect matches are just recorded
                    cur = allResults[t]

                # now the remainder is the n-mer number; that's why we used that form of ID defiend above
                allArrays.append(np.bincount(cur % nmerNum, minlength=(nmerNum)))

            # calculating the score with penalty slope
            for t in xrange(maxScoreSingle + 1):
                allArrays[t] = allArrays[t] / (float(penaltySlope) ** t)
            combined = sum(allArrays)

            combined, correction = self.adjustVectorByGC(combined)
            args = np.argsort(combined)[-bestNum:]
            values = combined[args]

            # prepare the return value
            toret = (args, values, correction[args])
            self.singleCache[posOne] = toret
            return toret

        if coordinates is None:
            pairs = [(i, j) for i in range(0, seqLen - 3) for j in range(i + 5, seqLen)]
        else:
            pairs = coordinates

        for i, j in pairs:
            if j < i:
                raise ValueError("We assume that left box position in the pattern is before the right box")
            def score(pat1, pat2):
                "a function to score a pair of patterns"
                scores = np.zeros(len(positions[i]), dtype=int) + 10000
                subs1 = [s[pat1] for s in subs]
                subs2 = [s[pat2] for s in subs]
                # reverse lookup for subs
                check1 = [np.zeros(self.patternNum, dtype=np.bool) for _ in subs]
                check2 = [np.zeros(self.patternNum, dtype=np.bool) for _ in subs]
                for k in xrange(len(subs)):
                    check1[k][subs1[k]] = True
                    check2[k][subs2[k]] = True

                for o1 in range(-maxShift, maxShift + 1):
                    for o2 in range(-maxShift, maxShift + 1):
                        for m1 in range(len(subs)):
                            for m2 in range(len(subs)):
                                if abs(o1 - o2) > maxShift:
                                    continue
                                curScore = abs(o1 - o2) + min(abs(o1), abs(o2)) + m1 + m2
                                if curScore <= maxScore:
                                    cur = (i + o1, j + o2, m1, m2)
                                    if (cur[0] < 0) or (cur[1] < 0) or (cur[0] >= seqLen) or (cur[1] >= seqLen):
                                        continue
                                    # finished selection process of which offsets and substitutions are allowed

                                    # Now relatively straightforward code to calculate scores:

                                    pos1 = positions[cur[0]]
                                    pos2 = positions[cur[1]]
                                    # performing lookup of upstream sequences
                                    # in the list of subs for a given pattern
                                    # only those which are in both are considered "hits"
                                    hits = check1[m1][pos1] * check2[m2][pos2]
                                    # updating scores for those which showed higher-than-yet-redorded score
                                    scores[hits] = np.minimum(scores[hits], curScore)
                # return scores[scores < 1000].sum()
                return (1. / (float(penaltySlope) ** scores[scores < 1000])).sum()

            best1 = zip(*scoreSinglePosition(i))
            best2 = zip(*scoreSinglePosition(j))

            "now saving lots of stuff"
            allData = []
            for p1, freq1, corr1 in best1:
                for p2, freq2, corr2 in best2:
                    myscore = score(p1, p2)
                    corrScore = myscore / (corr1 * corr2)
                    allData.append({"Pos l":i,
                                "Pos r":j,
                                "Ind l": p1,
                                "Ind r":p2,
                                "Freq corr l": freq1,
                                "Freq corr r": freq2,
                                "GC corr l": corr1,
                                "GC corr r": corr2,
                                "GC l": self.GC[p1],
                                "GC r": self.GC[p2],
                                "Score raw" : myscore,
                                "Score corr": corrScore,
                                "Patt l": self.indexToPattern[p1],
                                "Patt r": self.indexToPattern[p2]})

            # all = [np.array(s) for s in zip(*all)]
            myDataFrame = pd.DataFrame(allData)
            myDataFrame["Freq raw l"] = myDataFrame["Freq corr l"] * myDataFrame["GC corr l"]
            myDataFrame["Freq raw r"] = myDataFrame["Freq corr r"] * myDataFrame["GC corr r"]
            saveDir = os.path.join(self.folder, "saved")
            if not os.path.exists(saveDir):
                os.mkdir(saveDir)
            myDataFrame.to_pickle(os.path.join(saveDir, "{0}_{1}".format(i, j)))
            if not os.path.exists(os.path.join(self.folder, "CSVs")):
                os.mkdir(os.path.join(self.folder, "CSVs"))
            myDataFrame = myDataFrame.sort_values("Score corr", ascending=False)
            myDataFrame.to_csv(os.path.join(self.folder, "CSVs", "{0}_{1}.csv".format(i, j)))

            myDataFrame = myDataFrame[::len(myDataFrame) / 10]
            myDataFrame["ScoreProduct"] = myDataFrame["Freq corr l"] * myDataFrame["Freq corr r"]
            myDataFrame["ScoreBoth"] = myDataFrame["Score corr"]
            print "Working on positions: left={0}, right={1}".format(i, j)
            print "10 selected patterns, from best to worst, and their scores"
            print myDataFrame[["Patt l", "Patt r", "ScoreProduct", "ScoreBoth"]]

    def scoreSinglePattern(self, patPos, pattern, maxShift=3, maxSubs=3,
                            shiftPenalty=0.5, maxScore=5):
        "for pat1 at position pos1 finds promoters in which it scores"

        seqLen = len(self.sequences[0])
        nmerNum = self.patternNum
        nmerNum2 = nmerNum ** 2

        subs = [self.sub0, self.sub1, self.sub2, self.sub3][:maxSubs + 1]

        seqs = np.array(self.sequences)
        seqID = np.array(range(len(seqs)))
        positions = [np.array(seqs[:, s], order="C") for s in range(seqLen)]

        subs1 = [s[pattern] for s in subs]
        check1 = [np.zeros(self.patternNum, dtype=np.bool) for _ in subs]

        for k in xrange(len(subs)):
            check1[k][subs1[k]] = True
        allHits = np.zeros(len(self.sequences)) + 13
        offsets1 = np.zeros_like(allHits)
        for o1 in range(-maxShift, maxShift + 1):
            for m1 in range(len(subs)):
                score = shiftPenalty * np.abs(o1) + m1
                if score > maxScore:
                    continue
                cur = patPos + o1
                if (cur < 0) or (cur >= seqLen):
                    continue
                pos1 = positions[cur]
                hits = check1[m1][pos1]

                for hit in np.nonzero(hits)[0]:
                    if score < allHits[hit]:
                        allHits[hit] = score
                        offsets1[hit] = cur
        return allHits, offsets1



    def getPatternSeqs(self, pos1, pos2, patLeft, patRight,
                                 maxScore=7,
                                 maxShift=3, maxSubs=3, shiftPenalty=0.):
        """
        Copy of the previous function code
        """

        if pos1 > pos2:
            pos1, pos2 = pos2, pos1



        seqLen = len(self.sequences[0])
        nmerNum = self.patternNum
        nmerNum2 = nmerNum ** 2

        resultDict = {}
        subs = [self.sub0, self.sub1, self.sub2, self.sub3][:maxSubs + 1]

        seqs = np.array(self.sequences)
        seqID = np.array(range(len(seqs)))
        positions = [np.array(seqs[:, s], order="C") for s in range(seqLen)]



        pat1, pat2 = patLeft, patRight
        i, j = pos1, pos2


        seqs1 = []
        seqs2 = []

        scores = np.zeros(len(positions[i]), dtype=int) + 10000
        subs1 = [s[pat1] for s in subs]
        subs2 = [s[pat2] for s in subs]
        check1 = [np.zeros(self.patternNum, dtype=np.bool) for _ in subs]
        check2 = [np.zeros(self.patternNum, dtype=np.bool) for _ in subs]
        for k in xrange(len(subs)):
            check1[k][subs1[k]] = True
            check2[k][subs2[k]] = True

        allHits = np.zeros(len(self.sequences), dtype=int) + 13
        offsets1 = np.zeros_like(allHits, dtype=int)
        offsets2 = np.zeros_like(allHits, dtype=int)
        counters = np.zeros(len(allHits), dtype=int)

        for o1 in range(-maxShift, maxShift + 1):
            for o2 in range(-maxShift, maxShift + 1):
                for m1 in range(len(subs)):
                    for m2 in range(len(subs)):
                        if abs(o1 - o2) > maxShift:
                            continue
                        curScore = shiftPenalty * (abs(o1 - o2) + min(abs(o1), abs(o2))) + m1 + m2
                        if curScore <= maxScore:
                            cur = (i + o1, j + o2, m1, m2)
                            if (cur[0] < 0) or (cur[1] < 0) or (cur[0] >= seqLen) or (cur[1] >= seqLen):
                                continue

                            pos1 = positions[cur[0]]
                            pos2 = positions[cur[1]]
                            hits = check1[m1][pos1] * check2[m2][pos2]

                            c1, c2 = cur[0], cur[1]
                            for hit in np.nonzero(hits)[0]:
                                if curScore < allHits[hit]:
                                    allHits[hit] = curScore
                                    offsets1[hit] = c1
                                    offsets2[hit] = c2
                                    counters[hit] = 1
                                elif curScore == allHits[hit]:
                                    if np.random.random() < (1. / (counters[hit] + 1)):
                                        allHits[hit] = curScore
                                        offsets1[hit] = c1
                                        offsets2[hit] = c2
                                    counters[hit] += 1

                            scores[hits] = np.minimum(scores[hits], curScore)
        # return scores[scores < 1000].sum()

        return { "offsets1":offsets1, "offsets2":offsets2,
                 "hitArray":allHits}

        print set([len(i) for i in seqs1])
        print set([len(i) for i in seqs2])
        print len(seqs1), len(seqs2)
        seqs1 = np.array(seqs1)
        seqs2 = np.array(seqs2)
        print seqs1.shape, seqs2.shape
        seqs = np.concatenate([seqs1, seqs2], axis=1)

        print seqs.shape
        mat = np.array([np.sum(seqs == i, axis=0) / float(seqs.shape[0]) for i in range(4)])
        return mat



    def loadSequences(self, inSequences, minlen=18):
        """
        Loads sequences into the class.
        inSequences: str (filename), tuple or array
            If string, then interpreted as a filename to load

            If a list of arrays, or a list of strings, then the input is interpreted
            as raw sequences, or raw digital sequences

            If a length-two tuple, the input is interpreted as
            (digital sequences, n-mer sequences), where n-mer sequences are
            sequences encoded as an array of n-mers, not individual basepairs.
            This is needed for quick loading of data by other parts of this program

        minlen : int, optional
            When loading from file, ignore sequences less than minlen

        """
        if type(inSequences) == str:
            mySequences = [i for i in open(inSequences).readlines() if len(i) > 5]
            mySequences = [i.replace("\n", "").replace("\r", "") for i in mySequences]
            lenset = set([len(i) for i in mySequences])
            if not  len(lenset) == 1:
                mySequences = [i for i in mySequences if len(i) >= minlen]
                lenset = set([len(i) for i in mySequences])
                lenmin = min(lenset)
                mySequences = [i[-lenmin:] for i in mySequences]

            self.rawSequences = np.array([[{"A":0, "T":1, "G":2, "C":3}[i] for i in j] for j in mySequences])
        else:
            if len(inSequences) == 2:  # load everything at once
                self.rawSequences, self.sequences = inSequences
                return
            else:
                if type(inSequences[0]) == str:  # loading from a list of strings
                    mySequences = inSequences
                    self.rawSequences = np.array([[{"A":0, "T":1, "G":2, "C":3}[i] for i in j] for j in inSequences])
                elif issubclass(type(inSequences[0][0]), int):  # loading from a list of sequences
                    self.rawSequences = inSequences
                    mySequences = [translateRaw(i) for i in inSequences]
                    maxVal = max([np.asarray(i).max() for i in inSequences])
                    if maxVal >= len(self.alphabet):
                        raise ValueError("Maximum value in inSequences is {0}, more than the length of the alphabet {1}".format(maxVal, len(self.alphabet)))
                self.rawSequences = inSequences

        sequences = np.array([[self.patternToIndex[j[i:i + self.patternLength]] for i in xrange(len(j) - (self.patternLength - 1))] for j in mySequences])
        self.sequences = sequences

    def loadTestSequences(self, numSeq, percentEach=0.5):
        """Creates num test sequences which have slight enrichment of two motifs:
         TATATA at position 25 and GCGCGC at position 40
         TGTGTG at position 25 and GAGAGA at position 40

         Motifs are also in a slightly different basepair composition background
        """
        vals1 = {25:"T", 26:"A", 27:"G", 28:"T", 29:"C", 30:"G", 40:"G", 41:"A", 42:"C", 43:"T", 44:"C", 45:"G"}
        vals2 = {25:"C", 26:"A", 27:"G", 28:"C", 29:"G", 30:"A", 40:"C", 41:"T", 42:"T", 43:"G", 44:"G", 45:"A"}
        allSeqs = []
        for sn in xrange(numSeq):
            seq = []
            for num in xrange(60):
                if num in vals1:
                    if random.random() < percentEach:
                        if (sn % 2) == 0:
                            seq.append(vals1[num])
                        else:
                            seq.append(vals2[num])
                        continue
                if num > 35:
                    seq.append(random.choice(["A", "T", "G", "C"]))
                else:
                    seq.append(random.choice(["A", "T", "G", "C"]))
            allSeqs.append("".join(seq))
        testSequences = allSeqs
        self.sequences = np.array([[self.patternToIndex[j[i:i + 6]] for i in xrange(len(j) - 5)] for j in testSequences])










