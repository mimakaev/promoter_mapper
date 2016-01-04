# (c) 2015 Massachusetts Institute of Technology. All Rights Reserved
# Code written by Maxim Imakaev <imakaev@mit.edu>

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pandas as pd
import cPickle

np.set_printoptions(precision=4)

print "Usage: python scorePWM.py scoreFolder genomeGC(100-based) fullUpstreamFilename positionLeft positionRight"

scoreFolder = sys.argv[1]
GENOME_GC = float(sys.argv[2]) / 100.
assert 0 < GENOME_GC < 1
fullUpstreamList = sys.argv[3]
POSITION_LEFT = int(sys.argv[4])
POSITION_RIGHT = int(sys.argv[5])
name = os.path.split(scoreFolder)[-1]

def rank(x):
    "Returns rank of an array"
    tmp = np.argsort(x)
    return np.array(np.arange(len(x)), float).take(tmp.argsort())


def printlogo(pwm, filename, alphabet="ACGT", mode="pdf"):
    myAlphabet = {"A":0, "C":1, "G":2, "T":3}
    translate = np.array([myAlphabet[i] for i in alphabet])
    pwm = pwm[:, translate]
    "Prints logo from nucleotides as a pdf"
    import cPickle
    cPickle.dump(pwm, open(filename + ".pkl", 'wb'), -1)
    import weblogolib as wl  # @UnresolvedImport
    PWMdata = np.array(pwm)
    data = wl.LogoData.from_counts(wl.std_alphabets["dna"], PWMdata)
    options = wl.LogoOptions(resolution=300)
    options.title = filename
    options.color_scheme = wl.colorscheme.nucleotide
    formatt = wl.LogoFormat(data, options)
    if mode == "pdf":
        fout = open(filename + ".pdf", 'wb')
        fout.write(wl.pdf_formatter(data, formatt))
    elif mode == "png":
        fout = open(filename + ".png", 'wb')
        fout.write(wl.png_formatter(data, formatt))
    else:
        fout = open(filename + ".{0}".format(mode), 'wb')
        exec("""fout.write(wl.{0}_formatter(data, format))""".format(mode))

    fout.close()



def computeFrequencies(sequences, weights=None):
    """ a helper function to compute frequencies of nucleotides in a list of (binary) sequences"""
    freqs = []
    for i in [0, 1, 2, 3]:
        if weights is None:
            freqs.append(np.sum(np.array(sequences) == i, axis=0))
        else:
            freqs.append(np.sum((np.array(sequences) == i) * weights[:, None], axis=0))
    freqs = np.vstack(freqs).T + 1
    GC = 2 * GENOME_GC
    # return freqs
    freqs[:, 0] /= (2 - GC)
    freqs[:, 1] /= (2 - GC)
    freqs[:, 2] /= GC
    freqs[:, 3] /= GC
    freqs /= np.sum(freqs, axis=1)[:, None]
    return freqs


def loadSequences(filename):
    sequences = [i[:60] for i in open(filename).readlines() if len(i) > 5]
    rawData = sequences
    numbers = np.array([[{"A":0, "T":1, "G":2, "C":3}[i] for i in j] for j in sequences])
    positions = np.array(numbers.T, order="C")
    return positions


def translate(word):
    return [{"A":0, "T":1, "G":2, "C":3}[i] for i in word]


def translateRaw(array, translate=np.array(["A", "T", "G", "C"])):
    return translate[array].tostring()


class PWM(object):
    def __init__(self, n, pos1, pos2, maxShift, maxExt):
        """
        Initializes a PWM searcher
        Parameters
        ----------
        n : int
            length of each box of PWM
        pos1, pos2: int
            positions of starts of each box in a sequence
        """
        self.pos1 = pos1
        self.pos2 = pos2
        self.maxShift = maxShift
        self.maxExt = maxExt
        self.n = n
        self.energy1 = np.zeros((self.n, 4), float)
        self.energy2 = np.zeros((self.n, 4), float)
        self.shiftEnergies = np.zeros(maxShift, float)
        self.extEnergies = np.zeros(maxExt, float)
        self.tail = self.maxExt + self.maxShift - 2

    def save(self):
        """
        Returns values of PWM concatenated in an array
        Does not save stuff which you pass the constructor
        """
        allValues = []
        for i in [1, 2, 3]:
            allValues.append(self.energy1[:, i].flat)
        for i in [1, 2, 3]:
            allValues.append(self.energy2[:, i].flat)
        allValues.append(self.shiftEnergies[1:])
        allValues.append(self.extEnergies[1:])
        return np.concatenate(allValues)

    def load(self, values):
        "Re-initializes the PWM from saved values. You still need the constrtuctor."
        cur = 0
        for i in [1, 2, 3]:
            self.energy1[:, i] = values[cur:cur + self.n]
            cur += self.n
        for i in [1, 2, 3]:
            self.energy2[:, i] = values[cur:cur + self.n]
            cur += self.n
        self.shiftEnergies[1:] = values[cur:cur + len(self.shiftEnergies) - 1]
        cur += len(self.shiftEnergies) - 1
        assert len(values[cur:]) == len(self.extEnergies) - 1
        self.extEnergies[1:] = values[cur:]

        if np.random.random() < 0.005:
            self.probabilities()

    def probabilities(self):
        "Computes probabilities from energies; returns all PWM data"
        ens1 = self.energy1 - np.max(self.energy1, axis=1)[:, None]
        pe1 = np.exp(ens1)
        pe1 /= np.sum(pe1, axis=1)[:, None]

        ens2 = self.energy2 - np.max(self.energy2, axis=1)[:, None]
        pe2 = np.exp(ens2)
        pe2 /= np.sum(pe2, axis=1)[:, None]

        sens = self.shiftEnergies - np.max(self.shiftEnergies)
        pshift = np.exp(sens)
        pshift /= np.sum(pshift)

        extens = self.extEnergies - np.max(self.extEnergies)
        pext = np.exp(extens)
        pext /= np.sum(pext)

        tosave = {}
        tosave["StartLeft"] = self.pos1
        tosave["StartRight"] = self.pos2
        tosave["MaxShift"] = self.maxShift
        tosave["MaxExt"] = self.maxExt
        tosave["FormulaLeft"] = "StartLeft + shift"
        tosave["FormulaRight"] = "StartRight + shift + ext"
        tosave["ProbsATGCLeft"] = pe1
        tosave["ProbsATGCRight"] = pe2
        tosave["ProbsExt"] = pext
        tosave["ProbsShift"] = pshift
        return tosave


    def fromProbabilities(self, p1, p2, pshift, pext):
        "Initializes PWM from probabilities of nucleotides, shifts, extensions"
        e1 = np.log(p1)
        e1 -= e1[:, 0][:, None]
        self.energy1 = e1

        e2 = np.log(p2)
        e2 -= e2[:, 0][:, None]
        self.energy2 = e2

        eshift = np.log(pshift)
        eshift -= eshift[0]
        self.shiftEnergies = eshift

        eext = np.log(pext)
        eext -= eext[0]
        self.extEnergies = eext



    def fromLogo(self, logo1, logo2, startLeft="auto", startRight="auto"):
        "Inits PWM from logo"

        if startLeft == "auto":
            before1 = (self.n - len(logo1)) / 2
        else:
            before1 = startLeft
        if startRight == "auto":
            before2 = (self.n - len(logo2)) / 2
        else:
            before2 = startRight
        freqs1 = np.zeros((self.n, 4)) + 1
        freqs2 = np.zeros((self.n, 4)) + 1
        for ind, letter in enumerate(logo1):
            freqs1[before1 + ind, letter] = 15
        for ind, letter in enumerate(logo2):
            freqs2[before2 + ind, letter] = 15
        pshift = np.ones(self.maxShift)
        pext = np.ones(self.maxExt)
        self.fromProbabilities(p1=freqs1, p2=freqs2,
                                pshift=pshift, pext=pext)


    def scoreSequences(self, positions, returnPositions=False, dictToUpdate={}, returnSingle=False):
        """
        Score a PWM in an array of sequences
        Parameters
        ----------
        positions : list of numpy arrays
            sequences to score a PWM in
        returnPositions : bool (optional)
            return positions at which PWM scored, and values of shift/extension
        dictToUpdate : dict (optional)
            Provide a dict in which to write information about scores of left and right box.
        returnSingle : bool (optional)
            If set to True, will write score of left/right box to dictToUpdate

        Returns
        -------
        a tuple (scores, pos1, pos2, shift,ext) of numpy arrays
            It encodes scores of the best match, position of left/right boxes, shift and extension

        """
        pos1, pos2 = self.pos1, self.pos2
        N = len(positions[0])
        M = len(positions)
        allScores = []
        shiftExts = []
        if returnSingle:
            scoresLeft = []
            scoresRight = []
        for shift in xrange(self.maxShift):
            for ext in xrange(self.maxExt):
                scores = np.zeros(N, dtype=float)
                if returnSingle:
                    sl = np.zeros(N, dtype=float)
                    sr = np.zeros(N, dtype=float)
                lastNuc = pos2 + shift + ext + self.n
                if lastNuc >= M:
                    # print "continuing at shift {0}, ext {1}, pos2 {2}".format(shift, ext, pos2)
                    continue
                for j in xrange(self.n):
                    probs = np.array(self.energy1[j], order="C")
                    scores += probs[positions[pos1 + shift + j]]
                    if returnSingle:
                        sl += probs[positions[pos1 + shift + j]]

                    probs = np.array(self.energy2[j], order="C")
                    scores += probs[positions[pos2 + shift + ext + j]]
                    if returnSingle:
                        sr += (probs[positions[pos2 + shift + ext + j]])


                scores += (self.shiftEnergies[shift] + self.extEnergies[ext])
                allScores.append(scores)
                if returnSingle:
                    scoresLeft.append(sl)
                    scoresRight.append(sr)
                shiftExts.append((shift, ext))
        allScores = np.array(allScores)
        args = np.argmax(allScores, axis=0)
        allScores = np.max(allScores, axis=0)
        if returnSingle:
            scoresLeft = np.max(scoresLeft, axis=0)
            scoresRight = np.max(scoresRight, axis=0)
            dictToUpdate["ScoresLeft"] = scoresLeft
            dictToUpdate["ScoresRight"] = scoresRight
        assert len(allScores) == len(positions[0])
        if returnPositions == True:
            shiftExts = np.array(shiftExts, dtype=int)
            part1 = shiftExts[:, 0] + pos1
            part2 = shiftExts[:, 1] + pos2 + shiftExts[:, 0]
            shift, ext = shiftExts[:, 0], shiftExts[:, 1]
            return allScores, part1[args], part2[args], shift[args], ext[args]
        return allScores

    def makeShuffleControl(self, cdata, minLen=10000):
        controls = []
        repeat = 1 + minLen / len(cdata[0])

        for _ in xrange(repeat):
            control = []
            for i in cdata:
                args = np.argsort(np.random.random(len(i)))
                control.append(i[args])
            control = np.array(control, order="C")
            controls.append(control)

        self.control = np.concatenate(controls, axis=1)



    def iterativeScoring2(self, data, value=None, save=False):
        """If save=False, then it scores PWM in sequences (data). Then it gets
        all the hits (which are in some locations in the sequences), gets best hit for each
        sequence, and calculates frequencies of occurrences at all positions (weighted,
        as described in the paper).

        It then re-calculates the PWM and returns the saved PWM (as well as changes internal state).

        If value != None, then the PWM is started from the value, else it is started from the internal state.

        if save=True, it evaluates sequences in a PWM, in a control sequences, and creates a table.
        In the table, it highlights promoters using three cutoffs for the information content of a PWM.
        """

        if value is not None:
            self.load(value)

        scoreDict = {}
        scoreMy, pos1, pos2, shifts, exts = self.scoreSequences(data, True,
                                        dictToUpdate=scoreDict, returnSingle=True)
        sequences = np.array(data.T)

        if save:
            "This part only runs when we need to build a final table"
            seqs1, seqs2 = [], []
            n = self.n
            rawSequences = [translateRaw(i) for i in sequences]
            impL = self.importanceLeft
            impR = self.importanceRight
            L1 = np.nonzero(impL > 0.8)[0]
            L2 = np.nonzero(impL > 0.5)[0]
            L3 = np.nonzero(impL > 0.2)[0]
            R1 = np.nonzero(impR > 0.8)[0]
            R2 = np.nonzero(impR > 0.5)[0]
            R3 = np.nonzero(impR > 0.2)[0]

            seqFirst = []
            seqSecond = []
            seqThird = []

            for p1, p2, sequence in zip(pos1, pos2, rawSequences):
                seqs1.append(sequence[p1:p1 + n])
                seqs2.append(sequence[p2:p2 + n])
                chars = list(sequence.lower())

                for LL, RR, SS in zip([L1, L2, L3], [R1, R2, R3], [seqFirst, seqSecond, seqThird]):
                    for i in LL:
                        chars[p1 + i] = chars[p1 + i].upper()
                    for i in RR:
                        chars[p2 + i] = chars[p2 + i].upper()
                    SS.append("".join(chars))

            resultDict = {"sequences":rawSequences, "scores":scoreMy,
                                 "firstBoxPos":pos1, "secondBoxPos":pos2,
                                 "firstBoxSeq":seqs1, "secondBoxSeq":seqs2,
                                 "consensus0.8":seqFirst, "consensus0.5":seqSecond,
                                 "consensus0.2":seqThird
                                 }
            resultDict.update(scoreDict)
            myframe = pd.DataFrame(resultDict, index=range(len(rawSequences)))

            scores = self.probabilities()
            myframe["ConsensusLeft"] = translateRaw(np.argmax(scores["ProbsATGCLeft"], axis=1))
            myframe["ConsensusRight"] = translateRaw(np.argmax(scores["ProbsATGCRight"], axis=1))
            myframe = myframe.sort(["scores"], ascending=False)
            return myframe

        scoreMy -= scoreMy.min()
        seqs1, seqs2 = [], []
        n = self.n

        for p1, p2, sequence in zip(pos1, pos2, sequences):
            seqs1.append(sequence[p1:p1 + n])
            seqs2.append(sequence[p2:p2 + n])

        weights = (scoreMy)
        freqs1 = computeFrequencies(seqs1, weights=weights)
        freqs2 = computeFrequencies(seqs2, weights=weights)

        left = np.array(freqs1)
        right = np.array(freqs2)
        left /= np.sum(left, axis=1)[:, None]
        right /= np.sum(right, axis=1)[:, None]

        importanceLeft = np.sum(left * np.log(left / 0.25), axis=1)
        importanceRight = np.sum(right * np.log(right / 0.25), axis=1)
        self.importanceLeft = importanceLeft
        self.importanceRight = importanceRight

        shiftFreqs = np.bincount(shifts, minlength=self.maxShift, weights=weights) + 1
        extFreqs = np.bincount(exts, minlength=self.maxExt, weights=weights) + 1

        self.fromProbabilities(freqs1, freqs2,
                                 pshift=shiftFreqs,
                                 pext=extFreqs)
        if save:
            return myframe

        return self.save()


def scoreFilename(data, saveTo, pats, tableSaveTo):
    """Actual code to score one set of sequences
    Parameters
    ----------

    data : str
        A filename with pickled sequences and control sequences
    saveTo : str
        file to save logo
    pats : str
        Filename with patterns
    tableSaveTo:
        file to save resulting table


    """

    # we will have different parameters of PWM builder for sigma-70 and sigma-54


    # loading saved best pattern
    if os.path.exists(pats):
        pat1, pat2 = cPickle.load(open(pats))

    data, allData = cPickle.load(open(data))

    # loading data for all upstreams
    totalData = loadSequences(fullUpstreamList)

    data = np.array(np.array(data).T, order="C")
    allData = np.array(np.array(allData).T, order="C")

    # initializing PWM for regular sequences and for NO
    myPWM = PWM(8, pos1=POSITION_LEFT, pos2=POSITION_RIGHT, maxShift=6, maxExt=6)
    myPWM.fromLogo(translate(pat1), translate(pat2))

    start = myPWM.pos1
    start2 = myPWM.pos2 - 1

    cur = myPWM.save()

    # actually doing iterative scoring
    for _ in xrange(10):
        cur = myPWM.iterativeScoring2(data, cur)
    n = len(data[0])

    # Calculating PWM for logo
    probs = myPWM.probabilities()
    mv = np.mean(probs["ProbsATGCLeft"])
    pwmdata = np.concatenate([probs["ProbsATGCLeft"], mv * np.ones((2, 4)), probs["ProbsATGCRight"]])

    # Now scoring all data
    singleDict = {}
    table = myPWM.iterativeScoring2(allData, save=True)


    # Loading all-sequence control
    if not os.path.exists(fullUpstreamList + "_control"):
        myPWM.makeShuffleControl(totalData, minLen=40000)
        cPickle.dump(myPWM.control, open(fullUpstreamList + "_control", 'wb'), -1)
    myPWM.control = cPickle.load(open(fullUpstreamList + "_control", 'rb'))

    tableTotal = myPWM.iterativeScoring2(totalData, cur, save=True)
    tableControl = myPWM.iterativeScoring2(myPWM.control, cur, save=True)

    # saving histograms of scores for our data and for control data
    cutoff = np.percentile(tableControl["scores"], 0)
    tableTotal = tableTotal.ix[tableTotal["scores"] > cutoff]

    rank = np.searchsorted(np.sort(tableControl["scores"].values), tableTotal["scores"].values)
    rankControl = np.searchsorted(np.sort(tableControl["scores"].values), tableControl["scores"].values)
    tableTotal["pvalue"] = 1 - rank / float(len(tableControl["scores"]))
    tableControl["pvalue"] = 1 - rankControl / float(len(tableControl["scores"]))

    # Now calculating goodness
    myseqs = [translateRaw(i) for i in data.T]
    bestseqsTotal = tableTotal["sequences"][: n]
    bestseqs = table["sequences"][: n]

    s1 = set(myseqs)
    s2 = set(bestseqsTotal)
    s3 = s2.intersection(s1)

    tableTotal["In current set"] = tableTotal["sequences"].isin(set(table["sequences"]))
    table["In training set"] = table["sequences"].isin(s1)
    tableTotal["In training set"] = tableTotal["sequences"].isin(s1)
    gdns = (len(s3)) / float(len(s2))

    # Now saving all stuff to file
    printlogo(pwmdata, saveTo + "_{pat1}_{pat2}_num={n}_goodness={gdns:.2f}".format(**locals()), alphabet="ATGC", mode="png")
    tableTotal.to_csv(tableSaveTo + "_{pat1}_{pat2}_num={n}_goodness={gdns:.2f}.csv".format(**locals()))
    tableControl.to_csv(tableSaveTo + "_{pat1}_{pat2}_num={n}_goodness={gdns:.2f}.csv_control".format(**locals()))
    return gdns



def score1(i):
    "a wrapper to score one file with all the folders provided"
    scoreFilename(os.path.join(os.path.join(scoreFolder, "savedSeqs"), i),
                  os.path.join(os.path.join(scoreFolder, "savedLogos"), i),
                  os.path.join(os.path.join(scoreFolder, "savedPats"), i),
                  os.path.join(os.path.join(scoreFolder, "savedTables"), i))


for fol in ["savedLogos", "savedTables"]:
    if not os.path.exists(os.path.join(scoreFolder, fol)):
        os.mkdir(os.path.join(scoreFolder, fol))
map(score1, os.listdir(os.path.join(scoreFolder, "savedSeqs")))


