from getScores import PatternFinder  # @UnresolvedImport
import joblib
import sys, os
import numpy as np
import pandas as pd
import cPickle
import warnings
import scipy.sparse.linalg

try:
    sequenceFile = sys.argv[1]
    scoreFolder = sys.argv[2]
    POSITION_LEFT = int(sys.argv[3])
    POSITION_RIGHT = int(sys.argv[4])
except:
    print "Usage: python makeLogo.py sequenceFile  scoresFolder position_left position_right"
    exit()



def PCA(A, numPCs=6, verbose=False):
    """performs PCA analysis, and returns 6 best principal components
    result[0] is the first PC, etc"""
    A = np.array(A, float)
    if np.sum(np.sum(A, axis=0) == 0) > 0 :
        warnings.warn("Columns with zero sum detected. Use zeroPCA instead")
    M = (A - np.mean(A.T, axis=1)).T
    covM = np.dot(M, M.T)
    [latent, coeff] = scipy.sparse.linalg.eigsh(covM, numPCs)
    if verbose:
        print "Eigenvalues are:", latent
    return (np.transpose(coeff[:, ::-1]), latent[::-1])



def translate(word):
    return [{"A":0, "T":1, "G":2, "C":3}[i] for i in word]

def translateRaw(array):
    return "".join([{0:"A", 1:"T", 2:"G", 3:"C"}[i] for i in array])



a = PatternFinder(6, scoreFolder)
a.loadSequences(sequenceFile)
allSeqs = a.rawSequences

data = pd.read_csv(os.path.join(scoreFolder, "sortedBy/Best10000_sortBy_ScoreNew_8.csv"))

lef = data["Pos l"].values
rig = data["Pos r"].values
mask = (abs(rig - POSITION_RIGHT) < 4) * (abs(lef - POSITION_LEFT) < 4)
data = data[mask]
patLeftBest = data["Patt l"].values[0]
patRightBest = data["Patt r"].values[0]

data = data[:300]
assert len(data) > 90  # check that we have at least 90 unique patterns

allPats = []
for i in data.index.values:
    cur = data.ix[i]
    pat1 = cur["Ind l"]
    pat2 = cur["Ind r"]
    allPats.append((pat1, pat2))
allPats = sorted(list(set(allPats)))




def buildHitMatrix(dummy1, dummy2):
    def getHits(i):
        pat1, pat2 = i

        currentHit = a.getPatternSeqs(POSITION_LEFT, POSITION_RIGHT, pat1, pat2,
                                 maxScore=5,  # penalty should be less than that, i.e. 0... 5
                                 maxShift=3,
                                 shiftPenalty=0.,
                                 maxSubs=3)
        allhits, dummy1, dummy2 = [currentHit[i] for i in ["hitArray", "offsets1", "offsets2"]]
        return allhits
    hitMatrix = map(getHits, allPats)
    return np.array(hitMatrix)

mem = joblib.Memory(".")
buildHitMatrix = mem.cache(buildHitMatrix)

hm = buildHitMatrix(a.sequences, allPats)
lef = data["Pos r"].values
rig = data["Pos l"].values
hm[hm == 13] = hm[hm != 13].max() + 1
hm = np.array(hm.max() - hm, dtype=float)

selScores = ((np.mean(hm ** 10, axis=0)) ** 0.1)
args = np.argsort(selScores)

if len(allSeqs) < 200:
    select = allSeqs
else:
    select = [allSeqs[i] for i in args[-200:]]

seqFolder = os.path.join(scoreFolder, "savedSeqs")
patFolder = os.path.join(scoreFolder, "savedPats")
for fol in [seqFolder, patFolder]:
    if not os.path.exists(fol):
        os.mkdir(fol)

cPickle.dump((select, allSeqs), open(os.path.join(seqFolder, "TOTAL"), 'wb'), -1)
cPickle.dump((patLeftBest, patRightBest), open(os.path.join(patFolder, "TOTAL"), 'wb'), -1)

if len(selScores) > 600:
    mask = selScores > np.percentile(selScores, 66)
elif len(selScores) > 200:
    mask = selScores > np.sort(selScores)[-200]
else:
    mask = selScores > -999999999

hm = hm[:, mask]
seqs = [allSeqs[i] for i in range(len(allSeqs)) if mask[i]]
assert len(seqs) == hm.shape[1]


PCs = PCA(hm, 40)[0]
PCs = np.array(PCs)

for ss in xrange(2):
    curPC = PCs[ss]
    if np.corrcoef(curPC, range(len(curPC)))[0, 1] < 0:
        curPC *= -1

    args = np.argsort(curPC)
    targs = np.argsort(np.dot(hm, curPC))
    beg = targs[0]
    end = targs[-1]

    p1l, p1r = [a.indexToPattern[i] for i in allPats[beg]]
    p2l, p2r = [a.indexToPattern[i] for i in allPats[end]]
    print p1l, p1r, p2l, p2r

    first = 200
    last = 200
    if len(seqs) < 350:
        first = len(seqs) / 2
        last = len(seqs) / 2

    s1 = [seqs[i] for i in args[:first]]
    s2 = [seqs[i] for i in args[-last:]]

    cPickle.dump((s1, allSeqs), open(os.path.join(seqFolder, "{0}_{1}".format("PC", 2 * ss)), 'wb'), -1)
    cPickle.dump((s2, allSeqs), open(os.path.join(seqFolder, "{0}_{1}".format("PC", 2 * ss + 1)), 'wb'), -1)

    cPickle.dump((p1l, p1r), open(os.path.join(patFolder, "{0}_{1}".format("PC", 2 * ss)), 'wb'), -1)
    cPickle.dump((p2l, p2r), open(os.path.join(patFolder, "{0}_{1}".format("PC", 2 * ss + 1)), 'wb'), -1)


