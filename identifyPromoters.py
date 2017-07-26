import sys, os
from getScores import PatternFinder  # @UnresolvedImport
import joblib
import numpy as np
# matplotlib.use("WebAgg")
import pandas as pd
from copy import copy



def translate(word):
    return [{"A":0, "T":1, "G":2, "C":3}[i] for i in word]

def translateRaw(array):
    return "".join([{0:"A", 1:"T", 2:"G", 3:"C"}[i] for i in array])


scoreFolder = sys.argv[1]
upstreamList = sys.argv[2]
fullUpstreamList = sys.argv[3]


if not os.path.exists("temp"):
    os.mkdir("temp")
current = PatternFinder(6, "temp")
current.loadSequences(upstreamList)

full = PatternFinder(6, scoreFolder)
full.loadSequences(fullUpstreamList)

temp = PatternFinder(6, scoreFolder)



def makeTextShuffleControl(filename, minLen=10000):
    cdata = np.array([translate(i.replace("\n", "").replace("\r", "")) for i in open(filename).readlines() if len(i) > 10]).T

    controls = []
    repeat = 1 + minLen / len(cdata[0])

    for _ in xrange(repeat):
        control = []
        for i in cdata:
            args = np.argsort(np.random.random(len(i)))
            control.append(i[args])
        control = np.array(control, order="C")
        controls.append(control)

    values = np.concatenate(controls, axis=1).T
    assert len(values[0]) == len(cdata)
    strings = map(translateRaw, values)
    open(filename + "_control_text", 'w').write("\n".join(strings))


if not os.path.exists(fullUpstreamList + "_control_text"):
    makeTextShuffleControl(fullUpstreamList, 40000)

control = PatternFinder(6, "temp")
control.loadSequences(fullUpstreamList + "_control_text")


fullSeqs = copy(full.rawSequences)



data = pd.read_csv(os.path.join(scoreFolder, "sortedBy/Best1000_sortBy_ScoreNew_8.csv"))

data = data[:300]

allPats = []
for i in data.index.values:
    cur = data.ix[i]
    pat1 = cur["Ind l"]
    pat2 = cur["Ind r"]
    loc1 = cur["Pos l"]
    loc2 = cur["Pos r"]
    allPats.append((pat1, pat2, loc1, loc2))



def buildHitMatrix(sequences, patterns):
    temp.loadSequences(sequences)
    def getHits(i):
        pat1, pat2, loc1, loc2 = i
        currentHit = temp.getPatternSeqs(loc1, loc2, pat1, pat2,
                                 maxScore=6,  # penalty should be less than that, i.e. 0... 5
                                 maxShift=3,
                                 shiftPenalty=1,
                                 maxSubs=3)
        allhits, c1, c2 = [currentHit[i] for i in ["hitArray", "offsets1", "offsets2"]]
        return allhits, c1, c2,

    hitMatrix = map(getHits, patterns)
    hitMatrix = zip(*hitMatrix)
    hm, c1, c2 = map(np.array, hitMatrix)
    print hm.shape
    return hm, c1, c2
mem = joblib.Memory(".")
buildHitMatrix = mem.cache(buildHitMatrix)


hm, c1, c2 = buildHitMatrix(full.rawSequences, allPats)
hm[hm == 13] = hm[hm != 13].max() + 1
hm = hm.max() - hm
hm = hm * 1.
seqs = full.sequences
selScores = ((np.mean(hm ** 5, axis=0)) ** 0.2)

hmc, c1c, c2c = buildHitMatrix(control.rawSequences, allPats)
hmc[hmc == 13] = hmc[hmc != 13].max() + 1
hmc = hmc.max() - hmc
hmc = hmc * 1.
contScores = ((np.mean(hmc ** 5, axis=0)) ** 0.2)
contScores = np.sort(contScores)

rank = np.searchsorted(contScores, selScores)
pvalue = 1 - rank / float(len(contScores))

df = pd.DataFrame({"pvalue":pvalue, "score":selScores}, index=np.arange(len(pvalue), dtype=int))


sh = full.rawSequences.shape
importance = np.zeros((sh[0], sh[1], len(hm)), dtype=float)
importanceCount = np.zeros_like(full.rawSequences, dtype=int)

def countFrequences(sequences, args, starts, ends):

    countsLeft = []
    countsRight = []

    args = args[ends[args] + 7 < 60]

    for offset in range(-1, 7):
        seqsLeft = sequences[args, starts[args] + offset]
        countsLeft.append(np.bincount(seqsLeft, minlength=4) + 1)

        seqsRight = sequences[args, ends[args] + offset]
        countsRight.append(np.bincount(seqsRight, minlength=4) + 1)
    countsLeft = np.array(countsLeft, dtype=float)
    countsRight = np.array(countsRight, dtype=float)
    return countsLeft, countsRight

c1 = np.array(c1, dtype=int)
c2 = np.array(c2, dtype=int)

mySeqs = current.rawSequences
mySeqs = set([tuple(i) for i in mySeqs])
mask = np.array([tuple(i) in mySeqs for i in full.rawSequences])


for ind in xrange(len(hm)):



    currentHit = hm[ind]

    starts = c1[ind]
    ends = c2[ind]

    args = np.nonzero(currentHit > 1.5)[0]

    args = args[ends[args] + 7 < 60]
    args = args[starts[args] - 1 > 0 ]

    newargs = args[mask[args] == True]

    left, right = countFrequences(full.rawSequences, newargs, starts, ends)

    controlSeqs = full.rawSequences  # all sequences

    repeat = len(controlSeqs) / len(starts[args]) + 1
    newstarts = np.concatenate([starts[args]] * repeat)  # repeat selected starts
    newends = np.concatenate([ends[args]] * repeat)  # repeat selected endds
    shuffleArgs = np.argsort(np.random.random(len(controlSeqs)))
    shuffleArgs2 = np.argsort(np.random.random(len(controlSeqs)))
    newstarts = newstarts[shuffleArgs]
    newends = newends[shuffleArgs][shuffleArgs]
    leftC, rightC = countFrequences(controlSeqs, shuffleArgs2,
                                     newstarts, newends)

    left /= leftC
    right /= rightC
    left /= np.sum(left, axis=1)[:, None]
    right /= np.sum(right, axis=1)[:, None]

    importanceLeft = np.sum(left * np.log(left / 0.25), axis=1)
    importanceRight = np.sum(right * np.log(right / 0.25), axis=1)

    for offset in range(-1, 7):
        countL = importanceCount[args, starts[args] + offset]
        countR = importanceCount[args, ends[args] + offset]

        importance[args, starts[args] + offset, countL] = importanceLeft[offset + 1]
        importance[args, ends[args] + offset, countR] = importanceRight[offset + 1]

        importanceCount[args, starts[args] + offset] += 1
        importanceCount[args, ends[args] + offset] += 1


importance[(importanceCount < 20)[:, :, None]] = 0
importance = np.sort(importance, axis=2)
importance = np.mean(importance[:, :, -10:], axis=2)
seqSet = set([translateRaw(i) for i in current.rawSequences])

for impCutoff in [0.3, 0.45, 0.6]:
    select = importance > impCutoff
    myseqs = []

    lef1 = []
    lef2 = []
    rig1 = []
    rig2 = []

    inCurrent = []
    for seq, inds in zip(full.rawSequences, select):
        seq = translateRaw(seq)
        if seq in seqSet:
            inCurrent.append(True)
        else:
            inCurrent.append(False)
        if sum(inds) == 0:
            myseqs.append(seq.lower())
            lef1.append(-1)
            lef2.append(-1)
            rig1.append(-1)
            rig2.append(-1)
        else:
            chars = list(seq.lower())
            whereInds = np.nonzero(inds)[0]
            for i in whereInds:
                chars[i] = chars[i].upper()
            myseqs.append("".join(chars))

            if (whereInds < 40).sum() == 0:
                lef1.append(-1)
                rig1.append(-1)
            else:
                lef1.append(whereInds.min())
                rig1.append(whereInds[whereInds < 40].max())

            if (whereInds > 40).sum() == 0:
                lef2.append(-1)
                rig2.append(-1)
            else:
                lef2.append(whereInds[whereInds > 40].min())
                rig2.append(whereInds.max())




    counter = 0
    for i in np.argsort(selScores)[-500:]:
        seq = myseqs[i]

        if seq != seq.lower():
            print seq
            counter += 1

    df["cutoff={0}_sequence".format(impCutoff)] = myseqs
    df["{0}_leftBosMin".format(impCutoff)] = lef1
    df["{0}_leftBoxMax".format(impCutoff)] = rig1
    df["{0}_rightBoxMin".format(impCutoff)] = lef2
    df["{0}_rightBoxMax".format(impCutoff)] = rig2

df.insert(2, "In current set", inCurrent)
df = df.sort_values(["score"], ascending=False)
df.to_csv(os.path.join(scoreFolder, "promoterTable.csv"))

print counter


