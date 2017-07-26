# (c) 2015 Massachusetts Institute of Technology. All Rights Reserved
# Code written by Maxim Imakaev <imakaev@mit.edu>

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, sys
import pandas as pd
from getScores import nicePlot, cmap_map
from scipy.stats.stats import spearmanr


def listToColormap(colorList, cmapName=None):
    colorList = np.array(colorList)
    if colorList.min() < 0:
        raise ValueError("Colors should be 0 to 1, or 0 to 255")
    if colorList.max() > 1.:
        if colorList.max() > 255:
            raise ValueError("Colors should be 0 to 1 or 0 to 255")
        else:
            colorList = colorList / 255.
    return matplotlib.colors.LinearSegmentedColormap.from_list(cmapName, colorList, 256)


nMethList = ((236, 250, 255), (148, 189, 217),
             (118, 169, 68), (131, 111, 43), (122, 47, 25),
             (41, 0, 20))


def registerList(mylist, name):
    mymap = listToColormap(mylist, name)
    mymapR = listToColormap(mylist[::-1], name + "_r")
    matplotlib.cm.register_cmap(name, mymap)
    matplotlib.cm.register_cmap(name + "_r", mymapR)

registerList(nMethList, "nmeth")


baseFolder = os.getcwd()

if len(sys.argv) != 2:
    print "Usage: python aggregate.py scoreFolder"
    print """This file will add the "analyses" folder in the scoreFolder"""
    exit()


MIN_SEPARATION = 8  # minimum separation between patterns

folder = sys.argv[1]


files = os.listdir(os.path.join(folder, "saved"))
newfiles = []
for onefile in files:
    try:
        [int(i) for i in onefile.split("_")]
        newfiles.append(onefile)
    except:
        print "File ignored", onefile

matplotlib.rcParams.update({'font.size': 9})

newdatas = [pd.read_pickle(os.path.join(folder, "saved", i)) for i in newfiles]

# saved results as [[i1 i2 i3...], [j1 j2 j3...],
# [ipat1, ipat2, ipat3...], [jpat1, jpat2, jpat3...], [score1, score2, score3...]]
newdatas = [pd.DataFrame(i) for i in newdatas]
alldatas = pd.concat(newdatas, ignore_index=True)

print "Loaded {0} patterns".format(len(alldatas))

maxNum = max(alldatas["Pos l"].max(), alldatas["Pos r"].max())

alldatas["Freq prod"] = alldatas["Freq corr l"] * alldatas["Freq corr r"]

newScores = []
for i in xrange(6, 10):
    angle = 2 * np.pi * (i / 24.)
    alldatas["ScoreNew_{0}".format(i)] = alldatas["Score corr"] ** np.sin(angle) * alldatas["Freq prod"] ** np.cos(angle)
    newScores.append("ScoreNew_{0}".format(i))

alldatas["GC Both"] = alldatas["GC l"] + alldatas["GC r"]

alldatas["GC Max"] = np.maximum(alldatas["GC l"], alldatas["GC r"])



# loading and concatenating all datas
# i = position 1, j = position2, args = indices, values = their frequency

mask = np.abs(alldatas["Pos l"].values - alldatas["Pos r"].values) >= MIN_SEPARATION
# here we filter them out by distance. When I used the lowest distance of 3, then that was what scored!

alldatas = alldatas[mask]
alldatas = alldatas.reindex(np.random.permutation(alldatas.index))

# plt.plot(np.sort(alldatas["Score rel"].values))
# plt.title("Cumulative distribution of pattern scores")
# plt.show()


os.chdir(folder)
if not os.path.exists("analysis"):
    os.mkdir("analysis")
tf = alldatas[:20000]  # reindexed with random index, so don't worry

x = (tf["Freq raw l"] * tf["Freq raw r"]).values
y = tf["Score raw"].values
plt.scatter(x, y, s=3, marker=".", c=tf["GC Both"].values, linewidth=0)
plt.xlabel("Left score * right score (raw); {0} random patterns".format(len(tf)))
plt.ylabel("Pattern score (raw)")
plt.title("Color denotes cumulative GC, spearman r={0:.3%}".format(spearmanr(x, y)[0]))
plt.xscale("log")
plt.yscale("log")
plt.autoscale(tight=True)
nicePlot(show=False)
plt.savefig("analysis/GC_scatter_raw.png")
plt.savefig("analysis/GC_scatter_raw.pdf")
plt.clf()


x = (tf["Freq corr l"] * tf["Freq corr r"]).values
y = tf["Score corr"].values
plt.scatter(x, y, s=3, marker=".", c=tf["GC Both"].values, linewidth=0)
plt.xlabel("Left score * right score (corrected); {0} random patterns".format(len(tf)))
plt.ylabel("Pattern score (corrected)")
plt.title("Color denotes cumulative GC, spearman r = {0:.3%}".format(spearmanr(x, y)[0]))
plt.xscale("log")
plt.yscale("log")
plt.autoscale(tight=True)
nicePlot(show=False)
plt.savefig("analysis/GC_scatter_corrected.png")
plt.savefig("analysis/GC_scatter_corrected.pdf")
plt.clf()

print "Saved basic statistics"

if not os.path.exists("sortedBy"):
    os.mkdir("sortedBy")
os.chdir("sortedBy")


toScore = newScores
plt.figure(figsize=(5.5, 4.))
for j, sortBy in enumerate(toScore):


    bestNums = [100, 500, 3000, 10000, 30000]


    myDataFrame = alldatas.sort_values(sortBy)

    smallFrame = myDataFrame.ix[myDataFrame.index[-10000:]][::-1]
    smallFrame.to_csv("Best10000_sortBy_{0}.csv".format(sortBy))

    smallestFrame = myDataFrame.ix[myDataFrame.index[-1000:]][::-1]
    smallestFrame.to_csv("Best1000_sortBy_{0}.csv".format(sortBy))

    select = alldatas.reindex(np.random.permutation(alldatas.index)).copy()

    indsToColor = np.argsort(select[sortBy].values)
    select["color"] = 0


    col = select["color"].values
    for k, num in enumerate(sorted(bestNums)[::-1]):
        myarray = np.zeros(len(select["color"]), dtype=np.bool)
        myarray[indsToColor[-num:]] = True

        col[myarray] = k + 1
    select["color"] = col
    plt.clf()
    plen = len(select)
    select = select[:800000]

    x = (select["Freq corr l"] * select["Freq corr r"]).values
    y = select["Score corr"].values
    cm = cmap_map(cmap=matplotlib.cm.get_cmap("nmeth"), mapRange=(0.1, 0.9))

    plt.scatter(x, y, s=1., marker=".", c=select["color"].values,
                linewidth=0, cmap=cm)
    plt.autoscale(tight=True)
    plt.xlim((0, np.percentile(x, 100)))
    plt.ylim((0, np.percentile(y, 100)))
    # plt.xticks(range(0, int(x.max()), 1000000))
    plt.xlabel("Score left * score right, arbitrary units")
    plt.ylabel("Score of a pattern, arbitrary units")
    cbar = plt.colorbar(ticks=range(len(bestNums) + 1))
    cbar.ax.set_yticklabels(["all"] + [str(i) + " best" for i in bestNums[::-1]])

    nicePlot(fs=7, show=False)

    # plt.xscale("log")
    # plt.yscale("log")


    plt.savefig("{1:02}__sortBy_{0}_scatter.png".format(sortBy, j), dpi=600)
    plt.clf()


    for best in bestNums:
        plt.clf()
        # we used int8 for storage, gotta be careful now
        ID = myDataFrame["Pos l"].values + 200 * np.array(myDataFrame["Pos r"].values, dtype=np.int64)
        # leaving only the best
        ID = ID[-best:]
        # frequencies of each ID
        counts = np.bincount(ID)
        # renormalizing frequencies to get the same size of circles
        counts = counts / (best / 5000.)
        # list of all IDs that were used
        myrange = np.array(range(len(counts)))
        x = myrange / 200
        y = myrange % 200
        # keep only IDs that had counts
        mask = counts > 0


        x, y, counts = x[mask] , y[mask], counts[mask]
        # plt.xlim((-50, 0))
        # plt.ylim((-50, 0))

        plt.xlabel("Right box position")
        plt.ylabel("Left box position")
        plt.title("Dataset {0}".format(folder))

        plt.grid()
        plt.scatter(x, y, s=(.5 * counts) ** (1 / 2.), linewidth=0, c="black")

        # plt.title("Area of each dot represents relative representation of this position pair")
        # matplotlib.rcParams.update({'font.size': 9})
        plt.tight_layout(pad=0.3)

        # nicePlot(fs=7, show=False)
        # plt.tight_layout()
        plt.savefig("{2:02}_sortBy_{0}_best{1}_locations.pdf".format(sortBy, best, j))
        plt.clf()
print "finished! Ignore the segmentation fault which will be printed after"
