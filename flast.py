from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import random
import time
import statistics
import warnings


###############################################################################
# read data from file

def getDataPoints(path):
    dataPointsList = []
    for dataPointName in os.listdir(path):
        if dataPointName[0] == ".":
            continue
        with open(os.path.join(path, dataPointName)) as fileIn:
            dp = fileIn.read()
        dataPointsList.append(dp)
    return dataPointsList


def getDataPointsInfo(projectBasePath, projectName):
    # get list of tokenized test methods
    projectPath = os.path.join(projectBasePath, projectName)
    flakyPath = os.path.join(projectPath, "flakyMethods")
    nonFlakyPath = os.path.join(projectPath, "nonFlakyMethods")
    return getDataPoints(flakyPath), getDataPoints(nonFlakyPath)


###############################################################################
# compute effectiveness metrics

def computeResults(trainData, testData, trainLabels, testLabels, predictLabels):
    # compute metrics
    res = {}

    warnings.filterwarnings("error")  # to catch warnings, e.g., "prec set to 0.0"
    try:
        res["f-measure"] = f1_score(testLabels, predictLabels)
    except:
        res["f-measure"] = "-"
    try:
        res["precision"] = precision_score(testLabels, predictLabels)
    except:
        res["precision"] = "-"
    try:
        res["recall"] = recall_score(testLabels, predictLabels)
    except:
        res["recall"] = "-"
    warnings.resetwarnings()  # warnings are no more errors

    res["accuracy"] = accuracy_score(testLabels, predictLabels)
    tn, fp, fn, tp = confusion_matrix(testLabels, predictLabels).ravel()
    res["tp"] = tp
    res["fp"] = fp
    res["fn"] = fn
    res["tn"] = tn

    return res


###############################################################################
# FLAST

def flastVectorization(dataPoints, reduceDim=True, dim=0, eps=0.33):
    countVec = CountVectorizer()
    Z_full = countVec.fit_transform(dataPoints)
    if reduceDim:
        if dim <= 0:
            dim = johnson_lindenstrauss_min_dim(Z_full.shape[0], eps=eps)
        srp = SparseRandomProjection(n_components=dim)
        Z = srp.fit_transform(Z_full)
        return Z
    else:
        return Z_full


def flastClassification(trainData, trainLabels, testData, threshold, params):
    # setup kNN classifier
    kNN = KNeighborsClassifier(
        algorithm=params["algorithm"],
        metric=params["metric"],
        n_neighbors=params["k"],
        weights=params["weights"],
        n_jobs=1
    )

    # training
    t0 = time.perf_counter()
    kNN.fit(trainData, trainLabels)
    t1 = time.perf_counter()
    trainTime = t1 - t0

    # testing
    p0 = time.perf_counter()
    predictLabels = kNN.predict(testData)
    p1 = time.perf_counter()
    testTime = p1 - p0

    # adjust predictions based on threshold
    if threshold != 0.5:
        for i, pred in enumerate(predictLabels):
            if pred == 1:  # if predicted flaky
                dp = np.array(testData[i]).reshape(1, -1)
                dist, ind = kNN.kneighbors(dp)  # get neighbors
                f, nf = 0, 0
                for l in range(len(ind[0])):
                    j, d = ind[0][l], dist[0][l]
                    if trainLabels[j] == 1:
                        f += d
                    else:
                        nf += d
                try:
                    if (f / (f + nf)) < threshold:
                        predictLabels[i] = 0
                except:
                    predictLabels[i] = 0

    return trainTime, testTime, predictLabels
