import os
import time
import warnings

import numpy as np

from scipy import spatial

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score


###############################################################################
# read data from file

def getDataPoints(path):
    dataPointsList = []
    for dataPointName in os.listdir(path):
        if dataPointName[0] == ".":
            continue
        with open(os.path.join(path, dataPointName), encoding="utf-8") as fileIn:
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

def computeResults(testLabels, predictLabels):
    warnings.filterwarnings("error")  # to catch warnings, e.g., "prec set to 0.0"
    try:
        precision = precision_score(testLabels, predictLabels)
    except:
        precision = "-"
    try:
        recall = recall_score(testLabels, predictLabels)
    except:
        recall = "-"
    warnings.resetwarnings()  # warnings are no more errors
    return precision, recall


###############################################################################
# FLAST

def flastVectorization(dataPoints, dim=0, eps=0.3):
    countVec = CountVectorizer()
    Z_full = countVec.fit_transform(dataPoints)
    if eps == 0:
        Z = Z_full
    else:
        if dim <= 0:
            dim = johnson_lindenstrauss_min_dim(Z_full.shape[0], eps=eps)
        srp = SparseRandomProjection(n_components=dim)
        Z = srp.fit_transform(Z_full)
    return Z


def flastClassification(trainData, trainLabels, testData, sigma, k, params):
    # training
    t0 = time.perf_counter()
    kNN = KNeighborsClassifier(
        algorithm=params["algorithm"],
        metric=params["metric"],
        weights=params["weights"],
        n_neighbors=k,
        n_jobs=1
    )
    kNN.fit(trainData, trainLabels)
    t1 = time.perf_counter()
    trainTime = t1 - t0

    t0 = time.perf_counter()
    predictLabels = []
    neighborDist, neighborInd = kNN.kneighbors(testData)
    for (distances, indices) in zip(neighborDist, neighborInd):
        phi, psi = 0, 0
        for (distance, neighbor) in zip(distances, indices):
            if kNN.get_params()["weights"] == "distance":
                dInv = (1 / distance) if distance != 0 else float("Inf")
            else:
                dInv = 1
            if trainLabels[neighbor] == 1:
                phi += dInv
            else:
                psi += dInv

        # handle limit cases for prediction
        if phi == float("Inf") and psi == float("Inf"):
            prediction = 0
        elif psi == float("Inf"):
            prediction = 0
        elif phi == float("Inf"):
            prediction = 1
        elif (phi + psi) == 0:
            prediction = 0
        else:
            if phi / (phi + psi) >= sigma:
                prediction = 1
            else:
                prediction = 0
        predictLabels.append(prediction)

    t1 = time.perf_counter()
    testTime = t1 - t0

    return trainTime, testTime, predictLabels
