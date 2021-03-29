import os
import pickle
import time

import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

import flast


def flastKNN(outDir, projectBasePath, projectName, kf, dim, eps, k, sigma, params):
    v0 = time.perf_counter()
    dataPointsFlaky, dataPointsNonFlaky = flast.getDataPointsInfo(projectBasePath, projectName)
    dataPoints = dataPointsFlaky + dataPointsNonFlaky
    Z = flast.flastVectorization(dataPoints, dim=dim, eps=eps)
    dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
    dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))
    v1 = time.perf_counter()
    vecTime = v1 - v0

    # storage
    kNN = (dataPointsList, dataLabelsList)
    pickleDumpKNN = os.path.join(outDir, "flast-k{}-sigma{}.pickle".format(k, sigma))
    with open(pickleDumpKNN, "wb") as pickleFile:
        pickle.dump(kNN, pickleFile)
    storage = os.path.getsize(pickleDumpKNN)
    os.remove(pickleDumpKNN)

    avgP, avgR = 0, 0
    avgTPrep, avgTPred = 0, 0
    successFold, precisionFold = 0, 0
    for (trnIdx, tstIdx) in kf.split(dataPointsList, dataLabelsList):
        trainData, testData = dataPointsList[trnIdx], dataPointsList[tstIdx]
        trainLabels, testLabels = dataLabelsList[trnIdx], dataLabelsList[tstIdx]
        if sum(trainLabels) == 0 or sum(testLabels) == 0:
            print("Skipping fold...")
            print(" Flaky Train Tests", sum(trainLabels))
            print(" Flaky Test Tests", sum(testLabels))
            continue

        successFold += 1

        # prepare the data in the right format for kNN
        nSamplesTrainData, nxTrain, nyTrain = trainData.shape
        trainData = trainData.reshape((nSamplesTrainData, nxTrain * nyTrain))
        nSamplesTestData, nxTest, nyTest = testData.shape
        testData = testData.reshape((nSamplesTestData, nxTest * nyTest))
        trainTime, testTime, predictLabels = flast.flastClassification(trainData, trainLabels, testData, sigma, k, params)

        preparationTime = (vecTime * len(trainData) / len(dataPoints)) + trainTime
        predictionTime = (vecTime / len(dataPoints)) + (testTime / len(testData))
        (precision, recall) = flast.computeResults(testLabels, predictLabels)
        print(precision, recall)
        if precision != "-":
            precisionFold += 1
            avgP += precision
        avgR += recall
        avgTPrep += preparationTime
        avgTPred += predictionTime

    if precisionFold == 0:
        avgP = "-"
    else:
        avgP /= precisionFold
    avgR /= successFold
    avgTPrep /= successFold
    avgTPred /= successFold

    return (avgP, avgR, storage, avgTPrep, avgTPred)


def pintoKNN(outDir, projectBasePath, projectName, kf):
    v0 = time.perf_counter()
    dataPointsFlaky, dataPointsNonFlaky = flast.getDataPointsInfo(projectBasePath, projectName)
    dataPoints = dataPointsFlaky + dataPointsNonFlaky
    countVec = CountVectorizer()
    Z = countVec.fit_transform(dataPoints)
    dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
    dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))
    v1 = time.perf_counter()
    vecTime = v1 - v0

    # storage
    kNN = (dataPointsList, dataLabelsList)
    pickleDumpKNN = os.path.join(outDir, "PintoKNN.pickle")
    with open(pickleDumpKNN, "wb") as pickleFile:
        pickle.dump(kNN, pickleFile)
    storage = os.path.getsize(pickleDumpKNN)
    os.remove(pickleDumpKNN)

    avgP, avgR = 0, 0
    avgTPrep, avgTPred = 0, 0
    successFold, precisionFold = 0, 0
    for (trnIdx, tstIdx) in kf.split(dataPointsList, dataLabelsList):
        trainData, testData = dataPointsList[trnIdx], dataPointsList[tstIdx]
        trainLabels, testLabels = dataLabelsList[trnIdx], dataLabelsList[tstIdx]
        if sum(trainLabels) == 0 or sum(testLabels) == 0:
            print("Skipping fold...")
            print(" Flaky Train Tests", sum(trainLabels))
            print(" Flaky Test Tests", sum(testLabels))
            continue

        successFold += 1

        # prepare the data in the right format for kNN
        nSamplesTrainData, nxTrain, nyTrain = trainData.shape
        trainData = trainData.reshape((nSamplesTrainData, nxTrain * nyTrain))
        nSamplesTestData, nxTest, nyTest = testData.shape
        testData = testData.reshape((nSamplesTestData, nxTest * nyTest))

        # training
        t0 = time.perf_counter()
        kNN = KNeighborsClassifier(
            algorithm= "brute",
            metric="euclidean",
            weights="uniform",
            n_neighbors=1,
            n_jobs=1
        )
        kNN.fit(trainData, trainLabels)
        t1 = time.perf_counter()
        trainTime = t1 - t0

        # testing
        p0 = time.perf_counter()
        predictLabels = kNN.predict(testData)
        p1 = time.perf_counter()
        testTime = p1 - p0

        preparationTime = (vecTime * len(trainData) / len(dataPoints)) + trainTime
        predictionTime = (vecTime / len(dataPoints)) + (testTime / len(testData))
        (precision, recall) = flast.computeResults(testLabels, predictLabels)
        print(precision, recall)
        if precision != "-":
            precisionFold += 1
            avgP += precision
        avgR += recall
        avgTPrep += preparationTime
        avgTPred += predictionTime

    if precisionFold == 0:
        avgP = "-"
    else:
        avgP /= precisionFold
    avgR /= successFold
    avgTPrep /= successFold
    avgTPred /= successFold

    return (avgP, avgR, storage, avgTPrep, avgTPred)


if __name__ == "__main__":
    projectBasePath = "dataset"
    projectName = "pinto-ds"

    outDir = "results/"
    outFile = "compare-pinto.csv"
    os.makedirs(outDir, exist_ok=True)
    with open(os.path.join(outDir, outFile), "w") as fo:
        fo.write("approach,precision,recall,storage,preparationTime,predictionTime\n")

    # CV
    numSplit = 30
    testSize = 0.2
    kf = ShuffleSplit(n_splits=numSplit, test_size=testSize)

    # PINTO
    print("Pinto-KNN")
    (avgP, avgR, storage, avgTPrep, avgTPred) = pintoKNN(outDir, projectBasePath, projectName, kf)
    with open(os.path.join(outDir, outFile), "a") as fo:
        fo.write("Pinto-KNN,{},{},{},{},{}\n".format(avgP, avgR, storage, avgTPrep, avgTPred))

    # FLAST
    dim = 0  # number of dimensions (0: JL with error eps)
    eps = 0.3  # JL eps
    params = { "algorithm": "brute", "metric": "cosine", "weights": "distance" }
    for k in [3, 7]:
        for sigma in [0.5, 0.95]:
            print("FLAST", k, sigma)
            (avgP, avgR, storage, avgTPrep, avgTPred) = flastKNN(outDir, projectBasePath, projectName, kf, dim, eps, k, sigma, params)
            with open(os.path.join(outDir, outFile), "a") as fo:
                fo.write("FLAST-k{}-sigma{},{},{},{},{},{}\n".format(k, sigma, avgP, avgR, storage, avgTPrep, avgTPred))
