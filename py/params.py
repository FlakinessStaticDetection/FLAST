import os
import pickle
import time

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

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
    avgFlakyTrain, avgNonFlakyTrain, avgFlakyTest, avgNonFlakyTest = 0, 0, 0, 0
    successFold, precisionFold = 0, 0
    for (trnIdx, tstIdx) in kf.split(dataPointsList, dataLabelsList):
        dataPointsFlaky, dataPointsNonFlaky = flast.getDataPointsInfo(projectBasePath, projectName)
        dataPoints = dataPointsFlaky + dataPointsNonFlaky
        Z = flast.flastVectorization(dataPoints, dim=dim, eps=eps)
        dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
        dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))

        trainData, testData = dataPointsList[trnIdx], dataPointsList[tstIdx]
        trainLabels, testLabels = dataLabelsList[trnIdx], dataLabelsList[tstIdx]
        if sum(trainLabels) == 0 or sum(testLabels) == 0:
            print("Skipping fold...")
            print(" Flaky Train Tests", sum(trainLabels))
            print(" Flaky Test Tests", sum(testLabels))
            continue

        successFold += 1
        avgFlakyTrain += sum(trainLabels)
        avgNonFlakyTrain += len(trainLabels) - sum(trainLabels)
        avgFlakyTest += sum(testLabels)
        avgNonFlakyTest += len(testLabels) - sum(testLabels)

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
    avgFlakyTrain /= successFold
    avgNonFlakyTrain /= successFold
    avgFlakyTest /= successFold
    avgNonFlakyTest /= successFold

    return (avgFlakyTrain, avgNonFlakyTrain, avgFlakyTest, avgNonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred)


if __name__ == "__main__":
    projectBasePath = "dataset"
    projectName = "pinto-ds"
    outDir = "results/"
    os.makedirs(outDir, exist_ok=True)

    numSplit = 30
    testSetSize = 0.2
    kf = StratifiedShuffleSplit(n_splits=numSplit, test_size=testSetSize)

    # DISTANCE
    outFile = "params-distance.csv"
    with open(os.path.join(outDir, outFile), "w") as fo:
        fo.write("distance,k,sigma,eps,precision,recall,storage,preparationTime,predictionTime\n")

    k = 7
    sigma = 0.5
    dim = 0  # number of dimensions (0: JL with error eps)
    eps = 0.3  # JL eps
    params = { "algorithm": "brute", "metric": "cosine", "weights": "uniform" }
    for metric in ["cosine", "euclidean"]:
        for k in [3, 7]:
            print(f"{metric=}, {k=}")
            params["metric"] = metric
            (flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred) = flastKNN(outDir, projectBasePath, projectName, kf, dim, eps, k, sigma, params)
            with open(os.path.join(outDir, outFile), "a") as fo:
                fo.write("{},{},{},{},{},{},{},{},{}\n".format(params["metric"], k, sigma, eps, avgP, avgR, storage, avgTPrep, avgTPred))

    # EPSILON
    outFile = "params-eps.csv"
    with open(os.path.join(outDir, outFile), "w") as fo:
        fo.write("distance,k,sigma,eps,precision,recall,storage,preparationTime,predictionTime\n")

    k = 7
    sigma = 0.5
    dim = 0  # number of dimensions (0: JL with error eps)
    eps = 0.3  # JL eps
    params = { "algorithm": "brute", "metric": "cosine", "weights": "uniform" }
    for eps in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"{eps=}")
        (flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred) = flastKNN(outDir, projectBasePath, projectName, kf, dim, eps, k, sigma, params)
        with open(os.path.join(outDir, outFile), "a") as fo:
            fo.write("{},{},{},{},{},{},{},{},{}\n".format(params["metric"], k, sigma, eps, avgP, avgR, storage, avgTPrep, avgTPred))

    # NEIGHBORS K
    outFile = "params-k.csv"
    with open(os.path.join(outDir, outFile), "w") as fo:
        fo.write("distance,k,sigma,eps,precision,recall,storage,preparationTime,predictionTime\n")

    k = 7
    sigma = 0.5
    dim = 0  # number of dimensions (0: JL with error eps)
    eps = 0.3  # JL eps
    params = { "algorithm": "brute", "metric": "cosine", "weights": "uniform" }
    for k in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
        print(f"{k=}")
        (flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred) = flastKNN(outDir, projectBasePath, projectName, kf, dim, eps, k, sigma, params)
        with open(os.path.join(outDir, outFile), "a") as fo:
            fo.write("{},{},{},{},{},{},{},{},{}\n".format(params["metric"], k, sigma, eps, avgP, avgR, storage, avgTPrep, avgTPred))

    # THRESHOLD SIGMA
    outFile = "params-sigma.csv"
    with open(os.path.join(outDir, outFile), "w") as fo:
        fo.write("distance,k,sigma,eps,precision,recall,storage,preparationTime,predictionTime\n")

    k = 7
    sigma = 0.5
    dim = 0  # number of dimensions (0: JL with error eps)
    eps = 0.3  # JL eps
    params = { "algorithm": "brute", "metric": "cosine", "weights": "uniform" }
    for sigma in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, .35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
        print(f"{sigma=}")
        (flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred) = flastKNN(outDir, projectBasePath, projectName, kf, dim, eps, k, sigma, params)
        with open(os.path.join(outDir, outFile), "a") as fo:
            fo.write("{},{},{},{},{},{},{},{},{}\n".format(params["metric"], k, sigma, eps, avgP, avgR, storage, avgTPrep, avgTPred))

    # TRAINING
    outFile = "params-training.csv"
    with open(os.path.join(outDir, outFile), "w") as fo:
        fo.write("trainingSetSize,k,sigma,eps,precision,recall,storage,preparationTime,predictionTime\n")

    k = 7
    dim = 0  # number of dimensions (0: JL with error eps)
    eps = 0.3  # JL eps
    params = { "algorithm": "brute", "metric": "cosine", "weights": "uniform" }
    numSplit = 30
    for testSetSize in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        trainingSetSize = 1 - testSetSize
        for sigma in [0.5, 0.95]:
            kf = StratifiedShuffleSplit(n_splits=numSplit, test_size=testSetSize)
            print(f"{k=}, {sigma=}, {testSetSize=}")
            (flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred) = flastKNN(outDir, projectBasePath, projectName, kf, dim, eps, k, sigma, params)
            with open(os.path.join(outDir, outFile), "a") as fo:
                fo.write("{},{},{},{},{},{},{},{},{}\n".format(trainingSetSize, k, sigma, eps, avgP, avgR, storage, avgTPrep, avgTPred))
