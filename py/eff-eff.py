import os
import pickle
import time

import numpy as np

from sklearn.model_selection import StratifiedKFold

import flast


def flastKNN(outDir, projectBasePath, projectName, kf, dim, eps, k, sigma):
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
    successFold = 0
    for kFold, (trnIdx, tstIdx) in enumerate(kf.split(dataPointsList, dataLabelsList)):
        trainData, testData = dataPointsList[trnIdx], dataPointsList[tstIdx]
        trainLabels, testLabels = dataLabelsList[trnIdx], dataLabelsList[tstIdx]
        if sum(trainLabels) == 0 or sum(testLabels) == 0:
            print("Skipping fold...")
            print(" Flaky Train Tests", sum(trainLabels))
            print(" Flaky Test Tests", sum(testLabels))
            continue

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
        (precision, recall) = flast.computeResults(trainData, testData, trainLabels, testLabels, predictLabels)
        print(precision, recall)
        if precision != "-" and recall != "-":
            successFold += 1
            avgP += precision
            avgR += recall
            avgTPrep += preparationTime
            avgTPred += predictionTime

    if successFold == 0:
        avgP = "-"
        avgR = "-"
        avgTPrep = "-"
        avgTPred = "-"
        avgFlakyTrain = "-"
        avgNonFlakyTrain = "-"
        avgFlakyTest = "-"
        avgNonFlakyTest = "-"
    else:
        avgP /= successFold
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
    projectList = [
        "achilles",
        "alluxio-tachyon",
        "ambari",
        "hadoop",
        "jackrabbit-oak",
        "jimfs",
        "ninja",
        "okhttp",
        "oozie",
        "oryx",
        "spring-boot",
        "togglz",
        "wro4j",
    ]
    outDir = "results/"
    outFile = "eff-eff.csv"
    os.makedirs(outDir, exist_ok=True)
    with open(os.path.join(outDir, outFile), "w") as fo:
        fo.write("dataset,flakyTrain,nonFlakyTrain,flakyTest,nonFlakyTest,k,sigma,precision,recall,storage,preparationTime,predictionTime\n")

    numKFold = 5
    kf = StratifiedKFold(n_splits=numKFold, shuffle=True)

    # FLAST
    dim = 0  # number of dimensions (0: JL with error eps)
    eps = 0.3  # JL eps
    params = {
        "algorithm": "brute",
        "metric": "cosine",
    }
    for k in [3, 7]:
        for sigma in [0.5, 0.95]:
            for projectName in projectList:
                print(projectName.upper(), "FLAST", k, sigma)
                (flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred) = flastKNN(outDir, projectBasePath, projectName, kf, dim, eps, k, sigma)
                with open(os.path.join(outDir, outFile), "a") as fo:
                    fo.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(projectName, flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, k, sigma, avgP, avgR, storage, avgTPrep, avgTPred))
