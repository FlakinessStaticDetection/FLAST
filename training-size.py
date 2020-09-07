from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os
import time

import flast


if __name__ == "__main__":
    print("VARYING TRAINING SIZE")

    outDir = "results/training-size/"
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    projectBasePath = "dataset"
    projectList = [
        # msr sampled dataset
        "pinto-ds",

        # msr dataset, per project experiment
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
    testSetSizes = [ts/100 for ts in range(95, 4, -5)]
    numKFold = 10

    params = {
        "algorithm": "brute",
        "metric": "cosine",
        "k": 7,
        "weights": "distance",
    }
    dim = 0  # number of dimensions (-1: no reduction; 0: JL with error eps)
    eps = 0.33  # JL eps

    reduceDim = False if dim < 0 else True  # reduce dimensionality if True

    # similarity threshold [0.5, 1]
    for projectName in projectList:
        for threshold in [0.5, 0.95]:
            p0 = time.time()
            print("#"*80)
            print("TESTING {}".format(projectName.upper()))

            pName = projectName.split("/")[-1]
            outputFile = f"{outDir}/{pName}___t{threshold}.csv"
            with open(outputFile, "w") as fileOut:
                fileOut.write("fold,testSetSize,numFlakyTrainSet,numNonFlakyTrainSet,numFlakyTestSet,numNonFlakyTestSet,vecTime,trainTime,testTime,avgPredTime,f-measure,precision,recall,accuracy,tp,fp,fn,tn\n")

            # data points vectorization
            v0 = time.perf_counter()
            dataPointsFlaky, dataPointsNonFlaky = flast.getDataPointsInfo(projectBasePath, projectName)
            dataPoints = dataPointsFlaky + dataPointsNonFlaky
            Z = flast.flastVectorization(dataPoints, reduceDim=reduceDim, dim=dim, eps=eps)
            dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
            dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))
            v1 = time.perf_counter()
            vecTime = v1 - v0

            for testSetSize in testSetSizes:
                t0 = time.time()
                print()
                print()
                print("TESTING {}".format(projectName.upper()))
                print("TESTSET SIZE:", testSetSize)

                kf = StratifiedShuffleSplit(n_splits=numKFold, test_size=testSetSize)
                successFold = 0
                for kFold, (trnIdx, tstIdx) in enumerate(kf.split(dataPointsList, dataLabelsList)):
                    trainData, testData = dataPointsList[trnIdx], dataPointsList[tstIdx]
                    trainLabels, testLabels = dataLabelsList[trnIdx], dataLabelsList[tstIdx]
                    if sum(trainLabels) == 0 or sum(testLabels) == 0:
                        print("Skipping fold...")
                        print(" Flaky Train Tests", sum(trainLabels))
                        print(" Flaky Test Tests", sum(testLabels))
                        continue

                    # prepare the data in the right format for kNN
                    nSamplesTrainData, nxTrain, nyTrain = trainData.shape
                    trainData = trainData.reshape((nSamplesTrainData, nxTrain * nyTrain))
                    nSamplesTestData, nxTest, nyTest = testData.shape
                    testData = testData.reshape((nSamplesTestData, nxTest * nyTest))

                    try:
                        trainTime, testTime, predictLabels = flast.flastClassification(trainData, trainLabels, testData, threshold, params)
                        res = flast.computeResults(trainData, testData, trainLabels, testLabels, predictLabels)
                        print(f"Precision: {res['precision']}")
                        print(f"Recall: {res['recall']}")
                    except ValueError:
                        continue
                    successFold += 1

                with open(outputFile, "a") as fileOut:
                    fileOut.write(
                        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                            successFold,
                            testSetSize,
                            sum(trainLabels),
                            len(trainLabels) - sum(trainLabels),
                            sum(testLabels),
                            len(testLabels) - sum(testLabels),
                            vecTime,
                            trainTime,
                            testTime,
                            testTime / len(testData),
                            res["f-measure"],
                            res["precision"],
                            res["recall"],
                            res["accuracy"],
                            res["tp"],
                            res["fp"],
                            res["fn"],
                            res["tn"]
                        )
                    )
                t1 = time.time()
                print("Running Time for TestSet Size:", t1 - t0)

            p1 = time.time()
            print("Running time for project:", p1 - p0)
