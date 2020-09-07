from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import time

import flast


if __name__ == "__main__":
    print("SINGLE PROJECTS - K-FOLD CV")

    outDir = "results/single-projects/"
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
    numKFold = 5

    params = {
        "algorithm": "brute",
        "metric": "cosine",
        "k": 3,
        "weights": "distance",
    }
    dim = 0  # number of dimensions (-1: no reduction; 0: JL with error eps)
    eps = 0.3  # JL eps

    reduceDim = False if dim < 0 else True  # reduce dimensionality if True

    # similarity threshold [0.5, 1]
    for threshold in [0.5, 0.95]:
        avgFile = f"{outDir}/FLAST_AVG___t{threshold}.csv"
        with open(avgFile, "w") as fileOut:
            fileOut.write("project,numFlakyTrainSet,numNonFlakyTrainSet,numFlakyTestSet,numNonFlakyTestSet,vecTime,trainTime,testTime,avgPredTime,f-measure,precision,recall,accuracy,tp,fp,fn,tn\n")

        for projectName in projectList:
            p0 = time.time()
            print("#"*80)
            print("TESTING {}".format(projectName.upper()))

            pName = projectName.split("/")[-1]
            outputFile = f"{outDir}/{pName}___t{threshold}.csv"
            with open(outputFile, "w") as fileOut:
                fileOut.write("fold,numFlakyTrainSet,numNonFlakyTrainSet,numFlakyTestSet,numNonFlakyTestSet,vecTime,trainTime,testTime,avgPredTime,f-measure,precision,recall,accuracy,tp,fp,fn,tn\n")

            exceptionsF = 0
            exceptionsP = 0
            AVG_numFlakyTrainSet = 0
            AVG_numNonFlakyTrainSet = 0
            AVG_numFlakyTestSet = 0
            AVG_numNonFlakyTestSet = 0
            AVG_vecTime = 0
            AVG_trainTime = 0
            AVG_testTime = 0
            AVG_avgPredTime = 0
            AVG_fMeasure = 0
            AVG_precision = 0
            AVG_recall = 0
            AVG_accuracy = 0
            AVG_tp = 0
            AVG_fp = 0
            AVG_fn = 0
            AVG_tn = 0

            # data points vectorization
            v0 = time.perf_counter()
            dataPointsFlaky, dataPointsNonFlaky = flast.getDataPointsInfo(projectBasePath, projectName)
            dataPoints = dataPointsFlaky + dataPointsNonFlaky
            Z = flast.flastVectorization(dataPoints, reduceDim=reduceDim, dim=dim, eps=eps)
            dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
            dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))
            v1 = time.perf_counter()
            vecTime = v1 - v0

            kf = StratifiedKFold(n_splits=numKFold, shuffle=True)
            successFold = 0
            for kFold, (trnIdx, tstIdx) in enumerate(kf.split(dataPointsList, dataLabelsList)):
                trainData, testData = dataPointsList[trnIdx], dataPointsList[tstIdx]
                trainLabels, testLabels = dataLabelsList[trnIdx], dataLabelsList[tstIdx]
                if sum(trainLabels) == 0 or sum(testLabels) == 0:
                    print("Skipping fold...")
                    print(" Flaky Train Tests", sum(trainLabels))
                    print(" Flaky Test Tests", sum(testLabels))
                    continue
                else:
                    successFold += 1

                # prepare the data in the right format for kNN
                nSamplesTrainData, nxTrain, nyTrain = trainData.shape
                trainData = trainData.reshape((nSamplesTrainData, nxTrain * nyTrain))
                nSamplesTestData, nxTest, nyTest = testData.shape
                testData = testData.reshape((nSamplesTestData, nxTest * nyTest))

                trainTime, testTime, predictLabels = flast.flastClassification(trainData, trainLabels, testData, threshold, params)
                res = flast.computeResults(trainData, testData, trainLabels, testLabels, predictLabels)
                print(f"Precision: {res['precision']}")
                print(f"Recall: {res['recall']}")

                AVG_numFlakyTrainSet += sum(trainLabels)
                AVG_numNonFlakyTrainSet += len(trainLabels) - sum(trainLabels)
                AVG_numFlakyTestSet += sum(testLabels)
                AVG_numNonFlakyTestSet += len(testLabels) - sum(testLabels)
                AVG_vecTime += vecTime
                AVG_trainTime += trainTime
                AVG_testTime += testTime
                AVG_avgPredTime += testTime / len(testData)
                try:
                    AVG_fMeasure += res["f-measure"]
                except:
                    exceptionsF += 1
                try:
                    AVG_precision += res["precision"]
                except:
                    exceptionsP += 1
                AVG_recall += res["recall"]
                AVG_accuracy += res["accuracy"]
                AVG_tp += res["tp"]
                AVG_fp += res["fp"]
                AVG_fn += res["fn"]
                AVG_tn += res["tn"]

                with open(outputFile, "a") as fileOut:
                    fileOut.write(
                        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                            successFold,
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

            p1 = time.time()
            print("Running time for project:", p1 - p0)

            if successFold - exceptionsF == 0:
                AVG_fMeasure = "-"
            else:
                AVG_fMeasure /= successFold - exceptionsF

            if successFold - exceptionsP == 0:
                AVG_precision = "-"
            else:
                AVG_precision /= successFold - exceptionsP

            with open(avgFile, "a") as fileOut:
                fileOut.write(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        pName,
                        (AVG_numFlakyTrainSet / successFold),
                        (AVG_numNonFlakyTrainSet / successFold),
                        (AVG_numFlakyTestSet / successFold),
                        (AVG_numNonFlakyTestSet / successFold),
                        (AVG_vecTime / successFold),
                        (AVG_trainTime / successFold),
                        (AVG_testTime / successFold),
                        (AVG_avgPredTime / successFold),
                        AVG_fMeasure,
                        AVG_precision,
                        (AVG_recall / successFold),
                        (AVG_accuracy / successFold),
                        (AVG_tp / successFold),
                        (AVG_fp / successFold),
                        (AVG_fn / successFold),
                        (AVG_tn / successFold)
                    )
                )
