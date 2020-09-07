from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import random
import time
import warnings

import flast


if __name__ == "__main__":
    print("SINGLE PROJECTS - RANDOM CLASSIFIER")

    outDir = "results/random/"
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

    # probability of random classifier
    for prc in [0.16, 0.5]:
        avgFile = f"{outDir}/RANDOM_AVG___p{prc}.csv"
        with open(avgFile, "w") as fileOut:
            fileOut.write("project,numFlakyTrainSet,numNonFlakyTrainSet,numFlakyTestSet,numNonFlakyTestSet,vecTime,trainTime,testTime,avgPredTime,f-measure,precision,recall,accuracy,tp,fp,fn,tn\n")

        for projectName in projectList:
            p0 = time.time()
            print("#"*80)
            print("TESTING {}".format(projectName.upper()))

            # prepare output folder
            pName = projectName.split("/")[-1]
            outputFile = f"{outDir}/{pName}___p{prc}.csv"
            with open(outputFile, "w") as fileOut:
                fileOut.write("fold,numFlakyTrainSet,numNonFlakyTrainSet,numFlakyTestSet,numNonFlakyTestSet,vecTime,trainTime,testTime,avgPredTime,f-measure,precision,recall,accuracy,tp,fp,fn,tn\n")

            exceptions = 0
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
            dataPointsList = np.array(dataPoints)
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

                # RANDOM CLASSIFIER
                p0 = time.perf_counter()
                predictLabels = np.array([1 if random.random() < prc else 0
                    for _ in range(len(testLabels))])
                p1 = time.perf_counter()
                testTime = p1 - p0

                res = flast.computeResults(trainData, testData, trainLabels, testLabels, predictLabels)
                print(f"Precision: {res['precision']}")
                print(f"Recall: {res['recall']}")

                AVG_numFlakyTrainSet += sum(trainLabels)
                AVG_numNonFlakyTrainSet += len(trainLabels) - sum(trainLabels)
                AVG_numFlakyTestSet += sum(testLabels)
                AVG_numNonFlakyTestSet += len(testLabels) - sum(testLabels)
                AVG_vecTime += vecTime
                AVG_testTime += testTime
                AVG_avgPredTime += testTime / len(testData)
                try:
                    AVG_fMeasure += res["f-measure"]
                    AVG_precision += res["precision"]
                except:
                    exceptions += 1
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
                            0,
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

            if numKFold - exceptions == 0:
                AVGFMEASURE = "-"
                AVGPRECISION = "-"
            else:
                AVGFMEASURE = AVG_fMeasure / (numKFold - exceptions)
                AVGPRECISION = AVG_precision / (numKFold - exceptions)

            with open(avgFile, "a") as fileOut:
                fileOut.write(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        pName,
                        (AVG_numFlakyTrainSet / numKFold),
                        (AVG_numNonFlakyTrainSet / numKFold),
                        (AVG_numFlakyTestSet / numKFold),
                        (AVG_numNonFlakyTestSet / numKFold),
                        (AVG_vecTime / numKFold),
                        AVG_trainTime,
                        (AVG_testTime / numKFold),
                        (AVG_avgPredTime / numKFold),
                        AVGFMEASURE,
                        AVGPRECISION,
                        (AVG_recall / numKFold),
                        (AVG_accuracy / numKFold),
                        (AVG_tp / numKFold),
                        (AVG_fp / numKFold),
                        (AVG_fn / numKFold),
                        (AVG_tn / numKFold)
                    )
                )
