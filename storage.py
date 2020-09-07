from scipy import sparse
import numpy as np
import os
import pickle
import time

import flast


if __name__ == "__main__":
    print("RQ INTRA PROJECTS - STORAGE")

    outDir = "results/storage"
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

    params = {
        "algorithm": "brute",
        "metric": "cosine",
        "k": 7,
        "weights": "distance",
    }
    dim = 0  # number of dimensions (-1: no reduction; 0: JL with error eps)
    eps = 0.3  # JL eps

    reduceDim = False if dim < 0 else True  # reduce dimensionality if True

    tSize = 0.9  # percentage (in [0,1]) of training set

    outputFile = f"{outDir}/eps{eps}.csv"
    with open(outputFile, "w") as fileOut:
            fileOut.write("project,trainingSize,dimensions,eps,storage\n")

    for projectName in projectList:
        p0 = time.time()
        print("#"*80)
        print("TESTING {}".format(projectName.upper()))

        # data points vectorization
        v0 = time.perf_counter()
        dataPointsFlaky, dataPointsNonFlaky = flast.getDataPointsInfo(projectBasePath, projectName)
        dataPoints = dataPointsFlaky + dataPointsNonFlaky
        Z = flast.flastVectorization(dataPoints, reduceDim=reduceDim, dim=dim, eps=eps)
        dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
        dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))
        v1 = time.perf_counter()
        vecTime = v1 - v0

        trainSize = int(tSize * len(dataPointsList))
        trainData, testData = dataPointsList[:trainSize], dataPointsList[trainSize:]
        trainLabels, testLabels = dataLabelsList[:trainSize], dataLabelsList[trainSize:]
        # prepare the data in the right format for kNN
        nSamplesTrainData, nxTrain, nyTrain = trainData.shape
        trainData = trainData.reshape((nSamplesTrainData, nxTrain * nyTrain))

        kNN = (sparse.coo_matrix(trainData), sparse.coo_matrix(trainLabels))
        pName = projectName.split("/")[-1]
        pickleDumpKNN = f"{outDir}/{pName}___d{dim}.pickle"
        pickle.dump(kNN, open(pickleDumpKNN, "wb"))
        storage = os.path.getsize(pickleDumpKNN)

        print(f"Bytes: {storage}")

        with open(outputFile, "a") as fileOut:
            fileOut.write(f"{projectName},{tSize},{dim},{eps},{storage}\n")

        if os.path.exists(pickleDumpKNN):
            os.remove(pickleDumpKNN)
        else:
            print(f"{pickleDumpKNN} does not exists")

        p1 = time.time()
        print("Running time for project:", p1 - p0)
