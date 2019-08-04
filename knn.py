import operator

import numpy as np


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(test_vec, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    test_vec = np.tile(test_vec, (dataSetSize, 1))
    diffMat = test_vec - dataSet
    distance = (diffMat ** 2).sum(axis=1) ** 0.5
    sortedDistIndicies = distance.argsort()
    classCount = {}
    for i in range(k):
        voteIlables = labels[sortedDistIndicies[i]]
        classCount[voteIlables] = classCount.get(voteIlables, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    index = 0
    classLabelVector = []
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index = index + 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('./datingTestSet2.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    m = normDataSet.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normDataSet[i, :], normDataSet[numTestVecs:m, :], datingLabels[numTestVecs:m], 10)
        if classifierResult != datingLabels[i]:
            errorCount = errorCount + 1.0
    print(errorCount / float(numTestVecs))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("playing video games?"))
    ffMiles = float(input("liters of ice cream consumed per year?"))
    iceCream = float(input("liters of ice cream consumed?"))
    datingDataMat, datingLabels = file2matrix('./datingTestSet2.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    inArr = [percentTats, ffMiles, iceCream]
    inArr = (inArr - minVals) / ranges

    result = classify(inArr, normDataSet, datingLabels, 3)
    print(resultList[result-1])

if __name__ == '__main__':
    group, labels = createDataSet()
    test_vec = [0, 0]
    returnMat, classLabelVector = file2matrix("./datingTestSet2.txt")
    normDataSet, ranges, minVals = autoNorm(returnMat)
    datingClassTest()
    classifyPerson()
    # print(classify(test_vec, returnMat, classLabelVector, 3))
