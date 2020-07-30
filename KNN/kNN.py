from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
from pip._vendor.distlib.compat import raw_input


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDiatance = sqDiffMat.sum(axis=1)
    distances = sqDiatance ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

filename = "./datingTestSet2.txt"
datingDataMat, datingLabels = file2matrix(filename)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15*array(datingLabels), 15*array(datingLabels))
# plt.show()

def autoNorm(dataset):
    minValue = dataset.min(0)
    maxValue = dataset.max(0)
    ranges = maxValue - minValue
    normDataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normDataset = dataset - tile(minValue, (m,1))
    normDataset = normDataset / tile(ranges, (m,1))
    return normDataset, ranges, minValue

normat, ranges, minValue = autoNorm(datingDataMat)
print(normat)
print(ranges)
print(minValue)

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("./datingTestSet2.txt")
    normat, ranges, minValue = autoNorm(datingDataMat)
    m = normat.shape[0]
    numTestVecs = int(m*hoRatio)
    erroeCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normat[i,:], normat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m],3
                                     )
        print("the classifier came back with : %d , the real answer is: %d" % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            erroeCount += 1

    print("the total error rate is : %f" % (erroeCount / float(numTestVecs)))

def classifyPerson():
    resultList = ["not at all", "in small doses", "in large doses"]
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("./datingTestSet2.txt")
    normat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normat, datingLabels,3)
    print("You will probably like this person: ", resultList[classifierResult - 1])

