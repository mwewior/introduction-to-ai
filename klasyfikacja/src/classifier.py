from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from dataset import DataSet


clfTree = DecisionTreeClassifier(
    criterion="entropy", splitter="best", max_depth=6
)

clfSVM = SVC(
    C=1, kernel="poly", tol=10e-16  # , max_iter=int(10e4)
)

ds = DataSet(K=5)

trainFeatures = ds.joinedTrainFeatures
trainTarget = ds.joinedTrainTarget

singleGroupLength = len(trainTarget)//5

clfSVM.fit(
    trainFeatures[0:4*singleGroupLength],
    trainTarget[0:4*singleGroupLength])

print("\n\nvalidation\n")
testPrediction = clfSVM.predict(ds.testFeatures)
expectedTarget = ds.testTarget

for i in range(len(expectedTarget)):
    diff = testPrediction[i] - expectedTarget[i]
    print(f'{diff}\t{testPrediction[i]} : {expectedTarget[i]}')

print("\n\ntest\n")
testPrediction = clfSVM.predict(trainFeatures[4*singleGroupLength:])
expectedValues = trainTarget[4*singleGroupLength:]

for i in range(len(expectedValues)):
    diff = abs(testPrediction[i] - expectedValues[i])
    print(f'{diff}\t{testPrediction[i]} : {expectedValues[i]}')
