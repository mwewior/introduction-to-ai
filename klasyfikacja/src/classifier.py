from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from dataset import DataSet


clfTree = DecisionTreeClassifier(
    criterion="entropy", splitter="best", max_depth=6
)

clfSVM = SVC(
    C=1, kernel="linear", tol=10e-16  # , max_iter=int(10e4)
)

ds = DataSet(K=5)

trainFeatures = ds.joinedTrainFeatures
trainTarget = ds.joinedTrainTarget

scoresTree = cross_val_score(clfTree, trainFeatures, trainTarget, cv=5)
scoresSVM = cross_val_score(clfSVM, trainFeatures, trainTarget, cv=5)
print(f"\nDecision Tree: {scoresTree}\n{10*' '}SVM: {scoresSVM}\n")

clfSVM.fit(trainFeatures, trainTarget)

test_prediction = clfSVM.predict(ds.testFeatures)
testTarget = ds.testTarget

for i in range(len(testTarget)):
    print(f'{test_prediction[i]} : {testTarget[i]}')
