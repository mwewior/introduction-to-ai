from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from dataset import DataSet


clfTree = DecisionTreeClassifier(
    criterion="entropy", splitter="best", max_depth=6
)

clfSVC = SVC(
    C=1, kernel="linear", max_iter=int(10e4), tol=10e-16
)

ds = DataSet(K=5)

trainFeatures = ds.trainFeatures
trainTarget = ds.trainTarget

scoresTree = cross_val_score(clfTree, trainFeatures, trainTarget, cv=5)
scoresSVC = cross_val_score(clfSVC, trainFeatures, trainTarget, cv=5)
print(f"\n{scoresTree}\n\n{scoresSVC}\n")
