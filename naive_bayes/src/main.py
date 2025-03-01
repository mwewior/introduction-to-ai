import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
try:
    from naiveBayes import NaiveBayesClassifier
except ModuleNotFoundError:
    from src.naiveBayes import NaiveBayesClassifier

try:
    import dataset
except ModuleNotFoundError:
    from src import dataset


SEED = 318407
np.random.seed(SEED)


ds_skl = dataset.DataSetSKL()


SKF = StratifiedKFold(n_splits=5)
X = ds_skl.features
Y = ds_skl.target

clfSVM = SVC(
    C=0.1, kernel="rbf", tol=10e-16, max_iter=int(25), random_state=SEED
)

SVM_acc = cross_val_score(clfSVM, X, Y, cv=SKF, scoring="accuracy")
SVM_pre = cross_val_score(clfSVM, X, Y, cv=SKF, scoring="precision_weighted")
SVM_rec = cross_val_score(clfSVM, X, Y, cv=SKF, scoring="recall_weighted")
SVM_f1 = cross_val_score(clfSVM, X, Y, cv=SKF, scoring="f1_weighted")

SVM_scores = [SVM_acc, SVM_pre, SVM_rec, SVM_f1]


clfTree = DecisionTreeClassifier(
    criterion="entropy", splitter="best", max_depth=4, random_state=SEED
)

TREE_acc = cross_val_score(clfTree, X, Y, cv=SKF, scoring="accuracy")
TREE_pre = cross_val_score(clfTree, X, Y, cv=SKF, scoring="precision_weighted")
TREE_rec = cross_val_score(clfTree, X, Y, cv=SKF, scoring="recall_weighted")
TREE_f1 = cross_val_score(clfTree, X, Y, cv=SKF, scoring="f1_weighted")

TREE_scores = [TREE_acc, TREE_pre, TREE_rec, TREE_f1]

clfGNB = GaussianNB()

GNB_acc = cross_val_score(clfGNB, X, Y, cv=SKF, scoring="accuracy")
GNB_pre = cross_val_score(clfGNB, X, Y, cv=SKF, scoring="precision_weighted")
GNB_rec = cross_val_score(clfGNB, X, Y, cv=SKF, scoring="recall_weighted")
GNB_f1 = cross_val_score(clfGNB, X, Y, cv=SKF, scoring="f1_weighted")

GNB_scores = [GNB_acc, GNB_pre, GNB_rec, GNB_f1]


clfBayes = NaiveBayesClassifier()

BAYES_acc = cross_val_score(clfBayes, X, Y, cv=SKF, scoring="accuracy")
BAYES_pre = cross_val_score(clfBayes, X, Y, cv=SKF, scoring="precision_weighted")  # noqa
BAYES_rec = cross_val_score(clfBayes, X, Y, cv=SKF, scoring="recall_weighted")
BAYES_f1 = cross_val_score(clfBayes, X, Y, cv=SKF, scoring="f1_weighted")

BAYES_scores = [BAYES_acc, BAYES_pre, BAYES_rec, BAYES_f1]


metrics = [" accuracy", "precision", "   recall", "       f1"]


print("\nSVM scores (mean +- deviation)")
for score, metric in zip(SVM_scores, metrics):
    print(f"{metric}: {100*score.mean()} +- {100*score.std()} [%]")

print("\nTREE scores (mean +- deviation)")
for score, metric in zip(TREE_scores, metrics):
    print(f"{metric}: {100*score.mean()} +- {100*score.std()} [%]")

print("\n(Sklearn Gausian NB) Naive Bayes scores (mean +- deviation)")
for score, metric in zip(GNB_scores, metrics):
    print(f"{metric}: {100*score.mean()} +- {100*score.std()} [%]")

print("\n(Own) Naive Bayes scores (mean +- deviation)")
for score, metric in zip(BAYES_scores, metrics):
    print(f"{metric}: {100*score.mean()} +- {100*score.std()} [%]")
