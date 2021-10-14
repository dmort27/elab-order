from typing import Union

from scipy.stats import describe
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.base import clone
import numpy as np

class ClassificationExperiment:
    def __init__(self, clf: Union[DecisionTreeClassifier, SVC], X, y):
        self.clf = clf
        self.X = X
        self.y = y
        self.clf.fit(X, y)

    def get_full_train_acc(self):
        return self.clf.score(self.X, self.y)

    def train_test(self, X=None, y=None, clf=None, num_runs=50, test_size=0.3):
        if X is None: X = self.X
        if y is None: y = self.y
        if clf is None: clf = self.clf
        accs = []
        prns = []
        rcls = []
        for rnd in range(num_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rnd)
            clf_train = clf.fit(X_train, y_train)
            accs.append(clf_train.score(X_test, y_test))
            y_predict = clf_train.predict(X_test)
            prns.append(precision_score(y_test, y_predict))
            rcls.append(recall_score(y_test, y_predict))

        return accs, prns, rcls

    def select_features_from_model(self, method='from_model'):
        clf_best = None
        best_accs = 0
        if self.clf.__class__.__name__ == 'SVC':
            num_features_to_try = (160, 140, 120, 100, 80, 70, 60)
        else:
            num_features_to_try = (120, 110, 100, 90, 80, 70, 60, 50, 40, 35, 30, 25, 20, 15, 10, 8, 6)
        for t in num_features_to_try:
            if method == 'from_model':
                model = SelectFromModel(self.clf, prefit=True, threshold=-np.inf, max_features=t)
                X_new = model.transform(self.X)
            elif method == 'chi2':
                if t > self.X.shape[1]: continue
                X_new = SelectKBest(chi2, k=t).fit_transform(self.X, self.y)
            else:
                raise NotImplementedError

            print(X_new.shape[1], end=', ')
            clf_new = clone(self.clf).fit(X_new, self.y)
            print(f"{clf_new.score(X_new, self.y):.3f}", end=', ')

            accs, prns, rcls = self.train_test(X=X_new, clf=clf_new, num_runs=10)

            mean_accs = np.mean(accs)
            mean_prns = np.mean(prns)
            mean_rcls = np.mean(rcls)
            if mean_accs > best_accs:
                best_accs = mean_accs
                clf_best = clf_new
            print(f"{mean_accs:.3f}, prn {mean_prns:.3f}, rcl {mean_rcls:.3f}")

        return best_accs






