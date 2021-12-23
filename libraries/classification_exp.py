from typing import Union

from scipy.stats import describe
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.base import clone
import numpy as np

class ClassificationExperiment:
    def __init__(self, clf: Union[DecisionTreeClassifier, SVC], X, y, X_feature_names=None, orig_index=None):
        self.clf = clf
        self.X = X
        self.y = y
        self.X_feature_names = X_feature_names
        self.orig_index = orig_index
        self.clf.fit(X, y)

    def get_full_train_acc(self):
        return self.clf.score(self.X, self.y)

    def train_test(self, X=None, y=None, clf=None, num_runs=10, test_size=0.3):
        if X is None: X = self.X
        if y is None: y = self.y
        if clf is None: clf = self.clf
        accs = []
        prns = []
        rcls = []

        def clf_fit(X_train, X_test, y_train, y_test):
            clf_train = clf.fit(X_train, y_train)
            accs.append(clf_train.score(X_test, y_test))
            y_predict = clf_train.predict(X_test)
            prns.append(precision_score(y_test, y_predict))
            rcls.append(recall_score(y_test, y_predict))

        if self.orig_index is not None:
            gkf = GroupKFold(n_splits=num_runs)
            for train_idx, test_idx in gkf.split(X, y, groups=self.orig_index):
                clf_fit(X[train_idx], X[test_idx], y[train_idx], y[test_idx])
        else:
            # this should be deprecated
            for rnd in range(num_runs):
                clf_fit(*train_test_split(X, y, test_size=test_size, random_state=rnd))

        return accs, prns, rcls

    def select_features_from_model(self, method='from_model', verbose=False):

        clf_best = None
        best_accs = 0
        mean_accs_for_each_k = []
        if self.clf.__class__.__name__ in ('LinearSVC', 'SVC'):
            num_features_to_try = (160, 140, 120, 100, 80, 70, 60, 40, 20, 15, 10, 8)
        else:
            num_features_to_try = (120, 110, 100, 90, 80, 70, 60, 50, 40, 35, 30, 25, 20, 15, 10, 8, 6)
        for t in num_features_to_try:
            if method == 'from_model':
                select = SelectFromModel(self.clf, prefit=True, threshold=-np.inf, max_features=t)
                X_new = select.transform(self.X)
                X_new_feature_names = select.get_feature_names_out(self.X_feature_names)
            elif method in ['chi2', 'f_classif']:
                score_func = chi2 if method == 'chi2' else f_classif
                if t > self.X.shape[1]: continue
                select = SelectKBest(score_func, k=t)
                X_new = select.fit_transform(self.X, self.y)
                X_new_feature_names = select.get_feature_names_out(self.X_feature_names)
            elif method == 'none':
                X_new = self.X
                X_new_feature_names = self.X_feature_names
            else:
                raise NotImplementedError

            if verbose: print(X_new.shape[1], end=', ')
            clf_new = clone(self.clf).fit(X_new, self.y)
            if verbose: print(f"{clf_new.score(X_new, self.y):.3f}", end=', ')

            accs, prns, rcls = self.train_test(X=X_new, clf=clf_new, num_runs=10)

            mean_accs = np.mean(accs)
            mean_prns = np.mean(prns)
            mean_rcls = np.mean(rcls)
            mean_accs_for_each_k.append(mean_accs)
            if mean_accs > best_accs:
                best_accs = mean_accs
                clf_best = clf_new
            if verbose: print(f"{mean_accs:.3f}, prn {mean_prns:.3f}, rcl {mean_rcls:.3f}, MVF {X_new_feature_names}")
            # if verbose: print(f"{mean_accs:.3f}, prn {mean_prns:.3f}, rcl {mean_rcls:.3f}")
            if method == 'none':
                break
        # return best_accs
        return np.atleast_2d(mean_accs_for_each_k)

    def get_clf(self):
        return self.clf




