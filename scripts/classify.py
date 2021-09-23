import sys

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

sys.path.append("..")

import pandas as pd
import numpy as np
import csv
from libraries.hmong_rpa.rpa_regex import RPA_SYLLABLE as rpa
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.metrics import precision_score, recall_score
import pydot, graphviz
import re
from scipy.stats import describe
import epitran, panphon

# Note:
# this script contains the same functions as classify.ipynb
# it's just a more streamlined version of it.

df = pd.read_csv("../scripts/elabs_extracted.csv", quoting=csv.QUOTE_ALL)
all_syllables = df["word1"].tolist() + df["word2"].tolist() + df["word4"].tolist()
onsets, rhymes, tones = Counter(), Counter(), Counter()
for syl in all_syllables:
    m = rpa.match(syl)
    ons, rhy, ton = m.group("ons"), m.group("rhy"), m.group("ton")

    onsets[ons] += 1
    rhymes[rhy] += 1
    tones[ton] += 1

def add_unattested_data(df0):
    df = df0.copy()
    # for each ABAC phrase, add ACAB phrase
    unique_order_indices = []
    for i, (word1, word2, word3, word4) in df.iterrows():
        other_order = df[(df.word1 == word1) & (df.word2 == word4) & (df.word4 == word2)]
        if len(other_order) == 0:
            unique_order_indices.append(i)
    df['attested'] = True
    unattested = df.rename(columns={'word2': 'word4', 'word4': 'word2'}).iloc[unique_order_indices]
    unattested['attested'] = False
    print(f'len of attested {len(df)}, len of unattested {len(unattested)}')
    return df.append(unattested, ignore_index=True)


def add_onehot_features(df0, features='ton'):
    df = df0.copy()
    # for each of the three (w1 and w3 identical) words, add onehot vector of each tone
    if 'panphon' in features:
        epi = epitran.Epitran('hmn-Latn')
        ft = panphon.FeatureTable()
        # ft.bag_of_features(epi.transliterate('ntshoob'))

    for i in (1, 2, 4):
        if 'ton' in features:
            for ton in tones:
                if ton == '':
                    df[f'w{i}_ton_0'] = df[f'word{i}'].apply(lambda syl: rpa.match(syl).group("ton") == '')
                elif ton == 'd':
                    continue
                elif ton == 'm':
                    df[f'w{i}_ton_m'] = df[f'word{i}'].apply(lambda syl: syl.endswith('m') or syl.endswith('d'))
                else:
                    df[f'w{i}_ton_{ton}'] = df[f'word{i}'].str.endswith(ton)
        if 'rhy' in features:
            for rhy in rhymes:
                df[f'w{i}_rhy_{rhy}'] = df[f'word{i}'].apply(lambda syl: rpa.match(syl).group("rhy") == rhy)
        if 'ons' in features:
            for ons in onsets:
                df[f'w{i}_ons_{ons}'] = df[f'word{i}'].apply(lambda syl: rpa.match(syl).group("ons") == ons)
        if 'panphon' in features:
            wordi_feats = df[f'word{i}'].apply(lambda syl: ft.bag_of_features(epi.transliterate(syl)))
            panphon_names = [f'w{i}_{sign}{n}' for n in ft.names for sign in ('+', '0', '-')]

            df = pd.merge(
                df,
                pd.DataFrame(wordi_feats.tolist(), index=df.index, columns=panphon_names),
                left_index=True, right_index=True)

    return df


def train_test(clf, X, y):
    accs = []
    for rnd in range(50):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rnd)
        clf_train = clf.fit(X_train, y_train)
        accs.append(clf_train.score(X_test, y_test))

    print(describe(accs))


def select_features_from_model(clf, X, y, method='from_model'):
    clf_best = None
    best_accs = 0
    if clf.__class__.__name__ == 'SVC':
        num_features_to_try = (260, 240, 220, 200, 180, 160, 140, 120, 100)
    else:
        num_features_to_try = (120, 110, 100, 90, 80, 70, 60, 50, 40, 35, 30, 25, 20, 15, 10)
    for t in num_features_to_try:
        if method == 'from_model':
            model = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features=t)
            X_new = model.transform(X)
            #     print(f"new feature set has {X_new.shape[1]} features")
        elif method == 'chi2':
            if t > X.shape[1]: continue
            X_new = SelectKBest(chi2, k=t).fit_transform(X, y)
        else:
            raise NotImplementedError

        print(X_new.shape[1], end=', ')
        clf_new = DecisionTreeClassifier(criterion='entropy', random_state=0).fit(X_new, y)
        #     print(f"full training accuracy is {clf_new.score(X_new, y):.3f}")
        print(f"{clf_new.score(X_new, y):.3f}", end=', ')

        accs = []
        prns = []
        rcls = []
        for rnd in range(10):
            X_new_train, X_new_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=rnd)
            clf_train = DecisionTreeClassifier(criterion='entropy', random_state=rnd).fit(X_new_train, y_train)
            accs.append(clf_train.score(X_new_test, y_test))
            y_predict = clf_train.predict(X_new_test)
            prns.append(precision_score(y_test, y_predict))
            rcls.append(recall_score(y_test, y_predict))
        #     print(f"average test accuracy is {np.mean(accs):.3f}")
        #     print("="*40)
        mean_accs = np.mean(accs)
        mean_prns = np.mean(prns)
        mean_rcls = np.mean(rcls)
        if mean_accs > best_accs:
            best_accs = mean_accs
            clf_best = clf_new
        print(f"{mean_accs:.3f}, prn {mean_prns:.3f}, rcl {mean_rcls:.3f}")

    return clf_best


def visualize_tree(feature_names, use_features, clf):
    if clf.__class__.__name__ != 'DecisionTreeClassifier':
        print("can't visualize a classifier that is not a decision tree")
        return
    class_names = ['FAKE', 'ATT']

    d = 15  # max depth. Use None if unlimited

    fname = f'../out/tree_{use_features}_{d or ""}.dot'
    export_graphviz(clf,
                    out_file=fname,
                    impurity=False,
                    feature_names=feature_names,
                    class_names=class_names,
                    max_depth=d)

    f = pydot.graph_from_dot_file(fname)[0].to_string()
    # print(len(f), '\n', f[:1000])
    # f = re.sub(r'(\\nvalue = \[.*?\])', '', f)  # get rid of nvalue = [anychar, lazy]
    # f = f.replace(' <= 0.5', '?')  # change decision to a question
    # f = f.replace('headlabel="True"', 'headlabel="No"')  # change to yes no rather than <=0.5 true false
    # f = f.replace('headlabel="False"', 'headlabel="Yes"')
    # f = f.replace(R'samples = 1\nclass = ', R'***\n')  # change text of leaf node
    # print("============================")
    # print(len(f), '\n', f[:1000])

    with open(fname, 'w') as file:
        file.write(f)
    graphviz.render('dot', 'png', fname)


def main(use_features, clfs=None):
    print("="*40, use_features, "="*40)
    expanded_df = add_onehot_features(add_unattested_data(df),
                                      features=use_features).drop(columns=['word1', 'word2', 'word3', 'word4'])
    X = expanded_df.drop(columns=['attested']).to_numpy()
    y = expanded_df['attested'].to_numpy()

    for (clf_name, clf) in clfs.items():
        print("=====> ", clf_name)
        clf = clf.fit(X, y)
        print('training accuracy', clf.score(X, y))
        train_test(clf, X, y)  # no feature selection
        # select_features_from_model(clf, X, y, method='from_model')
        select_features_from_model(clf, X, y, method='chi2')
        visualize_tree(expanded_df.columns.to_list()[1:], use_features, clf)


if __name__ == '__main__':
    clfs = {
        'DecisionTree': DecisionTreeClassifier(criterion='entropy', random_state=0),
        # 'NaiveBayes': GaussianNB(),
        # 'MLP': MLPClassifier(early_stopping=True),
        # 'SVM': SVC(),
    }
    # main('ton', clfs)
    main('ons_rhy_ton', clfs)
    # main('ons_rhy_ton_panphon', clfs)