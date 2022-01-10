import numpy as np
import scipy
from sklearn.feature_selection import SelectFromModel

from libraries.elab_data import ElaborateExpressionData, HmongWordVectorsData, ChineseWordVectorsData
from libraries.classification_exp import ClassificationExperiment
from libraries.hmong_rpa.rpa_regex import RPA_SYLLABLE
from libraries.lahu_jam.lahu_jam_regex import LAHU_REGEX
from libraries.chinese.pinyin_regex import PINYIN_SYLLABLE
from libraries.chinese.mc_regex import LIRONG_SYLLABLE
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
import pandas as pd
import csv

LANGUAGES = {
    'Hmong': 'hmn-Latn',
    'Lahu': 'lhu-Latn',
    'Mandarin': 'cmn-Pinyin',
    'Middle Chinese': 'ltc-IPA',
}
CLASSIFIERS = {
    'DT': DecisionTreeClassifier(criterion='entropy'),
    'SVM': SVC(),
    'LinearSVM': LinearSVC(dual=False, C=1)
}

def visualize_tree(fname, data: ElaborateExpressionData, exp: ClassificationExperiment, d=15,
                   select_k=None):
    from sklearn.tree import export_graphviz
    import pydot, graphviz

    clf = exp.get_clf()
    if clf.__class__.__name__ != 'DecisionTreeClassifier':
        print('cannot generate visualization for this classifier')

    X, y, _, _ = data.get_Xy_data()
    if select_k is not None:
        select = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features=select_k)
        X_new = select.transform(X)
        feature_names = select.get_feature_names_out(exp.X_feature_names)
        clf.fit(X_new, y)
    else:
        feature_names = data.get_feature_names()
        clf.fit(X, y)

    class_names = ['FAKE', 'ATT']

    export_graphviz(clf,
                    out_file=fname,
                    impurity=False,
                    feature_names=feature_names,
                    class_names=class_names,
                    max_depth=d)

    # f = pydot.graph_from_dot_file(fname)[0].to_string()
    # with open(fname, 'w') as file:
    #     file.write(f)
    graphviz.render('dot', 'png', fname)


def get_data(lang_id, features='ton_rhy_ons', remove_dup_ordered=None, wv_model=None):
    if lang_id == 'hmn-Latn':
        csv_path = "../data/hmong/extracted_elabs/elabs_extracted.csv"
        regex = RPA_SYLLABLE
    elif lang_id == 'lhu-Latn':
        csv_path = "../data/lahu/elabs_from_ell/elabs_extracted.csv"
        regex = LAHU_REGEX
    elif lang_id == 'ltc-IPA':
        csv_path = "../data/chinese/extracted_coordinate_compounds/mc_lirong.csv"
        regex = LIRONG_SYLLABLE
    elif lang_id == 'cmn-Pinyin':
        csv_path = "../data/chinese/extracted_coordinate_compounds/pinyin.csv"
        regex = PINYIN_SYLLABLE
    else:
        raise NotImplementedError

    df = pd.read_csv(csv_path, quoting=csv.QUOTE_ALL)
    data = ElaborateExpressionData(df, lang_id=lang_id, syl_regex=regex, verbose=False,
                                   features=features, remove_dup_ordered=remove_dup_ordered,
                                   wv_model=wv_model)

    return data


def describe(arr):
    if len(arr) > 1:
        return scipy.stats.describe(arr)
    else:
        return f'One value: {arr[0]}'


def run_unique_cc_exp(lang_id, features, classifier=None,
                      remove_dup_ordered=None, num_repeats=10, vis_tree=False, wv_model_name=''):
    rule_accs = []
    pred_accs = []
    if wv_model_name:
        wv_model = HmongWordVectorsData(wv_model_name) if lang_id == 'hmn-Latn' else ChineseWordVectorsData(wv_model_name)
    else:
        wv_model = None
    for _ in range(1 if remove_dup_ordered is None else num_repeats):
        data = get_data(lang_id, features=features, remove_dup_ordered=remove_dup_ordered, wv_model=wv_model)
        for _ in range(50):
            rule_acc = data.rule_based_classification()
            rule_accs.append(rule_acc)
            if rule_acc == -1: break
        X, y, X_feature_names, orig_index = data.get_Xy_data()
        if classifier is None:
            classifier = CLASSIFIERS['DT']
        exp = ClassificationExperiment(classifier, X, y, X_feature_names, orig_index)

        # if wv features are used, cannot use chi2 since it only works with count features
        # if SVC is used, cannot used from_model since SVM does not return
        if wv_model_name and classifier.__class__.__name__ == 'SVC':
            fs_method = 'none'
        else:
            fs_method = 'from_model' if wv_model_name else 'chi2'
        mean_accs_for_each_k = exp.select_features_from_model(method=fs_method, verbose=('wv' in features and features!='wv'))
        acc_method = 'best_k_overall' # 'best_k_each_run'  # 'best_k_overall'
        if acc_method == 'best_k_overall':
            # average each column, then max
            best_acc = mean_accs_for_each_k.mean(axis=0).max()
        elif acc_method == 'best_k_each_run':
            # find the max in each row, then average
            best_acc = mean_accs_for_each_k.max(axis=1).mean()
        pred_accs.append(best_acc)
        if vis_tree:
            visualize_tree(f'../out/{lang_id}_{features}_paperver.dot', data, exp, d=15, select_k=None)
    print("*" * 80)
    print(lang_id, features if features!='wv' else wv_model_name,
          classifier.__class__.__name__, 'remove dup ordered:', remove_dup_ordered)
    print('RULES', describe(rule_accs))
    print('PRED', describe(pred_accs))
    print("*" * 80)

if __name__ == '__main__':
    # 'ons_rhy_ton'
    # 'ton'
    # 'rhy'
    # for remove_dup in (None, False):
    #     for m in ('SVM',):
    #         run_unique_cc_exp(LANGUAGES['Hmong'], 'wv_ons_rhy_ton', classifier=CLASSIFIERS[m], num_repeats=5,
    #                       remove_dup_ordered=remove_dup, vis_tree=False, wv_model_name='sg')
    "grpd1_nochar_noswap_tr0.10_run3"
    # for m in ('LinearSVM',):
    #     for remove_dup in (None, False, ):
    #         run_unique_cc_exp(LANGUAGES['Hmong'], 'ons_rhy_ton', classifier=CLASSIFIERS[m],
    #                           remove_dup_ordered=remove_dup, vis_tree=False, wv_model_name="", num_repeats=5)
        # for wv_model_name in  ('sg', "grpd1_nochar_noswap_tr0.10_run3"):
        #     for remove_dup in (None, False):
        #         run_unique_cc_exp(LANGUAGES['Hmong'], 'wv', classifier=CLASSIFIERS[m],
        #                       remove_dup_ordered=remove_dup, vis_tree=False, wv_model_name=wv_model_name, num_repeats=5)
        #
        # for feature in ('ton', 'ons_rhy_ton'):
        #     for remove_dup in (None, False):
        #         run_unique_cc_exp(LANGUAGES['Hmong'], feature, classifier=CLASSIFIERS[m],
        #                       remove_dup_ordered=remove_dup, vis_tree=False, wv_model_name='', num_repeats=5)
    # for m in ('DT', 'SVM'):
    #     for feat in ('rhy', 'ons_rhy_ton'):
    #         for remove_dup in (None, False,):
    #             run_unique_cc_exp(LANGUAGES['Lahu'], feat, classifier=CLASSIFIERS[m],
    #                               remove_dup_ordered=remove_dup, vis_tree=False, wv_model_name="", num_repeats=5)

    # for visualizing tree
    # run_unique_cc_exp(LANGUAGES['Hmong'], 'ton', classifier=CLASSIFIERS['DT'],
    #                   remove_dup_ordered=False, vis_tree=True, wv_model_name="", num_repeats=1)
    run_unique_cc_exp(LANGUAGES['Lahu'], 'rhy', classifier=CLASSIFIERS['DT'],
                      remove_dup_ordered=None, vis_tree=True, wv_model_name="", num_repeats=1)
    # run_unique_cc_exp(LANGUAGES['Middle Chinese'], 'ton', classifier=CLASSIFIERS['DT'],
    #                   remove_dup_ordered=False, vis_tree=True, wv_model_name="", num_repeats=1)