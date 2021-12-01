import scipy
from libraries.elab_data import ElaborateExpressionData
from libraries.classification_exp import ClassificationExperiment
from libraries.hmong_rpa.rpa_regex import RPA_SYLLABLE
from libraries.lahu_jam.lahu_jam_regex import LAHU_REGEX
from libraries.chinese.pinyin_regex import PINYIN_SYLLABLE
from libraries.chinese.mc_regex import LIRONG_SYLLABLE
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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
    'SVM': SVC()
}

def visualize_tree(fname, data: ElaborateExpressionData, exp: ClassificationExperiment, d=15):
    from sklearn.tree import export_graphviz
    import pydot, graphviz

    clf = exp.get_clf()
    if clf.__class__.__name__ != 'DecisionTreeClassifier':
        print('cannot generate visualization for this classifier')

    clf.fit(*data.get_Xy_data())
    feature_names = data.get_feature_names()
    class_names = ['FAKE', 'ATT']

    export_graphviz(clf,
                    out_file=fname,
                    impurity=False,
                    feature_names=feature_names,
                    class_names=class_names,
                    max_depth=d)

    f = pydot.graph_from_dot_file(fname)[0].to_string()
    with open(fname, 'w') as file:
        file.write(f)
    graphviz.render('dot', 'png', fname)


def get_data(lang_id, features='ton_rhy_ons', remove_dup_ordered=None, wv_model_name=''):
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
                                   wv_model_name=wv_model_name)

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
    for _ in range(1 if remove_dup_ordered is None else num_repeats):
        data = get_data(lang_id, features=features, remove_dup_ordered=remove_dup_ordered, wv_model_name=wv_model_name)
        for _ in range(50):
            rule_acc = data.rule_based_classification()
            rule_accs.append(rule_acc)
            if rule_acc == -1: break
        X, y, orig_index = data.get_Xy_data(return_orig_index=True)
        if classifier is None:
            classifier = CLASSIFIERS['DT']
        exp = ClassificationExperiment(classifier, X, y, orig_index)
        if wv_model_name and classifier.__class__.__name__ == 'SVC':
            fs_method = 'none'
        else:
            fs_method = 'from_model' if wv_model_name else 'chi2'
        pred_accs.append(exp.select_features_from_model(method=fs_method))
        if vis_tree:
            d = 15
            visualize_tree(f'../out/Pinyin_{features}_{d or ""}.dot', data, exp, d=d)
    print("*" * 80)
    print('RULES', describe(rule_accs))
    print('PRED', describe(pred_accs))
    print("*" * 80)

if __name__ == '__main__':
    # 'ons_rhy_ton'
    # 'ton'
    # 'rhy'

    run_unique_cc_exp(LANGUAGES['Hmong'], 'wv', classifier=CLASSIFIERS['SVM'],
                  remove_dup_ordered=None, vis_tree=False, wv_model_name='grpd1_nochar_swap_run1')
    run_unique_cc_exp(LANGUAGES['Hmong'], 'wv', classifier=CLASSIFIERS['DT'],
                  remove_dup_ordered=None, vis_tree=False, wv_model_name='grpd1_nochar_swap_run1')

