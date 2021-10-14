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


def get_data(lang_id, features='ton_rhy_ons', remove_dup_ordered=None):
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
                                   features=features, remove_dup_ordered=remove_dup_ordered)
    rule_acc = data.rule_based_classification()
    X, y = data.get_Xy_data()
    return X, y, rule_acc

def describe(arr):
    if len(arr) > 1:
        return scipy.stats.describe(arr)
    else:
        return f'One value: {arr[0]}'

def run_unique_cc_exp(lang_id, features, remove_dup_ordered=None, num_repeats=10):
    rule_accs = []
    pred_accs = []
    for _ in range(1 if remove_dup_ordered is None else num_repeats):
        X, y, rule_acc = get_data(lang_id, features=features, remove_dup_ordered=remove_dup_ordered)
        exp = ClassificationExperiment(DecisionTreeClassifier(criterion='entropy'), X, y)
        # exp = ClassificationExperiment(SVC(), X, y)
        rule_accs.append(rule_acc)
        pred_accs.append(exp.select_features_from_model(method='chi2'))
    print("*" * 80)
    print('RULES', describe(rule_accs))
    print('PRED', describe(pred_accs))
    print("*" * 80)

if __name__ == '__main__':
    # run_unique_cc_exp('cmn-Pinyin', 'ton_rhy_ons', remove_dup_ordered=None)
    run_unique_cc_exp('ltc-IPA', 'ton_rhy_ons', remove_dup_ordered=None)