import pandas as pd
import re
import wandb
from scipy.stats import wilcoxon

api = wandb.Api()
runs = api.runs("cuichenx/hmong-seq-tagging")


def get_values(root_name, column="test_full/FB1", max_match=9):
    regex = re.compile(fr"^(grpd[1-3]_{root_name}_run[1-3])$")
    res = {}
    for run in runs:
        if regex.match(run.name):
            res[run.name] = run.summary[column]
        if len(res) == max_match:
            break
    return res

def p_value(smaller, larger):
    smaller_val = [v for k, v in sorted(smaller.items())]
    larger_val = [v for k, v in sorted(larger.items())]
    return wilcoxon(smaller_val, larger_val, alternative='less')[1]


if __name__ == '__main__':
    word_dim = 50
    nochar = get_values(f"nochar_wd{word_dim}_Clf_tr0.10")
    phnms = get_values(f"phnms_cd30_wd{word_dim}_Clf_tr0.10")
    tones = get_values(f"tones_cd4_wd{word_dim}_Clf_tr0.10")
    phnms_ac = get_values(f"phnms_aftercnn_cd30_wd{word_dim}_Clf_tr0.10")
    tones_ac = get_values(f"tones_aftercnn_cd4_wd{word_dim}_Clf_tr0.10")

    print("phnms, one-sided wilxocon", p_value(nochar, phnms))
    print("tones, one-sided wilxocon", p_value(nochar, tones))
    print("phnms_ac, one-sided wilxocon", p_value(nochar, phnms_ac))
    print("tones_ac, one-sided wilxocon", p_value(nochar, tones_ac))
