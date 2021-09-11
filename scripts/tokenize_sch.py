#!/usr/bin/env python3

import glob
import regex as re
import os.path
import nltk.data
from nltk.tokenize.nist import NISTTokenizer


def collapse(s):
    s = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', 'URLTOKEN', s)
    s = s.replace('...', 'ELIPSISTOKEN')
    return s

def main():
    nist = NISTTokenizer()
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for fn in glob.glob('../data/hmong/sch_corpus2_raw/sch-*.txt'):
        fnout = '../data/hmong/sch_corpus2_tok/' + os.path.basename(fn)
        with open(fn, encoding='utf-8') as f, open(fnout, 'w', encoding='utf-8') as fout:
            for sent in sent_detector.tokenize(collapse(f.read())):
                print(' '.join(nist.tokenize(sent)), file=fout)

if __name__ == '__main__':
    main()
