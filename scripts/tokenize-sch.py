#!/usr/bin/env python

import glob
import argparse
import os.path
from tqdm import tqdm
from time import sleep
from nltk.tokenize import sent_tokenize, punkt, word_tokenize

def load_corpus_as_string(path):
    path = os.path.join(path, '*.txt')
    files = glob.glob(path)
    print('Reading corpus for training...')
    texts = [open(f).read() for f in files]
    print('Done.')
    return '\n'.join(texts)

def main(path, output_path):
    corpus = load_corpus_as_string(path)
    tokenizer = punkt.PunktSentenceTokenizer()
    print('Training....')
    tokenizer.train(corpus)
    print('Done.')
    for fn in tqdm(glob.glob(os.path.join(path, '*.txt'))):
        root, ext = os.path.splitext(os.path.basename(fn))
        fnout = os.path.join(output_path, root + ".conll")
        with open(fn) as f, open(fnout, 'w') as g:
            sentences = tokenizer.sentences_from_text(f.read())
            tok_sentences = [word_tokenize(s) for s in sentences]
            for sentence in tok_sentences:
                for word in sentence:
                    print(word, file=g)
                print('\n', file=g)

if __name__ == '__main__':
    main('data/hmong/sch_corpus2_raw', 'data/hmong/sch_corpus2_conll')

