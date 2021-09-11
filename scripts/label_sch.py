#!/usr/bin/env python3

import glob
import os.path

ELABS_FILE = '../data/hmong/extracted_elabs/elabs_extracted.csv'
SCH_CORPUS = '../data/hmong/sch_corpus_tok/'
SCH_CORPUS_TAGGED = '../data/hmong/sch_corpus_tag/'

def main():
    sch_filenames = glob.glob(SCH_CORPUS + '*.txt')
    for sch_filename in sch_filenames:
        tag_filename = SCH_CORPUS_TAGGED + os.path.basename(sch_filename)

if __name__ == '__main__':
    main()