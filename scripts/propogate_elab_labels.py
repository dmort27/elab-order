#!/usr/bin/env python3

import csv
import glob
import os.path
from collections import deque

from tqdm import tqdm

def read_csv(fnin):
    with open(fnin) as f:
        return {tuple(e) for e in csv.reader(f)}

def write_tagged_span(fout, buffer, count):
    b_token = buffer.popleft()
    print(f'{b_token}\tB', file=fout)
    for i_token in buffer:
        print(f'{i_token}\tI', file=fout)
    buffer.clear()
    
def write_outside_token(fout, token):
    print(f'{token}\tO', file=fout)

def tag_file(fn, fnout, elabs):
    buffer = deque()
    count =0
    with open(fn) as fin, open(fnout, 'w') as fout:
        for line in fin:
            token = line.strip().split()
            assert type(token) is list
            token = "" if not token else token[0]
            buffer.append(token)
            if len(buffer) > 4:
                token = buffer.popleft()
                if token:
                    write_outside_token(fout, token)
                else:
                    print('', file=fout)
            if tuple(buffer) in elabs:
                write_tagged_span(fout, buffer, count)
                count += 1
        for token in buffer:
            if token:
                write_outside_token(fout, token)
            else:
                print('', file=fout)

def main(elabs_filename, input_dir, output_dir):
    elabs = read_csv(elabs_filename)
    input_filenames = glob.glob(os.path.join(input_dir, '*.conll'))
    for fn in tqdm(input_filenames):
        root, ext = os.path.splitext(os.path.basename(fn))
        fnout = os.path.join(output_dir, root + '.conll')
        tag_file(fn, fnout, elabs)

if __name__ == '__main__':
    main('../data/hmong/extracted_elabs/elabs_extracted.csv',
    '../data/hmong/sch_corpus2_conll', 
    '../data/hmong/sch_corpus2_elab')