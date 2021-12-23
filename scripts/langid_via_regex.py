#!/usr/bin/env python3

import re
import argparse
import glob
import os
import os.path
import shutil
import tqdm

HM_RE = re.compile('''^(?P<ons>f|v|xy|x|s|z|y|h|
                    n?(?:dl|pl|tx|ts|p|t|r|c|k|q)h?|h?|d|dh|
                    h?(?:ny|n|ml|m|l)|)
                    (?P<rhy>aa|ee|oo|ai|aw|au|ia|ua|i|e|a|o|u|w)
                    (?P<ton>b|s|j|v|m|g|d|)*$''', flags=re.X)

def main(input_path, output_path, threshold):
    shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    for fn in tqdm.tqdm(glob.glob(os.path.join(input_path, '*.conll'))):
        with open(fn) as f:
            parseable = 0.0
            tokens = []
            for i, token in enumerate(f):
                tokens += token
                try:
                    form = token.split()[0]
                except IndexError:
                    continue
                if HM_RE.match(''.join(filter(lambda x: x.isalpha(), form.lower()))):
                    parseable += 1.0
        if parseable/i > threshold:
            fnout = os.path.join(output_path, os.path.basename(fn))
            with open(fnout, 'w') as f:
                for token in tokens:
                    f.write(token)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input path')
    parser.add_argument('-o', '--output', help='Output path')
    parser.add_argument('-t', '--threshold', type=float, help='Percentage of tokens that must be parseable in order for a doocument to be included.')
    args = parser.parse_args()
    main(args.input, args.output, args.threshold)