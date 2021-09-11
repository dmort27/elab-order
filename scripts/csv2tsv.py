#!/usr/bin/env python

import sys
import csv

def main(fnin, fnout):
    with open(fnin) as fin, open(fnout, 'w') as fout:
        reader = csv.reader(fin, dialect='excel')
        writer = csv.writer(fout, dialect='excel-tab')
        for record in reader:
            writer.writerow(record)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])