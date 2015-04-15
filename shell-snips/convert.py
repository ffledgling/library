#!/usr/bin/python
import sys

def usage():
    print 'USAGE: %s FILE_TO_CONVERT' % sys.argv[0]
    print 'Input format: CSV file with the columns as follows: LABEL,Feature1,Feature2,...'
    print 'Output format: The format accepted by `svm-train` from the svm-tools package'


if len(sys.argv) !=2:
    sys.exit(1)
filename = sys.argv[1]
with open(filename, 'r') as f:
    for line in f:
        l = line.split(',')
        print ord(l[0])-65,
        for i,v in enumerate(l[1:]):
            print '%s:%s' % (i,v),
