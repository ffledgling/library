#!/usr/bin/python

import classes
import helper

import argparse
import os
import pprint
import sys

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC as lsvc


if __name__ == '__main__':

    # Do dataset specific things here in main

    # Digit dataset
    #TRAINING_DATA_PATH = 'optdigits/optdigits.tra'
    #TESTING_DATA_PATH = 'optdigits/optdigits.tes'
    #CLASS_LABELS_PATH = None
    #CLASS_LABELS = set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    # Latin Letter Dataset
    TRAINING_DATA_PATH='../datasets/letter-recognition/letter-recognition.tra'
    TESTING_DATA_PATH='../datasets/letter-recognition/letter-recognition.tes'
    CLASS_LABELS_PATH = None
    CLASS_LABELS = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

    # Arg Handling
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train-file', help='Path to the training file.')
    parser.add_argument('-p', '--test-file', help='Path to the test file.')
    parser.add_argument('-l', '--label-file',
                        help='Path to the file containing class labels.\n'
                              'Should match the labels in the testing and training files.\n'
                              'Each label on a new line.\n')
    parser.add_argument('-a', '--approach', help='Choose approach for partitioning. NOT IMPLEMENTED YET.')
    args = parser.parse_args()

    if args.train_file or args.test_file or args.label_file:
        if not args.train_file or not args.test_file or not args.label_file:
            parser.error('When specifying a custom data set, please specify all of:\n'
                    '(1) Training file path\n(2) Test file path\n(3) Label file path')
        else:
            if not os.path.exists(args.train_file):
                parser.error('Train file specified by "%s" does not exist' % args.train_file)
            else:
                TRAINING_DATA_PATH = args.train_file

            if not os.path.exists(args.test_file):
                parser.error('Train file specified by "%s" does not exist' % args.train_file)
            else:
                TESTING_DATA_PATH = args.test_file

            if not os.path.exists(args.label_file):
                parser.error('Train file specified by "%s" does not exist' % args.train_file)
            else:
                CLASS_LABELS_PATH = args.label_file


    # create class label list
    CLASS_LABELS = helper.read_labels(CLASS_LABELS_PATH) or CLASS_LABELS

    # create training set
    train = helper.create_set(CLASS_LABELS)
    helper.populate_dataset(train, TRAINING_DATA_PATH)

    # create testing set
    test = helper.create_set(CLASS_LABELS)
    helper.populate_dataset(test, TESTING_DATA_PATH)

    x = classes.TreeNode(train, CLASS_LABELS)
    test_vectors = []
    test_labels = []
    for key in test.keys():
        test_vectors += test[key]
        test_labels += [key]*len(test[key])
    x._score(test_vectors, test_labels)
    print x

    print x.classifier.__dict__.keys()
    pprint.pprint(x.classifier.coef_)
    print x.classifier._enc.classes_
    print x.classifier.coef_.shape
    print test_labels[0]
    print x.class_labels
    pprint.pprint(helper.exportTreeToJSON(x.returnDictRepr()))
