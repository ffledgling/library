#!/usr/bin/python

import classes
import helper

import argparse
import os
import pprint
import sys

import numpy as np
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC as lsvc
from sklearn import preprocessing
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.neighbors import KNeighborsClassifier, KDTree, BallTree
from sklearn.gaussian_process import GaussianProcess
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':

    # Digit dataset
    #TRAINING_DATA_PATH = 'optdigits/optdigits.tra'
    #TESTING_DATA_PATH = 'optdigits/optdigits.tes'
    # CLASS_LABELS_PATH = None
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
    le = preprocessing.LabelEncoder()
    le.fit(list(CLASS_LABELS))

    # create training set
    train = helper.create_set(CLASS_LABELS)
    helper.populate_dataset(train, TRAINING_DATA_PATH)
    print 'Done populating Training dataset'

    # create testing set
    test = helper.create_set(CLASS_LABELS)
    helper.populate_dataset(test, TESTING_DATA_PATH)
    print 'Done populating Test dataset'

    test_vectors = []
    test_labels = []
    for key in test.keys():
        test_vectors += test[key]
        l = [key]*len(test[key])
        test_labels += list(le.transform(l))
    print 'Done generating test vectors and labels'

    # Create Train
    train_vectors = []
    train_labels = []
    for key in train.keys():
        train_vectors += train[key]
        l = [key]*len(train[key])
        train_labels += list(le.transform(l))
    print 'Done generating train vectors and labels'

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    train_vectors = np.array(train_vectors)
    test_vectors = np.array(test_vectors)

    print 'OneVsOne based on LinearSVC: %s' % OneVsOneClassifier(lsvc(random_state=0)).fit(train_vectors, train_labels).score(test_vectors, test_labels)
    print 'OneVsRest based on LinearSVC: %s' % OneVsRestClassifier(lsvc(random_state=0)).fit(train_vectors, train_labels).score(test_vectors, test_labels)
    print 'LDA: %s' % LDA().fit(train_vectors, train_labels).score(test_vectors, test_labels)
    print 'QDA: %s' % QDA().fit(train_vectors, train_labels).score(test_vectors, test_labels)
    print 'KNN: %s' % KNeighborsClassifier().fit(train_vectors, train_labels).score(test_vectors, test_labels)
    #print 'Gaussian Process: %s' % GaussianProcess().fit(train_vectors, train_labels).score(test_vectors, test_labels)
    print 'Random Forest: %s' % RandomForestClassifier().fit(train_vectors, train_labels).score(test_vectors, test_labels)

