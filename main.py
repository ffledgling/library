#!/usr/bin/python

import classes
import helper

import pprint

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC as lsvc


if __name__ == '__main__':

    # Do dataset specific things here in main

    # Digit dataset
    TRAINING_DATA_PATH = 'optdigits/optdigits.tra'
    TESTING_DATA_PATH = 'optdigits/optdigits.tes'
    CLASS_LABELS = set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    # Latin Letter Dataset
    #TRAINING_DATA_PATH='../datasets/letter-recognition/letter-recognition.tra'
    #TESTING_DATA_PATH='../datasets/letter-recognition/letter-recognition.tes'
    #CLASS_LABELS = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

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

    #pprint.pprint(test.keys())
    # Create Train
    train_vectors = []
    train_labels = []
    for key in train.keys():
        train_vectors += train[key]
        train_labels += [key]*len(train[key])

    #print 'OneVsOne based on LinearSVC: %s' % OneVsOneClassifier(lsvc(random_state=0)).fit(train_vectors, train_labels).score(test_vectors, test_labels)
    #print 'OneVsRest based on LinearSVC: %s' % OneVsRestClassifier(lsvc(random_state=0)).fit(train_vectors, train_labels).score(test_vectors, test_labels)
