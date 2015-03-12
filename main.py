import sklearn
import sklearn.svm as svm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import itertools
import collections
import pprint

class Class(object):
    # Assume that all samples in the class have same number of features
    # TODO: enforce this assumption
    def __init__(self):
        self.multiclass_label = None
        self.binary_label = None
        self.samples = []

    def add_sample(sample):
        self.samples.append(sample)

    def add_samples(samples):
        if type(samples) is not type([]):
            raise Exception
        self.samples += sample

class Metrics(object):

    def __init__(self, value=None, accuracy=None, balance=None, overlap=None, margin=None, partition=None):
        self.accuracy = accuracy
        self.overlap = overlap
        self.margin = margin
        self.partition = partition

        self.balance = 2.0*min(len(partition[0]), len(partition[1]))/(len(partition[0]) + len(partition[1]))
        self.value = objective_function(accuracy=accuracy, balance=balance, overlap=overlap, margin=margin)

    def __repr__(self):
        return ('(Value: %s' ' Accuracy: %s, Balance: %s, Overlap: %s, Margin: %s, Partition: %s)'
                % (repr(self.value),  repr(self.accuracy),  repr(self.balance),  repr(self.overlap),  repr(self.margin),  repr(self.partition)))


def objective_function(accuracy=None, balance=None, overlap=None, margin=None):
    # Take all the different things we measure and create a single unifying function
    # use the value of this function to decide if something a good choice
    # Values returned should always be b/w 0 and 1.

    value = 1.0

    if accuracy:
        value *= accuracy
    if balance:
        value *= balance
    if overlap:
        value *= overlap
    if margin:
        value *= margin

    return value

def get_statistics(results):
    # Takes a list of 'Metrics' as input
    # Outputs:
    #   Best value
    #   Maximum accuracy
    #   Average Value
    #   Average Accuracy

    max_val=0
    max_acc=0
    sum_val=0
    sum_acc=0
    size = len(results)

    for metric in results:
        if metric.accuracy > max_acc:
            max_acc = metric.accuracy
        if metric.value > max_val:
            max_val = metric.value

        sum_val += metric.value
        sum_acc += metric.accuracy

    print 'Best Accuracy: %s' % max_acc
    print 'Highest Value: %s' % max_val
    print 'Average Accuracy: %s' % str(sum_acc/size)
    print 'Average Value: %s' % str(sum_val/size)


def create_set(label_list):
    d = {}
    for key in label_list:
        d[key] = []

    return d

def populate_dataset(dataset, filepath):
    # Assumes last value in CSV is class label
    # TODO: make this more flexible
    with open(filepath, 'r') as f:
        for line in f:
            datum = line.rstrip().strip().split(',')
            feature_vector, category = tuple([int(x) for x in datum[:64]]), int(datum[64])
            dataset[category].append(feature_vector)

    return dataset

def class_partitions(class_labels):

    # The frozenset method is a little hacky for duplicate elimination, optimize later
    splits = set()
    count = 0

    for split in itertools.combinations(CLASS_LABELS, len(CLASS_LABELS)/2):
        count += 1
        splits.add(frozenset((frozenset(split),frozenset(CLASS_LABELS - set(split)))))

    return splits


def bruteforce_SVM(train, test, class_labels, linear_kernel=True):
    # Generates all possible 2-set eqipartitions from the class labels
    # Computes the SVM, linear or otherwise, for each of these partitions
    # returns the results

    partition_list = class_partitions(class_labels)
    results = []

    for count, partition in enumerate(partition_list):
        partition = tuple(partition)
        print '#', count
        twoclass_train_data = []
        twoclass_train_label = []
        twoclass_test_data = []
        twoclass_test_label = []

        # Process for first set in parition
        for label in partition[0]:
            # Setup training data to feed SVM
            twoclass_train_data += train[label]
            twoclass_train_label += [0]*len(train[label]) # add labels for each of the data points to the label list
            # Setup testing data for validation
            twoclass_test_data += test[label]
            twoclass_test_label += [0]*len(test[label]) # add labels for each of the data points to the label list

        # Process for second set in parition
        for label in partition[1]:
            # Setup training data to feed SVM
            twoclass_train_data += train[label]
            twoclass_train_label += [1]*len(train[label]) # add labels for each of the data points to the label list
            # Setup testing data for validation
            twoclass_test_data += test[label]
            twoclass_test_label += [1]*len(test[label]) # add labels for each of the data points to the label list


        # Train the SVM here
        if linear_kernel:
            clf = svm.LinearSVC()
        else:
            clf = svm.SVC()
        clf.fit(twoclass_train_data, twoclass_train_label)
        #print clf
        current_metric = Metrics(accuracy=clf.score(twoclass_test_data, twoclass_test_label), partition=partition)

        results.append(current_metric)

    return results


def pairwise_SVM_A1(test, train, class_labels):
    # Take pairwise classes
    # Compute SVM for that class
    # Then put each class in one of the two major classes
    # Then recompute accuracy on training set.
    # do so until all classes are done.
    # Compute the accuracy on the training set, keep track of results, find best result

    count = 0
    results = []

    # For every possible pair of classes
    for initial_two in itertools.combinations(class_labels, 2):
        #print '#%s' % count

        #print 'Initial Two labels of choice are:', initial_two
        remaining_labels = set(class_labels - set(initial_two))
        #print 'remaining labels are: %s' % remaining_labels
        mapping = {0: {initial_two[0]}, 1: {initial_two[1]}}
        clf = svm.LinearSVC()
        # Train classifier based on initial two classes.
        # extract data points from the train object, generate labels on the fly
        clf.fit(train[initial_two[0]] + train[initial_two[1]],
                [0]*len(train[initial_two[0]]) + [1]*len(train[initial_two[1]]))

        for label in remaining_labels:
            score = clf.score(train[label], [0]*len(train[label]))
            # This is interesting to print, and observe actual values
            #print 'scoring (similarity to %s:' % initial_two[0], label, score
            if score > 0.5:
                mapping[0].add(label)
            else:
                mapping[1].add(label)
        #print 'Optimal paritioning is:', mapping

        # Create a new SVC
        clf = svm.LinearSVC()

        # Formulate training set and labels
        train_samples, train_labels = [], []
        for label in mapping[0]:
            train_samples += train[label]
            train_labels += [0]*len(train[label])
        for label in mapping[1]:
            train_samples += train[label]
            train_labels += [1]*len(train[label])

        # Train the new SVC, based on decided split
        clf.fit(train_samples, train_labels)

        # Generate reverse mapping
        reverse_mapping = {}
        for binary_label, labels in mapping.items():
            for label in labels:
                reverse_mapping[label] = binary_label

        # Test it
        test_samples, test_labels = [], []
        for label in class_labels:
            test_samples += test[label]
            test_labels += [reverse_mapping[label]]*len(test[label])

        #print 'Testing the re-trained optimal partition, score:',
        current_metric = Metrics(accuracy=clf.score(test_samples, test_labels),
                                 partition=tuple(mapping.itervalues()))

        print current_metric
        results.append(current_metric)
        count += 1

    return results


if __name__ == '__main__':

    # Do dataset specific things here in main

    TRAINING_DATA_PATH='optdigits/optdigits.tra'
    TESTING_DATA_PATH='optdigits/optdigits.tes'
    CLASS_LABELS = set(range(0,10)) # Class labels are b/w 0..9 (inclusive)

    # create training set
    train = create_set(CLASS_LABELS)
    #pprint.pprint(train)
    populate_dataset(train, TRAINING_DATA_PATH)
    #pprint.pprint(train)

    # create testing set
    test = create_set(CLASS_LABELS)
    #pprint.pprint(train)
    populate_dataset(test, TESTING_DATA_PATH)
    #pprint.pprint(train)

    #results = bruteforce_SVM(train, test, CLASS_LABELS, linear_kernel=True)
    #results = bruteforce_SVM(train, test, CLASS_LABELS, linear_kernel=False)
    results = pairwise_SVM_A1(train, test, CLASS_LABELS)

    pprint.pprint(results)
    get_statistics(results)

