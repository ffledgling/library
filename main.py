import sklearn
import sklearn.svm as svm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import itertools
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
        #print set(split), CLASS_LABELS - set(split)
        count += 1
        splits.add(frozenset((frozenset(split),frozenset(CLASS_LABELS - set(split)))))

    #print len(splits)
    #print count

    return splits
        


if __name__ == '__main__':

    TRAINING_DATA_PATH='optdigits/optdigits.tra'
    TESTING_DATA_PATH='optdigits/optdigits.tes'
    CLASS_LABELS = set(range(0,10)) # Class labels are b/w 0..9 (inclusive)


    accuracy_list = []

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

    partition_list = class_partitions(CLASS_LABELS)

    for count, partition in enumerate(partition_list):
        partition = tuple(partition)
        print '#', count, partition,
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

        #print twoclass_train_data
        #print twoclass_train_label

        #print len(twoclass_train_data)
        #print len(twoclass_train_label)

        # Train the SVM here
        clf = svm.LinearSVC()
        clf.fit(twoclass_train_data, twoclass_train_label)
        #print clf
        accuracy = clf.score(twoclass_test_data, twoclass_test_label)
        print accuracy

        accuracy_list.append((partition, accuracy))

    #pprint.pprint(sorted(accuracy_list, key = lambda x: x[1]))
    sorted_accuracy_list = sorted(accuracy_list, key = lambda x: x[1])
    pprint.pprint(sorted_accuracy_list)
    print ''
    print 'Highest Accuracy was achieved with the following partition', sorted_accuracy_list[-1]
