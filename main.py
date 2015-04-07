import sklearn
import sklearn.svm as svm
#from sklearn.qda import QDA
#from sklearn.lda import LDA
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.multiclass import OneVsRestClassifier

#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import numpy as np

import itertools
import collections
import pprint
from multiprocessing import Pool, cpu_count
import sys
import time

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

class Result(object):

    def __init__(self, value=None, accuracy=None, balance=None, overlap=None, margin=None, partition=None, classifier=None):
        self.accuracy = accuracy
        self.overlap = overlap
        self.margin = margin
        self.partition = partition
        self.classifier = classifier

        self.balance = 2.0*min(len(partition[0]), len(partition[1]))/(len(partition[0]) + len(partition[1]))
        self.value = objective_function(accuracy=self.accuracy, balance=self.balance, overlap=self.overlap, margin=self.margin)

    def __repr__(self):
        return ('(Value: %s' ' Accuracy: %s, Balance: %s, Overlap: %s, Margin: %s, Partition: %s, Classifier: %s)'
                % (repr(self.value),  repr(self.accuracy),  repr(self.balance),  repr(self.overlap),  repr(self.margin),  repr(self.partition), repr(self.classifier)))


class TreeNode(object):
    def __init__(self, train, class_labels):
        # Assumes train is well shuffled/random/not-ordered
        #print 'Working with the partition: %s' % class_labels
        self.class_labels = class_labels
        self.subtrain = {}
        self.subtest = {}
        for key in train.keys():
            l = len(train[key])
            if l < 8:
                raise Exception
                    # too few samples
            else:
                self.subtrain[key] = train[key][:l*7/8]
                self.subtest[key] = train[key][l*7/8:]

        optimal = get_optimal_classifier(pairwise_SVM_A1, self.subtest, self.subtrain, class_labels)

        self.classifier = optimal.classifier
        self.overlap = optimal.overlap
        self.overlapping_classes = set(map(lambda y: y[0], filter(lambda x: x[1]>=0.0001, optimal.overlap.iteritems())))
        self.accuracy = optimal.accuracy

        self.lkeys = optimal.partition[0]
        self.rkeys = optimal.partition[1]

        #print "Overlapping Classes %s" % self.overlapping_classes

        if (self.lkeys | self.overlapping_classes) != class_labels:
            self.lkeys = self.lkeys | self.overlapping_classes
        if (self.rkeys | self.overlapping_classes) != class_labels:
            self.rkeys = self.rkeys | self.overlapping_classes


        self.lchild = None
        self.rchild = None
        if len(self.lkeys) > 1:
           self.lchild = TreeNode(train, self.lkeys)
        if len(self.rkeys) > 1:
           self.rchild = TreeNode(train, self.rkeys)


    def __repr__(self):
        s = ('%s -> (%s, %s)\n%s\n%s' % (self.class_labels, self.accuracy, repr(self.overlap),  repr(self.lchild), repr(self.rchild))).split('\n')
        return s[0] + '\n' + '\n'.join('\t' + string for string in s[1:])

    def predict(self, feature_vector):
        if self.classifier.predict(feature_vector) == 0:
            if len(self.lkeys) > 1:
                return self.lchild.predict(feature_vector)
            else:
                val, = self.lkeys
                return val
        else:
            if len(self.rkeys) > 1:
                return self.rchild.predict(feature_vector)
            else:
                val, = self.rkeys
                return val


    def _score(self, test_vectors, test_labels):
        predictions = map(self.predict, test_vectors)
        correct = filter(lambda x: x[0]==x[1], zip(test_labels, predictions))
        print 'Correctly labeled classes %s/%s' % (len(correct), len(predictions))
        print 'Accuracy: %s' % (1.0*len(correct)/len(predictions))
        

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
        value *= (1 - (sum((1 for x in filter(lambda x: x!=1.0, (overlap.values()))))/len(overlap)))
        #value *= 1.0
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

    max_val=results[0]
    max_acc=results[0]
    sum_val=results[0].value
    sum_acc=results[0].accuracy

    size = len(results)

    for metric in results[1:]:
        if metric.accuracy > max_acc.accuracy:
            max_acc = metric
        if metric.value > max_val.value:
            max_val = metric

        sum_val += metric.value
        sum_acc += metric.accuracy

    print 'Best Accuracy: %s\n%s' % (max_acc.accuracy, max_acc)
    print 'Highest Value: %s\n%s' % (max_val.value, max_val)
    print 
    print 'Average Accuracy: %s' % (str(sum_acc/size),)
    print 'Average Value: %s' % (str(sum_val/size),)

    pprint.pprint(max_val.classifier.__dict__)


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
            feature_vector, category = tuple([int(x) for x in datum[:-1]]), datum[-1]
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

def get_optimal_classifier(approach, *args, **kwargs):
    results = approach(*args, **kwargs)
    optimal_result = results[0]
    for result in results:
        #if result.accuracy > optimal_result.accuracy:
        if result.value > optimal_result.value:
            optimal_result = result

    return optimal_result

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
        resutl = Result(accuracy=clf.score(twoclass_test_data, twoclass_test_label), partition=partition)

        results.append(result)

    return results

def _train_and_test(arg_tuple):
    initial_two, remaining_labels, mapping, test, train, class_labels = arg_tuple
    clf = svm.LinearSVC()

    #pprint.pprint('This is the test set being passed:')
    #pprint.pprint(map(lambda x: (x[0], len(x[1])), test.iteritems()))
    # Train classifier based on initial two classes.
    # extract data points from the train object, generate labels on the fly
    clf.fit(train[initial_two[0]] + train[initial_two[1]],
            [0]*len(train[initial_two[0]]) + [1]*len(train[initial_two[1]]))

    for label in remaining_labels:
        score = clf.score(train[label], [0]*len(train[label]))
        # This is interesting to print, and observe actual values
        if score > 0.5:
            mapping[0].add(label)
        else:
            mapping[1].add(label)

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


    # Get overlap
    # Greater the value of each overlap, the more it is.
    overlap = {}
    for key in class_labels:
        overlap[key] = 0.5 - abs(clf.score(test[key], [key]*len(test[key])) - 0.5)
        #print 'Key: %s: Score: %s' % (key, clf.score(test[key], [key]*len(test[key])))

    #print 'Testing the re-trained optimal partition, score:',
    result = Result(accuracy=clf.score(test_samples, test_labels),
                             partition=tuple(mapping.itervalues()),
                             overlap=overlap,
                             classifier=clf)

    #results.append(result)
    return result

def pairwise_SVM_A1(test, train, class_labels):
    # Take pairwise classes
    # Compute SVM for that class
    # Then put each class in one of the two major classes
    # Then recompute accuracy on training set.
    # do so until all classes are done.
    # Compute the accuracy on the training set, keep track of results, find best result

    results = []


    inputs = []
    # For every possible pair of classes
    for initial_two in itertools.combinations(class_labels, 2):

        remaining_labels = set(class_labels - set(initial_two))
        mapping = {0: {initial_two[0]}, 1: {initial_two[1]}}
        inputs.append((initial_two, remaining_labels, mapping, test, train, class_labels))
    
    #print inputs[0]
    print len(inputs)
    print 'CPU Count: {}'.format(cpu_count())
    #sys.exit(0)

    pool = Pool(cpu_count())
    results = pool.map_async(_train_and_test, inputs)
    while not results.ready():
        print('Num left: {}'.format(results._number_left))
        time.sleep(2)
    results = results.get()
    pool.close()
    pool.join()

    #print results
    #sys.exit(1)

    return results


if __name__ == '__main__':

    # Do dataset specific things here in main

    # Digit dataset
    TRAINING_DATA_PATH='optdigits/optdigits.tra'
    TESTING_DATA_PATH='optdigits/optdigits.tes'
    #CLASS_LABELS = set(range(0,10)) # Class labels are b/w 0..9 (inclusive)
    CLASS_LABELS = set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    # Latin Letter Dataset
    #TRAINING_DATA_PATH='../datasets/letter-recognition/letter-recognition.tra'
    #TESTING_DATA_PATH='../datasets/letter-recognition/letter-recognition.tes'
    #CLASS_LABELS = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

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
    #results = pairwise_SVM_A1(train, test, CLASS_LABELS)
    #results = pairwise_SVM_recursive(train, test, CLASS_LABELS)
    #print get_optimal_classifier(pairwise_SVM_A1, test, train, CLASS_LABELS)
    #sys.exit(0)

    x = TreeNode(train, CLASS_LABELS)
    print x
    test_vectors = []
    test_labels = []
    for key in test.keys():
        test_vectors += test[key]
        test_labels += [key]*len(test[key])
    x._score(test_vectors, test_labels)
    print x


    # Baseline testing using different classifiers
    ##train_data = []
    ##train_labels = []
    ##test_data = []
    ##test_labels = []
    ##for key in train.keys():
    ##    train_data += train[key]
    ##    train_labels += [key]*len(train[key])
    ##for key in test.keys():
    ##    test_data += test[key]
    ##    test_labels += [key]*len(test[key])

    ##clf_qda = QDA()
    ##clf_lda = LDA()
    ##clf_svc = svm.SVC()
    ##clf_dtree = DecisionTreeClassifier()
    ##clf_bayes = GaussianNB()
    ##clf_rand_forest = RandomForestClassifier()
    ##clf_ada_boost = AdaBoostClassifier()
    ##clf_1vr = OneVsRestClassifier(svm.SVC())

    ##clf_qda.fit(train_data, train_labels)
    ##clf_lda.fit(train_data, train_labels)
    ##clf_svc.fit(train_data, train_labels)
    ##clf_dtree.fit(train_data, train_labels)
    ##clf_bayes.fit(train_data, train_labels)
    ##clf_rand_forest.fit(train_data, train_labels)
    ##clf_ada_boost.fit(train_data, train_labels)
    ##clf_1vr.fit(train_data, train_labels)


    ##print ''
    ##print 'Base Line accuraccies:'
    ##print 'QDA: %s' % clf_qda.score(test_data, test_labels)
    ##print clf_svc.__dict__
    ##print 'LDA: %s' % clf_lda.score(test_data, test_labels)
    ##print clf_svc.__dict__
    ##print 'SVC: %s' % clf_svc.score(test_data, test_labels)
    ##print clf_svc.__dict__
    ###print 'DecisionTreeClassifier: %s' % clf_dtree.score(test_data, test_labels)
    ##print 'RandomForestClassifier: %s' % clf_rand_forest.score(test_data, test_labels)
    ##print 'GaussianNB: %s' % clf_bayes.score(test_data, test_labels)
    ##print 'AdaBoostClassifier: %s' % clf_ada_boost.score(test_data, test_labels)
    ##print 'One Vs. Rest: %s' % clf_1vr.score(test_data, test_labels)
    #pprint.pprint(results)
    #get_statistics(results)

