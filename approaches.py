import sklearn
import sklearn.svm as svm

from multiprocessing import Pool, cpu_count
import itertools
import time

import classes
import helper


def bruteforce_SVM(train, test, class_labels, linear_kernel=True):
    # Generates all possible 2-set eqipartitions from the class labels
    # Computes the SVM, linear or otherwise, for each of these partitions
    # returns the results

    partition_list = helper.class_partitions(class_labels)
    results = []

    for count, partition in enumerate(partition_list):
        partition = tuple(partition)
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
        result = classes.Result(accuracy=clf.score(twoclass_test_data, twoclass_test_label), partition=partition)

        results.append(result)

    return results

def _train_and_test(arg_tuple):
    initial_two, remaining_labels, mapping, test, train, class_labels = arg_tuple
    clf = svm.LinearSVC()

    # Train classifier based on initial two classes.
    # extract data points from the train object, generate labels on the fly
    clf.fit(train[initial_two[0]] + train[initial_two[1]],
            [0]*len(train[initial_two[0]]) + [1]*len(train[initial_two[1]]))

    for label in remaining_labels:
        score = clf.score(train[label], [0]*len(train[label]))
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
        overlap[key] = 0.5 - abs(clf.score(test[key], [reverse_mapping[key]]*len(test[key])) - 0.5)

    result = classes.Result(accuracy=clf.score(test_samples, test_labels),
                             partition=tuple(mapping.itervalues()),
                             overlap=overlap,
                             classifier=clf)
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
    
    print len(inputs)
    print 'CPU Count: {}'.format(cpu_count())

    pool = Pool(cpu_count())
    results = pool.map_async(_train_and_test, inputs)
    while not results.ready():
        print('Num left: {}'.format(results._number_left))
        time.sleep(2)
    results = results.get()
    pool.close()
    pool.join()

    return results
