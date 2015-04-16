""" This module contains helper functions for use by other modules """

import itertools
import pprint
import json

import config

def create_set(label_list):
    """ Simply create and return a dictionary with the labels as keys """
    # This is a horrible misnomer
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
    # Don't really need to return this since it's modified in-place
    return dataset

def class_partitions(class_labels):

    # The frozenset method is a little hacky for duplicate elimination, optimize later
    splits = set()
    count = 0

    for split in itertools.combinations(class_labels, len(class_labels)/2):
        count += 1
        splits.add(frozenset((frozenset(split),frozenset(class_labels - set(split)))))

    return splits

def get_optimal_classifier(approach, *args, **kwargs):
    """
    Given an approach to find a classifier, runs all the data through it
    and obtains the best-performing classifier
    """

    results = approach(*args, **kwargs)
    #print 'PRINTING ALL RESULTS'
    #pprint.pprint([x.value for x in results])
    #pprint.pprint(results)
    #print 'DONE.'
    optimal_result = results[0]
    for result in results:
        if result.value > optimal_result.value:
            optimal_result = result

    return optimal_result

def get_statistics(results):
    # Takes a list of 'Metrics' as input
    # Outputs:
    #   Best value
    #   Maximum accuracy
    #   Average Value
    #   Average Accuracy

    max_val = results[0]
    max_acc = results[0]
    sum_val = results[0].value
    sum_acc = results[0].accuracy

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

def log_format(labels, lpart=None, rpart=None, overlap=None, accuracy=None, classifier=None):
    """ Returns a formatted log string for use with the progress tracking file/log """

    data = {}
    data['labels'] = list(sorted(labels))
    if lpart:
        data['lpart'] = list(sorted(lpart))
        data['lpure'] = list(sorted(lpart - overlap))
    if rpart:
        data['rpart'] = list(sorted(rpart))
        data['rpure'] = list(sorted(rpart - overlap))
    if overlap:
        data['overlap'] = list(sorted(overlap))
    if accuracy is not None:
        data['accuracy'] = '{:.4f}'.format(accuracy)
    if classifier:
        data['classifier'] = list(sorted(classifier))

    with open(config.FILENAME, 'a') as f:
        f.write(json.dumps(data)+'\n')
    return json.dumps(data)

def exportTreeToJSON(tree):
    def _exportTreeToJSON(tree):
        offset_table = {}
        def removeKey(root):
            root['classes'] = list(root['classes'])
            h = hash(frozenset(root['classes']))
            if not h in offset_table:
                offset_table[h] = list(root['classifier'])
            root['hash'] = h
            del root['classifier']

            if root.has_key('negative'):
                removeKey(root['negative'])
            if root.has_key('positive'):
                removeKey(root['positive'])

        removeKey(tree)
        return {'table': offset_table, 'tree': tree}

    return json.dumps(_exportTreeToJSON(tree))
    #return _exportTreeToJSON(tree)
