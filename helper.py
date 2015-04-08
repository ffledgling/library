import itertools
import pprint

def create_set(label_list):
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

    for split in itertools.combinations(CLASS_LABELS, len(CLASS_LABELS)/2):
        count += 1
        splits.add(frozenset((frozenset(split),frozenset(CLASS_LABELS - set(split)))))

    return splits

def get_optimal_classifier(approach, *args, **kwargs):
    results = approach(*args, **kwargs)
    print 'PRINTING ALL RESULTS'
    pprint.pprint([x.value for x in results])
    pprint.pprint(results)
    print 'DONE.'
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
