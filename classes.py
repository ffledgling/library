import approaches
import helper
import objective

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

class Result(object):

    def __init__(self, value=None, accuracy=None, balance=None, overlap=None, margin=None, partition=None, classifier=None):
        self.accuracy = accuracy
        self.overlap = overlap
        self.margin = margin
        self.partition = partition
        self.classifier = classifier

        self.balance = 2.0*min(len(partition[0]), len(partition[1]))/(len(partition[0]) + len(partition[1]))

        self.overlapping_classes = set(map(lambda y: y[0], filter(lambda x: x[1]!=0.0, overlap.iteritems())))

        self.overlap_value = 1 - (sum((1 for x in filter(lambda x: x!=0.0, (overlap.values()))))*1.0/len(overlap))
        self.purity = 4.0*len(partition[0]-set(self.overlapping_classes))*len(partition[1]-set(self.overlapping_classes))/((len(partition[0] | partition[1]))**2)
        self.value = objective.objective_function(accuracy=self.accuracy, balance=self.balance, overlap=self.overlap, margin=self.margin, purity=self.purity)

    def __repr__(self):
        return ('(Value: %s' ', Accuracy: %s, Balance: %s, Overlap: %s, Margin: %s, Purity: %s, Overlapping-Classes: %s, Overlap-value: %s, Partition: %s, Classifier: %s)'
                % (repr(self.value),  repr(self.accuracy),  repr(self.balance),  repr(self.overlap),  repr(self.margin),  repr(self.purity), repr(self.overlapping_classes), repr(self.overlap_value), repr(self.partition), repr(self.classifier)))


class TreeNode(object):
    def __init__(self, train, class_labels):
        # Assumes train is well shuffled/random/not-ordered
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

        optimal = helper.get_optimal_classifier(approaches.pairwise_SVM_A1, self.subtest, self.subtrain, class_labels)
        print 'Optimal:'
        pprint.pprint(optimal)

        self.classifier = optimal.classifier
        self.overlap = optimal.overlap
        self.overlapping_classes = set(map(lambda y: y[0], filter(lambda x: x[1]!=0.0, optimal.overlap.iteritems())))
        self.accuracy = optimal.accuracy

        self.lkeys = optimal.partition[0]
        self.rkeys = optimal.partition[1]

        if (self.lkeys | self.overlapping_classes) != class_labels:
            self.lkeys = self.lkeys | self.overlapping_classes
        if (self.rkeys | self.overlapping_classes) != class_labels:
            self.rkeys = self.rkeys | self.overlapping_classes


        self.lchild = None
        self.rchild = None

        # Logging for progress
        print 'PROG: %s' % class_labels #PROG

        # left child
        if len(self.lkeys) > 1:
           self.lchild = TreeNode(train, self.lkeys)
        else:
            print 'PROG: %s' % self.lkeys #PROG

        # right child
        if len(self.rkeys) > 1:
           self.rchild = TreeNode(train, self.rkeys)
        else:
            print 'PROG: %s' % self.rkeys #PROG


    def __repr__(self):
        s = ('%s -> (%s, %s)\n%s\n%s' % (self.class_labels, self.accuracy, repr(self.overlapping_classes),  repr(self.lchild), repr(self.rchild))).split('\n')
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
