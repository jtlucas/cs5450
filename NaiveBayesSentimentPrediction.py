"""
@author: jtlucas
@author: nconlon
@author: pghantasala
"""
from __future__ import print_function, unicode_literals

import os
import sys
import nltk.classify.util

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.probability import FreqDist, DictionaryProbDist, ELEProbDist, sum_logs
from nltk.classify.api import ClassifierI

##//////////////////////////////////////////////////////
##  Naive Bayes Classifier runner
##  inut: path - the path to training data
##//////////////////////////////////////////////////////

def runNaiveBayesClassifier(trainPath, testPath):
    [neg,pos] = parseTrainReviews(trainPath)

    if testPath != 0:
        testReviews = parseTestReviews(testPath) # We are using actual test data
    else
        testReviews = parseTestReviews(trainPath)

    negcutoff = len(neg)*3/4
    poscutoff = len(pos)*3/4

    trainfeats = neg[:negcutoff] + pos[:poscutoff]
    testfeats = neg[negcutoff:] + pos[poscutoff:]
    print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))
     
    niaveBayesClassifier = NaiveBayesClassifier.train(trainfeats)
    print ('accuracy:', nltk.classify.util.accuracy(niaveBayesClassifier, testfeats))
    niaveBayesClassifier.show_most_informative_features()

    fout = open('output.csv','w+')
    fout.write('id,labels,,\r\n')
    s = '';
    for words in testReviews:
        output = niaveBayesClassifier.classify(words[1])
        
        if output == 'pos':
            s = "%s%d,1,,\r\n" %(s,words[0])
        elif output == 'neg':
            s = "%s%d,0,,\r\n" %(s,words[0])

    fout.write(s)          
    fout.close()

    return 0

##//////////////////////////////////////////////////////
##  Parser for the training data
##//////////////////////////////////////////////////////

def parseTrainReviews(directory):
    print("Parsing training reviews\n")

    pos = [];
    neg = [];

    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(".txt"):
                fullPath = os.path.join(root, name)
                file = open(fullPath)
                text = file.read()
                splitter = name.index("_")
                dot = name.index(".")
                id = int(name[:splitter])
                rating = int(name[splitter+1:dot])
                sentiment = 1
                if rating <= 4:
                    sentiment = 0
                file.close()

                parseme = text.decode('unicode_escape').encode('ascii','ignore')

                sentoke = sent_tokenize(parseme)
                words = {}
                for sent in sentoke:
                    wordtoke = word_tokenize(sent)
                    for word in wordtoke:
                        if word.__contains__('/'):
                            #words[word] = 'false'
                            pass
                        else:
                            words[word] = 'true'
                if sentiment == 0: # Neg
                    neg.append((words, "neg"))
                else:
                    pos.append((words, "pos"))    
    return [neg,pos]

##//////////////////////////////////////////////////////
##  Parser for the test data
##//////////////////////////////////////////////////////

def parseTestReviews(directory):
    print("Parsing test reviews\n")

    output = []

    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(".txt"):
                fullPath = os.path.join(root, name)
                file = open(fullPath)
                text = file.read()
                try:
                    splitter = name.index("_")
                except:
                    splitter = name.index(".")
                id = int(name[:splitter])
                file.close()
                parseme = text.decode('unicode_escape').encode('ascii','ignore')

                sentoke = sent_tokenize(parseme)
                words = {}
                for sent in sentoke:
                    wordtoke = word_tokenize(sent)
                    for word in wordtoke:
                        if word.__contains__('/'):
                            #words[word] = 'false'
                            pass
                        else:
                            words[word] = 'true'
                output.append((id,words))  

    return output    

##//////////////////////////////////////////////////////
##  Naive Bayes Classifier
##//////////////////////////////////////////////////////

class NaiveBayesClassifier(ClassifierI):

    def __init__(self, label_probdist, feature_probdist):
        self._label_probdist = label_probdist
        self._feature_probdist = feature_probdist
        self._labels = list(label_probdist.samples())

    def labels(self):
        return self._labels

    def classify(self, featureset):
        return self.prob_classify(featureset).max()

    def prob_classify(self, featureset):
        # Discard any feature names that we've never seen before.
        # Otherwise, we'll just assign a probability of 0 to
        # everything.
        featureset = featureset.copy()
        for fname in list(featureset.keys()):
            for label in self._labels:
                if (label, fname) in self._feature_probdist:
                    break
            else:
                #print 'Ignoring unseen feature %s' % fname
                del featureset[fname]

        # Find the log probabilty of each label, given the features.
        # Start with the log probability of the label itself.
        logprob = {}
        for label in self._labels:
            logprob[label] = self._label_probdist.logprob(label)

        # Then add in the log probability of features given labels.
        for label in self._labels:
            for (fname, fval) in featureset.items():
                if (label, fname) in self._feature_probdist:
                    feature_probs = self._feature_probdist[label, fname]
                    logprob[label] += feature_probs.logprob(fval)
                else:
                    # nb: This case will never come up if the
                    # classifier was created by
                    # NaiveBayesClassifier.train().
                    logprob[label] += sum_logs([]) # = -INF.

        return DictionaryProbDist(logprob, normalize=True, log=True)

    def show_most_informative_features(self, n=10):
        # Determine the most relevant features, and display them.
        cpdist = self._feature_probdist
        print('Most Informative Features')

        for (fname, fval) in self.most_informative_features(n):
            def labelprob(l):
                return cpdist[l, fname].prob(fval)

            labels = sorted([l for l in self._labels
                             if fval in cpdist[l, fname].samples()],
                            key=labelprob)
            if len(labels) == 1:
                continue
            l0 = labels[0]
            l1 = labels[-1]
            if cpdist[l0, fname].prob(fval) == 0:
                ratio = 'INF'
            else:
                ratio = '%8.1f' % (cpdist[l1, fname].prob(fval) /
                                   cpdist[l0, fname].prob(fval))
            print(('%24s = %-14r %6s : %-6s = %s : 1.0' %
                   (fname, fval, ("%s" % l1)[:6], ("%s" % l0)[:6], ratio)))

    def most_informative_features(self, n=100):
        # The set of (fname, fval) pairs used by this classifier.
        features = set()
        # The max & min probability associated w/ each (fname, fval)
        # pair.  Maps (fname,fval) -> float.
        maxprob = defaultdict(lambda: 0.0)
        minprob = defaultdict(lambda: 1.0)

        for (label, fname), probdist in self._feature_probdist.items():
            for fval in probdist.samples():
                feature = (fname, fval)
                features.add(feature)
                p = probdist.prob(fval)
                maxprob[feature] = max(p, maxprob[feature])
                minprob[feature] = min(p, minprob[feature])
                if minprob[feature] == 0:
                    features.discard(feature)

        # Convert features to a list, & sort it by how informative
        # features are.
        features = sorted(features,
                          key=lambda feature_:
                          minprob[feature_]/maxprob[feature_])
        return features[:n]

    @classmethod
    def train(cls, labeled_featuresets, estimator=ELEProbDist):
        label_freqdist = FreqDist()
        feature_freqdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)
        fnames = set()

        # Count up how many times each feature value occurred, given
        # the label and featurename.
        for featureset, label in labeled_featuresets:
            label_freqdist[label] += 1
            for fname, fval in featureset.items():
                # Increment freq(fval|label, fname)
                feature_freqdist[label, fname][fval] += 1
                # Record that fname can take the value fval.
                feature_values[fname].add(fval)
                # Keep a list of all feature names.
                fnames.add(fname)

        # If a feature didn't have a value given for an instance, then
        # we assume that it gets the implicit value 'None.'  This loop
        # counts up the number of 'missing' feature values for each
        # (label,fname) pair, and increments the count of the fval
        # 'None' by that amount.
        for label in label_freqdist:
            num_samples = label_freqdist[label]
            for fname in fnames:
                count = feature_freqdist[label, fname].N()
                # Only add a None key when necessary, i.e. if there are
                # any samples with feature 'fname' missing.
                if num_samples - count > 0:
                    feature_freqdist[label, fname][None] += num_samples - count
                    feature_values[fname].add(None)

        # Create the P(label) distribution
        label_probdist = estimator(label_freqdist)

        # Create the P(fval|label, fname) distribution
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label, fname] = probdist

        return cls(label_probdist, feature_probdist)

##//////////////////////////////////////////////////////
##  Main
##//////////////////////////////////////////////////////

if __name__ == '__main__':
    if(len(sys.argv) == 1):
        print("Usage: <python> NaiveBayesClassifier <path to training data> <path to test data>\n")
    elif(len(sys.argv) == 2):
        runNaiveBayesClassifier(sys.argv[1],0)#"data/train", 0)
    elif(len(sys.argv) == 3):
        runNaiveBayesClassifier(sys.argv[1], sys.argv[2])

