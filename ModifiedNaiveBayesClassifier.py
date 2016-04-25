#!/usr/bin/python

from __future__ import print_function
from collections import defaultdict

class NaiveBayesClassifier:

    def __init__(self, trainingFeatures):
        # trainingFeatures is a list of training samples. Each sample is a tuple with a list of
        # features that the sample exhibits and the class it belongs to
        self.trainModel(trainingFeatures)

    # train the classifier by recording counts of features and classes in the provided training set
    def trainModel(self, trainingFeatures):
        # counts of each possible class
        classCounts = defaultdict(int)
        # count of samples that exhibit features given class
        featureClassCounts = defaultdict(int)
        # set of all features
        allFeatures = set()

        for featureList, className in trainingFeatures:
            # increment count of class for this sample
            classCounts[className] += 1
            for feature in featureList:
                # add feature to set of all features
                allFeatures.add(feature)
                # increment count of this feature given the class
                featureClassCounts[className, feature] += 1

        # calculate probability distributions from counts to use for classification
        self.numSamples = self.__totalCount(classCounts)
        self.allFeatures = allFeatures
        self.classPriorDist = {nm: float(cnt)/self.numSamples for nm, cnt in classCounts.items()}
        self.featureClassDist = defaultdict(float)
        for (className, feature), cnt in featureClassCounts.items():
            self.featureClassDist[className, feature] = float(featureClassCounts[className, feature]) / classCounts[className]

    # classify a new feature set based on the trained model
    def classify(self, featureSet):
        # remember to check if feature is in allFeatures set, and ignore if it isn't

        # find class prior from counts
        classPrior = {}

    def __totalCount(counterDict):
        total = 0
        for name, count in counterDict.items():
            total += count
        return total
