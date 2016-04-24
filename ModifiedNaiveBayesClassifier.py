#!/usr/bin/python

from __future__ import print_function
from collections import defaultdict

class NaiveBayesClassifier:

    def __init__(self, trainingFeatures):
        # trainingFeatures is a list of training samples. Each sample is a tuple with a list of
        # features that the sample exhibits and the class it belongs to
        self.trainModel(trainingFeatures)

    def trainModel(self, trainingFeatures):
        # counts of each possible class
        classCounts = defaultdict(int)
        # counts of features given class
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

        
