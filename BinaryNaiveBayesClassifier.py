#!/usr/bin/python

from __future__ import print_function
from collections import defaultdict
import math

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
        numSamples = self.__totalCount(classCounts)
        self.allFeatures = allFeatures
        self.classPriorDist = {nm: (cnt + 0.5)/(numSamples + 1) for nm, cnt in classCounts.items()}
        self.featureClassDist = {}
        for className in classCounts.keys():
            for feature in allFeatures:
                self.featureClassDist[className, feature] = (featureClassCounts[className, feature] + 0.5) / (classCounts[className] + 1)

    # classify a new feature set based on the trained model
    def classify(self, featureSet):
        # remove features in featureSet that we haven't seen
        temp = []
        for feature in featureSet:
            if feature in self.allFeatures:
                temp.append(feature)
        featureSet = temp

        # get log prob of class prior for initial class probabilities
        prob = {className: math.log(prob, 2) for className, prob in self.classPriorDist.items()}

        # add feature class conditional probability information
        for className in prob.keys():
            for feature in featureSet:
                condProb = self.featureClassDist[className, feature]
                prob[className] += math.log(condProb, 2)

        # get class with maximum probability
        maxClass = prob.keys()[0]
        maxProb = prob[maxClass]
        for className in prob.keys():
            if (prob[className] > maxProb):
                maxProb = prob[className]
                maxClass = className

        return maxClass

    # def showMostInformativeFeatures(self, n=10):

    def __totalCount(self, counterDict):
        total = 0
        for name, count in counterDict.items():
            total += count
        return total
