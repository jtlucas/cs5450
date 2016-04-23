#!/usr/bin/python

from __future__ import print_function
import cPickle as pickle
import os
import random
import nltk.classify.util

import NaiveBayesClassifier

def reviewFeatureExtractor(reviewWords):
    # review is list of words, return dictionary of features
    # can do any filtering of features here (e.g. removing articles, prepositions etc)
    reviewWordSet = set(reviewWords)
    features = {}
    for word in reviewWordSet:
        features[word] = "true"
    return features

# //////////////////////////////////////////////////
# MAIN SCRIPT
# //////////////////////////////////////////////////

# load training reviews from pickled file and randomize the list
reviews = pickle.load(open("./data/train.p", "rb"))
random.shuffle(reviews)

# extract features for each review and store in list of tuples pertaining to each review
# this is the training data to be passed to the classifier
featureSet = [(reviewFeatureExtractor(words), sentiment) for (id, rating, sentiment, text, words) in reviews]

# create training and cross-validation feature sets
trainCutoff = len(featureSet) * 3/4
trainSet = featureSet[:trainCutoff]
cvSet = featureSet[trainCutoff:]

# train Naive Bayes classifier and display output
classifier = NaiveBayesClassifier.train(trainSet)
print ("training accuracy: ", nltk.classify.util.accuracy(classifier, trainSet))
print ("cross-validation accuracy: ", nltk.classify.util.accuracy(classifier, cvSet))
classifier.show_most_informative_features()
