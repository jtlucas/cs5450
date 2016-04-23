#!/usr/bin/python

from __future__ import print_function
import cPickle as pickle
import os
import random
import nltk.classify.util

from NaiveBayesClassifier import NaiveBayesClassifier

def reviewFeatureExtractor(reviewWords):
    # review is list of words, return dictionary of features
    # can do any filtering/transformation of features here (e.g. removing articles, prepositions etc)
    reviewWordSet = set(reviewWords)
    features = {}
    for word in reviewWordSet:
        features[word] = "true"
    return features

# //////////////////////////////////////////////////
# MAIN SCRIPT
# //////////////////////////////////////////////////

if __name__ == "__main__":
    # load training reviews from pickled file and randomize the list
    print ("Loading data..")
    reviews = pickle.load(open("./data/train_nofulltext.p", "rb"))
    random.shuffle(reviews)

    # extract features for each review and store in list of tuples pertaining to each review
    # this is the training data to be passed to the classifier
    print ("Extracting features..")
    featureSet = [(reviewFeatureExtractor(words), sentiment) for (id, rating, sentiment, words) in reviews]

    # create training and cross-validation feature sets
    trainCutoff = len(featureSet) * 3/4
    trainSet = featureSet[:trainCutoff]
    cvSet = featureSet[trainCutoff:]

    # train Naive Bayes classifier and display output
    print ("Training model..")
    classifier = NaiveBayesClassifier.train(trainSet)
    print ("Training accuracy: ", nltk.classify.util.accuracy(classifier, trainSet))
    print ("Cross-validation accuracy: ", nltk.classify.util.accuracy(classifier, cvSet))
    classifier.show_most_informative_features()

    # save model to reuse for testing
    print ("Saving model to classifier.p")
    pickle.dump(classifier, open("./classifier.p", "wb"))
