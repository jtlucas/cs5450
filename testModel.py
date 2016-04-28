#!/usr/bin/python

# Test the model on the test data and create the output submission file

from __future__ import print_function
import cPickle as pickle
from BinaryNaiveBayesClassifier import NaiveBayesClassifier
from parseAndStoreReviews import parseTestReviews
from trainModel import getBestWords, extractFeaturesFromSet, reviewFeatureExtractor

print ("Training classifier..")
trainReviews = pickle.load(open("./data/train_nofulltext.p", "rb"))
bestwords = getBestWords(trainReviews)
trainFeatureSet = extractFeaturesFromSet(trainReviews, bestwords)
classifier = NaiveBayesClassifier(trainFeatureSet)

print ("Parsing test data..")
testReviews = parseTestReviews("./data/test")

# Initialize test results file
testResults = open("results.csv", "w+")
testResults.write("id,labels\n")

print ("Classifying test data..")
i = 0
for id, review in testReviews:
    features = reviewFeatureExtractor(review, bestwords)
    result = classifier.classify(features)
    line = ""
    if (result == "pos"):
        line = "%d,%d\n" % (id, 1)
    elif (result == "neg"):
        line = "%d,%d\n" % (id, 0)
    testResults.write(line)

    if(i%20==0):
        print (".", end="")
    if(i%1000==0):
        print (str(i))
    i = i + 1;

testResults.close()
