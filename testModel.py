#!/usr/bin/python

# Test the model on the test data and create the output submission file

from __future__ import print_function
import cPickle as pickle
import os
import random
import nltk.classify.util
from nltk.tokenize import word_tokenize

from NaiveBayesClassifier import NaiveBayesClassifier
from trainModel import reviewFeatureExtractor

# Load model from classifier.p
print ("Loading model..")
classifier = pickle.load(open("./classifier.p", "rb"))

# Initialize test results file
testResults = open("results.txt", "w+")
testResults.write("id, labels\n")

# Parse test reviews, classify, and write to output file
print ("Classifying test data..")
for root, dirs, files in os.walk("./data/test"):
    for name in files:
        if name.endswith(".txt"):
            fullPath = os.path.join(root, name)
            file = open(fullPath)
            text = file.read()
            file.close()
            dot = name.index(".")
            id = int(name[:dot])
            text = text.decode('unicode_escape').encode('ascii','ignore')
            features = reviewFeatureExtractor(word_tokenize(text))
            result = classifier.classify(features)
            line = ""
            if (result == "pos"):
                line = "%d, %d\n" % (id, 1)
            elif (result == "neg"):
                line = "%d, %d\n" % (id, 0)
            testResults.write(line)

testResults.close()
