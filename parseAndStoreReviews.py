#!/usr/bin/python

# This file is responsible for parsing movie reviews for training data and storing them in a
# pickled file train.p. Read from the file by importing cPickle as below and running:
# reviews = pickle.load(open("./data/train.p", "rb"))

from __future__ import print_function
import os
import cPickle as pickle
from nltk.tokenize import word_tokenize

# create list of all reviews. List of tuples - each tuple contains full text, id, rating, sentiment (0 = negative, 1 = positive)
def parseReviews(directory, includeFullText):
    reviewList = []
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
                sentiment = "pos"
                if rating <= 4:
                    sentiment = "neg"
                file.close()
                text = text.decode('unicode_escape').encode('ascii','ignore')
                if (includeFullText):
                    reviewList.append((id, rating, sentiment, text, word_tokenize(text)))
                else:
                    reviewList.append((id, rating, sentiment, word_tokenize(text)))
    return reviewList

reviews = parseReviews("./data/train", False)
pickle.dump(reviews, open("./data/train_nofulltext.p", "wb"))
