#!/usr/bin/python

from __future__ import print_function
import os
from nltk import tokenize

# create list of all reviews. List of tuples - each tuple contains full text, id, rating, sentiment (0 = negative, 1 = positive)
def parseReviews(directory):
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
                sentiment = 1
                if rating <= 4:
                    sentiment = 0
                file.close()
                reviewList.append((id, rating, sentiment, text))
    return reviewList

reviews = parseReviews("./data/train")

print(len(reviews))
print(reviews[0])
print(reviews[100])
print(reviews[1000])
print(reviews[22000])
