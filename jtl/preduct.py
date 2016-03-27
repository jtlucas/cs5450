# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:09:03 2016

@author: jtlucas
"""
from __future__ import print_function
import os
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
import nltk.classify.util

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
    
reviews = parseReviews("../data/train")
pos = []
neg = []
for x in reviews:
    parseme = x[3].decode('unicode_escape').encode('ascii','ignore')
    sentoke = sent_tokenize(parseme)
    words = {}
    for sent in sentoke:
        wordtoke = word_tokenize(sent)
        for word in wordtoke:
            words[word] = 'true'
        
    if x[2] == 0: # Neg
        neg.append((words, "neg"))
    else:
        pos.append((words, "pos"))
        
negcutoff = len(neg)*3/4
poscutoff = len(pos)*3/4

trainfeats = neg[:negcutoff] + pos[:poscutoff]
testfeats = neg[negcutoff:] + pos[poscutoff:]
print ('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))
 
c1 = NaiveBayesClassifier.train(trainfeats)
print ('accuracy:', nltk.classify.util.accuracy(c1, testfeats))
c1.show_most_informative_features()

print(len(reviews))

