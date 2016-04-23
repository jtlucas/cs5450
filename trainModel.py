#!/usr/bin/python

from __future__ import print_function
import cPickle as pickle
import os
import string
import random
import collections
import nltk.classify.util
import nltk

from NaiveBayesClassifier import NaiveBayesClassifier

def reviewFeatureExtractor(reviewWords, bestwords=None):
    # review is list of words, return dictionary of features
    # can do any filtering/transformation of features here (e.g. removing articles, prepositions etc)
    reviewWordSet = set(reviewWords)
    features = {}

    # Remove words that have numbers
    #nodWordSet = set()
    #for word in reviewWordSet:
    #    skip = False
    #    for ch in word:
    #        if ch in string.digits:
    #            skip = True
    #            break
    #    if not skip:
    #        nodWordSet.add(word.lower())

    # Utilize NLTK to determine parts of speech
    #posWordSet = set()
    #posWordSet = nltk.pos_tag(nodWordSet)
    #for word in posWordSet:
    #    if word[1].__contains__('RB'): #Adverbs
    #        features[word[0]] = True
    #    elif word[1].__contains__('JJ'): #Adjectives
    #        features[word[0]] = True

    for word in reviewWordSet:
        if(bestwords != None):
            if(word.lower() in bestwords):
                features[word] = True
        else:
            features[word] = True

    bigram_finder = nltk.BigramCollocationFinder.from_words(reviewWords)
    bigrams = bigram_finder.nbest(nltk.BigramAssocMeasures.chi_sq, 200)
    b = dict([(bigram, True) for bigram in bigrams])
    b.update(features)

    #trigram_finder = nltk.TrigramCollocationFinder.from_words(reviewWords)
    #trigrams = trigram_finder.nbest(nltk.TrigramAssocMeasures.chi_sq, int(len(reviewWords)/10))
    #t = dict([(trigram, True) for trigram in trigrams])
    #t.update(b)

    return b

# //////////////////////////////////////////////////
# MAIN SCRIPT
# //////////////////////////////////////////////////

if __name__ == "__main__":
    # load training reviews from pickled file and randomize the list
    print ("Loading data..")
    reviews = pickle.load(open("./data/train_nofulltext.p", "rb"))
    random.shuffle(reviews)

    # create training and cross-validation feature sets
    trainCutoff = len(reviews) * 3/4
    trainSet = reviews[:trainCutoff]
    cvSet = reviews[trainCutoff:]

    # extract features for each review and store in list of tuples pertaining to each review
    # this is the training data to be passed to the classifier
    print ("Extracting features..")
    word_freq = nltk.probability.FreqDist()
    label_freq = nltk.probability.ConditionalFreqDist()

    print ("Getting word frequency..")
    i = 0
    for review in trainSet:
        if(review[2] == 'pos'):
            for word in review[3]:
                word_freq.update(nltk.probability.FreqDist(word.lower()))
                label_freq['pos'].update(nltk.probability.FreqDist(word.lower()))
        elif(review[2] == 'neg'):
            for word in review[3]:
                word_freq.update(nltk.probability.FreqDist(word.lower()))
                label_freq['neg'].update(nltk.probability.FreqDist(word.lower()))

        if(i%5==0):
            print (".", end="")
        if(i%1000==0):
            print (str(i))
        i = i + 1;

    print(str(i) + " Finished")
    pos_words = label_freq['pos'].N()
    neg_words = label_freq['neg'].N()
    total_words = pos_words + neg_words
    word_scores = {}

    print("Calculating word scores..")
    for word, freq in word_freq.iteritems():
        pos_score = nltk.BigramAssocMeasures.chi_sq(label_freq['pos'][word],
                                              (freq, pos_words), total_words)
        neg_score = nltk.BigramAssocMeasures.chi_sq(label_freq['neg'][word],
                                              (freq, neg_words), total_words)
        word_scores[word] = pos_score + neg_score

    print("Sorting Word scores..")
    best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]
    print("Getting Best words..")
    bestwords = set([w for w, s in best])

    i = 0;
    trainFeatureSet = []
    for (id, rating, sentiment, words) in trainSet:
        trainFeatureSet.append((reviewFeatureExtractor(words,bestwords), sentiment))

        if(i%5==0):
            print (".", end="")
        if(i%1000==0):
            print (str(i))
        i = i + 1;

    cvFeatureSet = []
    for (id, rating, sentiment, words) in cvSet:
        cvFeatureSet.append((reviewFeatureExtractor(words,bestwords), sentiment))

        if(i%5==0):
            print (".", end="")
        if(i%1000==0):
            print (str(i))
        i = i + 1;

    print(str(i) + " Finished")

    # train Naive Bayes classifier and display output
    print ("Training model..")
    classifier = NaiveBayesClassifier.train(trainFeatureSet)

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    for i, (feats, label) in enumerate(cvFeatureSet):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    print ("Training accuracy: ", nltk.classify.util.accuracy(classifier, trainFeatureSet))
    print ("Cross-validation accuracy: ", nltk.classify.util.accuracy(classifier, cvFeatureSet))
    print ("'pos' Precision: ", nltk.precision(refsets['pos'], testsets['pos']))
    print ("'pos' Recall: ", nltk.recall(refsets['pos'], testsets['pos']))
    print ("'neg' Precision: ", nltk.precision(refsets['neg'], testsets['neg']))
    print ("'neg' Recall: ", nltk.recall(refsets['neg'], testsets['neg']))

    classifier.show_most_informative_features()

    # save model to reuse for testing
    # print ("Saving model to classifier.p")
    # pickle.dump(classifier, open("./classifier.p", "wb"))
