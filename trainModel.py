#!/usr/bin/python

from __future__ import print_function
import cPickle as pickle
import random
import collections
import nltk.classify.util
import nltk

from BinaryNaiveBayesClassifier import NaiveBayesClassifier

def reviewFeatureExtractor(reviewWords, bestwords=None):
    # review is list of words, return dictionary of features
    # can do any filtering/transformation of features here (e.g. removing articles, prepositions etc)
    stemmer = nltk.PorterStemmer()
    reviewWords = [stemmer.stem(w.lower()) for w in reviewWords]
    reviewWordSet = set(reviewWords)
    features = []

    for word in reviewWordSet:
        if(bestwords != None):
            if(word.lower() in bestwords):
                features.append(word)
        else:
            features.append(word)

    bigram_finder = nltk.BigramCollocationFinder.from_words(reviewWords)
    bigrams = bigram_finder.nbest(nltk.BigramAssocMeasures.chi_sq, 200)
    features.extend(bigrams)

    return features

def getClassifierAccuracy(classifier, featureSet):
    correctPredictions = 0
    for featureList, actualClass in featureSet:
        predicted = classifier.classify(featureList)
        if predicted == actualClass:
            correctPredictions += 1

    return (float(correctPredictions) / len(featureSet))

def extractFeaturesFromSet(dataSet, bestwords):
    i = 0;
    featureSet = []
    for (id, rating, sentiment, words) in dataSet:
        featureSet.append((reviewFeatureExtractor(words, bestwords), sentiment))

        if(i%20==0):
            print (".", end="")
        if(i%1000==0):
            print (str(i))
        i = i + 1;

    print(str(i) + " Finished")
    return featureSet

def getBestWords(trainSet):
    # extract features for each review and store in list of tuples pertaining to each review
    # this is the training data to be passed to the classifier
    word_freq = nltk.probability.FreqDist()
    label_freq = nltk.probability.ConditionalFreqDist()
    stemmer = nltk.PorterStemmer()

    print ("Getting word frequency..")
    i = 0
    for review in trainSet:
        if(review[2] == 'pos'):
            words = [stemmer.stem(x.lower()) for x in review[3]]

            word_freq.update(nltk.probability.FreqDist(words))
            word_freq.update(nltk.probability.FreqDist([x.lower() for x in review[3]]))
            label_freq['pos'].update(nltk.probability.FreqDist([x.lower() for x in review[3]]))
            label_freq['pos'].update(nltk.probability.FreqDist(words))
        elif(review[2] == 'neg'):
            words = [stemmer.stem(x.lower()) for x in review[3]]

            word_freq.update(nltk.probability.FreqDist(words))            
            word_freq.update(nltk.probability.FreqDist([x.lower() for x in review[3]]))
            label_freq['neg'].update(nltk.probability.FreqDist([x.lower() for x in review[3]]))
            label_freq['neg'].update(nltk.probability.FreqDist(words))

        if(i%20==0):
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
        tag = nltk.pos_tag([word])[0][1]
        if (tag.__contains__('VB') or tag.__contains__('NN') or tag.__contains__('RB') or tag.__contains__('JJ')):
            word_scores[word] = pos_score + neg_score

    print("Sorting Word scores..")
    best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:5000]
    print("Getting Best words..")
    bestwords = set([w for w, s in best])

    return bestwords

# //////////////////////////////////////////////////
# MAIN SCRIPT
# //////////////////////////////////////////////////

if __name__ == "__main__":
    # load training reviews from pickled file and randomize the list
    print ("Loading data..")
    reviews = pickle.load(open("./data/train_nofulltext.p", "rb"))
    random.shuffle(reviews)

    # create training and cross-validation feature sets
    trainCutoff = len(reviews) * 4/5
    trainSet = reviews[:trainCutoff]
    cvSet = reviews[trainCutoff:]

    print ("Getting best words..")
    bestwords = getBestWords(trainSet)
    print ("Extracting feature sets..")
    trainFeatureSet = extractFeaturesFromSet(trainSet, bestwords)
    cvFeatureSet = extractFeaturesFromSet(cvSet, bestwords)

    print ("Training model..")
    classifier = NaiveBayesClassifier(trainFeatureSet)

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    for i, (feats, label) in enumerate(cvFeatureSet):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    print ("Training accuracy: ", getClassifierAccuracy(classifier, trainFeatureSet))
    print ("Cross-validation accuracy: ", getClassifierAccuracy(classifier, cvFeatureSet))
    print ("'pos' Precision: ", nltk.precision(refsets['pos'], testsets['pos']))
    print ("'pos' Recall: ", nltk.recall(refsets['pos'], testsets['pos']))
    print ("'neg' Precision: ", nltk.precision(refsets['neg'], testsets['neg']))
    print ("'neg' Recall: ", nltk.recall(refsets['neg'], testsets['neg']))

    classifier.showMostInformativeFeatures(20)
