import cPickle as pickle
import random
from trainModel import reviewFeatureExtractor
from NaiveBayesClassifier import NaiveBayesClassifier as NBC
from ModifiedNaiveBayesClassifier import NaiveBayesClassifier as MNBC

reviews = pickle.load(open("./data/train_nofulltext.p", "rb"))
random.shuffle(reviews)
reviews = reviews[:2000]

trainFeatureSet = [(reviewFeatureExtractor(words,None,False), sentiment) for id, rating, sentiment, words in reviews]
trainFeatureSetMod = [(reviewFeatureExtractor(words,None,True), sentiment) for id, rating, sentiment, words in reviews]

classifier = NBC.train(trainFeatureSet)
classifierMod = MNBC(trainFeatureSetMod)

classifier.show_most_informative_features()

# classifier._feature_probdist["pos", ("the", "worst")].prob(True)
# 0.0014464802314368371
# classifierMod.featureClassDist["pos", ("the", "worst")]
# 0.0009652509652509653
# classifierMod.featureClassDist["neg", "awful"]
# 0.10269709543568464
# classifier._feature_probdist["neg", "awful"].prob(True)
# 0.10310880829015544
