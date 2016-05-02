## COMP 5450 - Machine Learning

### Sentiment Analysis in Movie Reviews using Naive Bayes

This project contains the data and code for the UML CS5450 Machine Learning Spring 2016 final project. The purpose of this project is to classify a movie review as either positive or negative. The training and test are given in the `data` directory. This includes 25,000 labeled reviews for training and 11,000 unlabeled reviews for evaluation.

In order to classify the data, we trained a Naive Bayes classifier on features extracted out of each review. Feature extraction was accomplished with Python and [NLTK](http://www.nltk.org/). Feature processes we used included word bigrams, best word distribution by part-of-speech, and word stemming.

The project report can be found [here](http://jtlucas.github.io/cs5450/).

#### Running the project

1. [Install NLTK](http://www.nltk.org/install.html)
2. Parse training data with `python parseAndStoreReviews.py`
3. Train the classifier with `python trainModel.py`. This will automatically take 80% of the training data to train with, and 20% for cross-validation. Accuracy, precision, and recall will be printed along with the most informative features.
4. Evaluate the classifier with `python testModel.py`. This will parse the unlabeled test reviews and generate an output file with the predicted sentiment.
