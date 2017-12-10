#!/usr/bin/python

import pickle
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

numpy.random.seed(42)

# The words (features) and authors (labels), already largely processed.
# These files should have been created from the previous (Lesson 10)
# mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load(open(words_file, "rb"))
authors = pickle.load(open(authors_file, "rb"))

# test_size is the percentage of events assigned to the test set (the
# remainder go into training)
# feature matrices changed to dense representations for compatibility with
# classifier functions in versions 0.15.2 and earlier
# from sklearn import cross_validation


features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors,
                                                                            test_size=0.1,
                                                                            random_state=42)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()

# how many train points?


# a classic way to overfit is to use a small number
# of data points and a large number of features;
# train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train = labels_train[:150]

# your code goes here
print("Train points:", features_train.shape[0])

# create a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
# My training score of Decision Tree
print("Test Score: ", clf.score(features_test, labels_test))

# Finding the most important feature
feature_imp_list = clf.feature_importances_
i = 0
for feature_imp in feature_imp_list:
    if feature_imp > 0.2:
        print("the importance of the significance:", feature_imp,
              "\nThe number of this feature:", i)
    i += 1

# find the most important word in the feature list
word_list = vectorizer.get_feature_names()
print("The most important word:", word_list[21323])

