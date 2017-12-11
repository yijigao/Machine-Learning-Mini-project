#!/usr/bin/python
import pickle
import sys
from tools.feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

data_dict = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb"))

# first element is our labels, any added elements are predictor
# features. Keep this the same for the mini-project, but you'll
# have a different feature list when you do the final project.
features_list = ["poi", "salary"]


data = featureFormat(data_dict, features_list, sort_keys="../tools/python2_lesson13_keys_unix.pkl")
labels, features = targetFeatureSplit(data)

# it's all yours from here forward!
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            test_size=0.3, random_state=42)
#
# # using Decision tree as classifier
clf = DecisionTreeClassifier()
# # training
# clf.fit(features, labels)
clf.fit(features_train, labels_train)
# # # Score
# print("Test score:", clf.score(features, labels))
print("Test score:", clf.score(features_test, labels_test))