#!/usr/bin/python

import pickle

from tools.feature_format import featureFormat, targetFeatureSplit
from final_project.tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2
from sklearn.model_selection import GridSearchCV
from final_project.tester import test_classifier

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ["poi","bonus", "exercised_stock_options", "director_fees"]  # You will need to use more features

# Load the dictionary containing the dataset
with open("final_project_dataset_unix.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
print("Before Clean:", len(data_dict))

data_dict.pop("TOTAL", 0)

# Task 3: Create new feature(s)


# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
"""
featureFormat(dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False,
                  sort_keys=False):
 convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
"""
data = featureFormat(my_dataset, features_list, sort_keys=True, remove_all_zeroes=False, remove_any_zeroes=False)
print("After Clean:", len(data))
labels, features = targetFeatureSplit(data)

# Task 4: Try a variety of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
clf = KNeighborsClassifier(n_neighbors=3)

names = ["Naive Bayes", "Decision Tree", "Nearest Neighbor", "Random Forest", "AdaBoost"]

classifiers = [GaussianNB(),
               DecisionTreeClassifier(),
               KNeighborsClassifier(n_neighbors=3),
               RandomForestClassifier(),
               AdaBoostClassifier(base_estimator=DecisionTreeClassifier())]
# parameters = {"Naive Bayes":{},
#               "Decision Tree":{"max_depth": range(5,15),
#                                "min_samples_leaf": range(1,5)},
#               "Nearest Neighbor":{"n_neighbors": range(1,10),
#                                   "weights":("uniform", "distance"),
#                                   "algorithm":("auto", "ball_tree", "kd_tree", "brute")},
#               "Random Forest":{"n_estimators": range(2, 5),
#                                "min_samples_split": range(2, 5),
#                                "max_depth": range(2, 15),
#                                "min_samples_leaf": range(1, 5),
#                                "random_state": [0, 10, 23, 36, 42],
#                                "criterion": ["entropy", "gini"]},
#               "AdaBoost":{"base_estimator":[GaussianNB(), DecisionTreeClassifier(), KNeighborsClassifier()],
#                           "n_estimators": range(2, 5),
#                           "algorithm":("SAMME", "SAMME.R"),
#                           "random_state":[0, 10, 23, 36, 42]}}

# clf = Pipeline(
#     [("feature_selection", SelectFromModel(LinearSVC("l1"))), ("reduce_dim", PCA()), ("clf_GaussianNB", GaussianNB()),
#      ("clf_DecisionTree", DecisionTreeClassifier()),
#      ("clf_svc", SVC()), ("clf_neighbor", NearestNeighbors()), ("clf_RandomForest", RandomForestClassifier()),
#      ("clf_AdaBoost", AdaBoostClassifier())])

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

for name, clf in zip(names, classifiers):
    test_classifier(clf, my_dataset, features_list, folds=1000)


# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
