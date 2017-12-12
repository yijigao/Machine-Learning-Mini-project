#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""

import pickle
import numpy
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans

sys.path.append("../tools/")
from tools.feature_format import featureFormat, targetFeatureSplit


def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    # plot each cluster with a different color--add more colors for
    # drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    # if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


# load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb"))
# there's an outlier--remove it!
data_dict.pop("TOTAL", 0)

# print("Before Clean NaN len(data_dict):", len(data_dict))
# NaN_salary = [name for name in data_dict if data_dict[name]["salary"] == "NaN"]
# for name in NaN_salary:
# 	data_dict.pop(name,0)
# NaN_stock = [name for name in data_dict if data_dict[name]["exercised_stock_options"] == "NaN"]
# for name in NaN_stock:
# 	data_dict.pop(name, 0)
# print("After Clean NaN len(data_dict):",len(data_dict))


### the input features we want to use
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi = "poi"
features_list = [poi, feature_1, feature_2]
# features_list = [feature_1, feature_2]
# features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)

# MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

stocks = [stock for _, stock in finance_features]
salary = [salary for salary, _ in finance_features]
stocks_reshape = numpy.reshape(numpy.array(stocks), (len(stocks), 1))
salary_reshape = numpy.reshape(numpy.array(salary), (len(salary), 1))
print("Min salary:", min(salary_reshape))
print("Min stock:", min(stocks_reshape))

scaler = MinMaxScaler()
# finance_features_reshape = numpy.reshape(numpy.array(finance_features), (len(finance_features), 1))
scaler.fit(finance_features)
print(scaler.transform([[200000., 1000000.]]))
scaled_finance_features = scaler.transform(finance_features)

scaler.fit(salary_reshape)
print("rescaled salary 200000:", scaler.transform(200000.))
scaler.fit(stocks_reshape)
print("rescaled stock 1000000:", scaler.transform(1000000.))

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in scaled_finance_features:
    plt.scatter(f1, f2)
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
kmeans = KMeans(n_clusters=2).fit(scaled_finance_features)
pred = kmeans.predict(scaled_finance_features)

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, scaled_finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("no predictions object named pred found, no clusters to plot")
