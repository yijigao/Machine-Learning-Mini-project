#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
from tools.feature_format import featureFormat, targetFeatureSplit
sys.path.append("../tools/")


# read in data dictionary, convert to numpy array
data_dict = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb"))

data_dict.pop("TOTAL")

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

# your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)
    if salary >= 1000000 and bonus >= 5000000:
        print(point)

plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.show()
