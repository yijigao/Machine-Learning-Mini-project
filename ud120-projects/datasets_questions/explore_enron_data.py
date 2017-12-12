#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
from tools.convert_to_unix import convert_pkl
from final_project.poi_email_addresses import poiEmails

import numpy as np
from collections import defaultdict
import pandas as pd

convert_pkl("../final_project/final_project_dataset.pkl")
enron_data = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb"))
enron_data_df = pd.DataFrame(enron_data)

print("How Many People In dataset? ", len(enron_data))
print("For each person, how many features are available ? ")
print("DataFrame info:", enron_data_df.info())
print()

# print(enron_data_df.loc["poi"])
counter1 = 0
for e in enron_data_df.loc["poi"]:
    if e == True:
        counter1 += 1
print("How many POI in the dataset? ", counter1)

email_list = poiEmails()
poi_names = []
with open("../final_project/poi_names.txt", "r", encoding="utf-8", newline=None) as f:
    k = 0
    for line in f:
        if k > 1:
            poi_names.append(line)
        k += 1
    f.close()

poi_n = [name for name in poi_names if "(n)" in name]
poi_y = [name for name in poi_names if "(y)" in name]
print("How many POI total in my list? ", len(poi_n) + len(poi_y))

## Looking Dataset
print("What is the total value of the stock belonging to James Prentice?",
      enron_data["PRENTICE JAMES"]["total_stock_value"])
# print(enron_data_df.loc["total_stock_value", "PRENTICE JAMES"])
print("How many messages do we have from Wesley Colwell to POIs? ",
      enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])
print("What's the value of stock options exercised by Jeff Skilling? ",
      enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])
print("Of lay, Skilling and Fastow, who took home the most money? how much many was it?", "\n",
      enron_data_df.loc["total_payments", ["LAY KENNETH L", "SKILLING JEFFREY K", "FASTOW ANDREW S"]])

folks_salary_not_NaN = [salary for salary in enron_data_df.loc["salary"] if salary != "NaN"]
folks_email_not_NaN = [email for email in enron_data_df.loc["email_address"] if email != "NaN"]
print("How many folks in this dataset have a quantified salary? Known email address?\n",
      "folks salary not NaN:", len(folks_salary_not_NaN), "\n",
      "emails not NaN:", len(folks_email_not_NaN))
print()
payments_NaN = [name for name in enron_data if enron_data[name]["total_payments"] == "NaN"]
print("What percentage of people in the dataset have 'NaN' for their total payments? \n",
      "Number of NaN total payments: ", len(payments_NaN), "\n",
      "Percentage: ", round(100 * float(len(payments_NaN) / len(enron_data)), 3))
print()
poi_list = set()
for name in enron_data:
    if enron_data[name]["poi"]:
        poi_list.add(name)
print("poi_name_list:", poi_list)
print()
poi_NaN_payments = [name for name in poi_list if enron_data[name]["total_payments"] == "NaN"]

print("What percentage of POI's in the dataset has 'NaN' for their total payments?\n",
      "Number of NaN total payments POI: ", len(poi_NaN_payments), "\n",
      "Percentage: ", round(100 * float(len(poi_NaN_payments) / len(enron_data)), 3))

# Add 10 new poi to poi_list, and set their total_payments to 'NaN'
a = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
enron_data_new = enron_data

for name in a:
    enron_data_new[name] = {}
    enron_data_new[name]["poi"] = True
    enron_data_new[name]["total_payments"] = "NaN"

poi_list_new = [name for name in enron_data_new if enron_data_new[name]["poi"]]
print("New Number of poi list: ", len(poi_list_new))
poi_new_NaN_payments = [name for name in poi_list_new if enron_data_new[name]["total_payments"] == "NaN"]
print("New Number of poi wit 'NaN' total_payments: ", len(poi_new_NaN_payments))
