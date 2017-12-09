#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf = SVC(C=10000, kernel="rbf")

"""
加快算法速度的一种方式是在一个较小的训练数据集上训练它。这样做换来的是准确率几乎肯定下降
"""
## 缩小训练数据集, 切割至原始大小的1%, 丢弃掉99%的训练数据
#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]


t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")
t1 = time()
pred = clf.predict(features_test)
print("predicting time:", round(time()-t0, 3), "s")

count_Chris = 0
#print("10: ", pred[10])
#print("26: ", pred[26])
#print("50: ", pred[50])
for i in pred:
	if i == 1:
		count_Chris  += 1

print("There are ", count_Chris, "emails predicted to be in Chris!")
print("accuracy score:",accuracy_score(pred,labels_test))
#clf.score(features_test, labels_test)
#########################################################


