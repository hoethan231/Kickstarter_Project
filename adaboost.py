import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, roc_auc_score

from df import X, Y

# Since the feature "spotlight" is a one to one with "SuccessfulBool" we will drop it to see if the other 
# features are strong enough to predict.
X = X.drop('spotlight', axis=1)

# We have a dataset that consist of ~18000 samples, 10% of the dataset for the test set should
# be good enough.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

models = [] # Will apply cross validation at the end

# Creating a bunch of models through AdaBoost with varying learning rates
rate = 0.05

while rate < 1:
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=500, learning_rate=rate, random_state=42)
    ada.fit(x_train, y_train)
    y_pred = ada.predict(x_test)
    print(f"Accuracy Score of Decision Tree w/ Adaboost w/ Learning rate {rate:.2f}: ", accuracy_score(y_test, y_pred))
    rate+= 0.05
    
    name = "AdaBoost N_EST=500, DEPTH=1, LR=" + str(rate)
    models.append((name, ada))
    
'''
    With the fixed parameters of max_depth=1, n_estimators=500, the best learning rate that we were able to produced was at 
    0.9 learning rate. Now I want to see if we are running too many estimators that will impact the performance of the model.
    We don't want more estimators than needed. 
'''

rate = 0.9
estimators = 50

while estimators < 600:
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=estimators, learning_rate=rate, random_state=42)
    ada.fit(x_train, y_train)
    y_pred = ada.predict(x_test)
    print(f"Accuracy Score of Decision Tree w/ Adaboost w/ Estimator {estimators}: ", accuracy_score(y_test, y_pred))
    estimators += 50
    
    name = "AdaBoost DEPTH=1, LR=0.9, N_EST=" + str(estimators)
    models.append((name, ada))
    
''' 
    After going through different models with a varying number of estimators, we conclude
    that around 250 estimators is good enough and after that we get very slight improvement.
    We don't think that improvement is worth the extra computational power so we will settle 
    with the model at learning rate of 0.9 and estimators of 250.
    
    Out of curiosity, I want to see the number of depths will improve the model or not.
'''

estimators = 50

print("\nAdaboosting with Depth of 2")

while estimators < 600:
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=estimators, learning_rate=rate, random_state=42)
    ada.fit(x_train, y_train)
    y_pred = ada.predict(x_test)
    print(f"Accuracy Score of Decision Tree w/ Adaboost w/ Estimator {estimators}: ", accuracy_score(y_test, y_pred))
    estimators += 50
    
    name = "AdaBoost DEPTH=1, LR=0.9, N_EST=" + str(estimators)
    models.append((name, ada))
    
estimators = 50

print("\nAdaboosting with Depth of 3")
    
while estimators < 600:
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=estimators, learning_rate=rate, random_state=42)
    ada.fit(x_train, y_train)
    y_pred = ada.predict(x_test)
    print(f"Accuracy Score of Decision Tree w/ Adaboost w/ Estimator {estimators}: ", accuracy_score(y_test, y_pred))
    estimators += 50
    
    name = "AdaBoost DEPTH=1, LR=0.9, N_EST=" + str(estimators)
    models.append((name, ada))