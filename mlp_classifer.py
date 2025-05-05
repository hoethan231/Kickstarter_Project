from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np

from df import X, Y

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.3, stratify=Y)

iterations = [200, 250, 300, 350, 400, 450, 500]
scores = []
for inter in iterations:
    clf = MLPClassifier(max_iter=300, random_state=42).fit(x_train, y_train)
    score = cross_val_score(clf, X, Y, cv=5, scoring="f1").mean()
    scores.append(score)

print(scores)
# It seems like even after changing the interations the model's f1 score doesn't seem to change. 
