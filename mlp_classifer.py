from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

from df import X, Y

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.3, stratify=Y)

clf = MLPClassifier(max_iter=300, random_state=42).fit(x_train, y_train)
scores = cross_val_score(clf, X, Y, cv=5, scoring="f1")
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
print("Target classes:", np.unique(Y, return_counts=True))
