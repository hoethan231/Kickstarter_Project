import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, roc_auc_score, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import Normalizer

from df import X, Y

norm_scaler = Normalizer()

# X['goal'] = np.log(X['goal'])
# X['goal'] = np.sqrt(X['goal'])
# X['goal'] = norm_scaler.fit_transform(X[['goal']])

# We have a dataset that consist of ~18000 samples, 10% of the dataset for the test set should
# be good enough.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

models = [] # Will apply cross validation at the end
best_ada = None

# # Creating a bunch of models through AdaBoost with varying learning rates
# rate = 0.05

# while rate < 1:
#     ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=700, learning_rate=rate, random_state=42)
#     ada.fit(x_train, y_train)
#     y_pred = ada.predict(x_test)
#     print(f"Accuracy Score of Decision Tree w/ Adaboost w/ Learning rate {rate:.2f}: ", accuracy_score(y_test, y_pred))
#     rate+= 0.05
    
#     name = "AdaBoost N_EST=500, DEPTH=1, LR=" + str(rate)
#     models.append((name, ada))
    
# rate = 0.05

# while rate < 1:
#     ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=700, learning_rate=rate, random_state=42)
#     ada.fit(x_train, y_train)
#     y_pred = ada.predict(x_test)
#     print(f"Accuracy Score of Decision Tree w/ Adaboost w/ Learning rate {rate:.2f}: ", accuracy_score(y_test, y_pred))
#     rate+= 0.05
    
#     name = "AdaBoost N_EST=500, DEPTH=2, LR=" + str(rate)
#     models.append((name, ada))

# rate = 0.05

# while rate < 1:
#     ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=700, learning_rate=rate, random_state=42)
#     ada.fit(x_train, y_train)
#     y_pred = ada.predict(x_test)
#     print(f"Accuracy Score of Decision Tree w/ Adaboost w/ Learning rate {rate:.2f}: ", accuracy_score(y_test, y_pred))
#     rate+= 0.05
    
#     name = "AdaBoost N_EST=500, DEPTH=2, LR=" + str(rate)
#     models.append((name, ada))

'''
    With the fixed parameters of max_depth=1, n_estimators=500, the best learning rate that we were able to produced was at 
    0.3 learning rate. Now I want to see if we are running too many estimators that will impact the performance of the model.
    We don't want more estimators than needed. 
'''

# rate = 0.30
# estimators = 50

# while estimators < 600:
#     ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=estimators, learning_rate=rate, random_state=42)
#     ada.fit(x_train, y_train)
#     y_pred = ada.predict(x_test)
#     print(f"Accuracy Score of Decision Tree w/ Adaboost w/ Estimator {estimators}: ", accuracy_score(y_test, y_pred))
#     estimators += 50
    
#     name = "AdaBoost DEPTH=1, LR=0.3, N_EST=" + str(estimators)
#     models.append((name, ada))
    
''' 
    After going through different models with a varying number of estimators, we conclude
    that around 250 estimators is good enough and after that we get very slight improvement.
    We don't think that improvement is worth the extra computational power so we will settle 
    with the model at learning rate of 0.3 and estimators of 250.
    
    Out of curiosity, I want to see the number of depths will improve the model or not.
'''

# rate = 0.30
# estimators = 50

# print("\nAdaboosting with Depth of 2")

# while estimators < 600:
#     ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=estimators, learning_rate=rate, random_state=42)
#     ada.fit(x_train, y_train)
#     y_pred = ada.predict(x_test)
#     print(f"Accuracy Score of Decision Tree w/ Adaboost w/ Estimator {estimators}: ", accuracy_score(y_test, y_pred))
#     estimators += 50
    
#     name = "AdaBoost DEPTH=2, LR=0.3, N_EST=" + str(estimators)
#     models.append((name, ada))
#     if (estimators == 550):
#         best_ada = ada
        
    
estimators = 50
rate = 0.95
print("\nAdaboosting with Depth of 3")
    
while estimators < 600:
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=estimators, learning_rate=rate, random_state=42)
    ada.fit(x_train, y_train)
    y_pred = ada.predict(x_test)
    print(f"Accuracy Score of Decision Tree w/ Adaboost w/ Estimator {estimators}: ", accuracy_score(y_test, y_pred))
    estimators += 50
    
    name = "AdaBoost DEPTH=3, LR=0.3, N_EST=" + str(estimators)
    models.append((name, ada))
    if (estimators == 400):
        best_ada = ada
    
    
# Applying Cross Validation to each model
# results_accuracy = []
# names_accuracy = []
# scoring = 'accuracy'

# for name, model in models:
#     kfold = KFold(n_splits=5)
#     cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
#     results_accuracy.append(cv_results)
#     names_accuracy.append(name)
    
# # Box Plot Comparison of K-Fold Cross Validation Results from Accuracy Score
# fig = plt.figure()
# fig.suptitle("Cross Validation Comparison Comparison")
# ax = fig.add_subplot(111)
# plt.boxplot(results_accuracy)
# ax.set_xticklabels(names_accuracy)
# plt.figure(figsize=(25, 6))
# plt.show()
    
# From all these iterations, we have selected the model we though is the best considering performance and computation

# Now I want to plot the confusion matrix and other stuff to visualize the performance

y_pred_prob = best_ada.predict_proba(x_test)[:, 1]
y_pred = best_ada.predict(x_test)

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)

print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail', 'Success'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

feat_imp = pd.Series(best_ada.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(8, 6))
feat_imp.plot(kind='barh')
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

perm = permutation_importance(best_ada, x_test, y_test, n_repeats=10, random_state=42)
perm_imp = pd.Series(perm.importances_mean, index=X.columns).sort_values()
plt.figure(figsize=(8, 6))
perm_imp.plot(kind='barh')
plt.title("Permutation Importance")
plt.xlabel("Mean Importance (shuffled)")
plt.tight_layout()
plt.show()

'''
    The reason I am NOT doing grid search optimization is because I am also looking into the computational power. Since 
    my laptop is very limited, I want to ensure parameters chosen will make the most impact rather than the best impact on 
    the model. If a parameter tuned higher where the model's accuracy barely improves, then I will chose the parameter when it 
    first hits that plateau. So I first look at the learning rate with a max depth of 1 and 500 esimators. I know 500 esimators is 
    quite a lot so I know this parameter is overtuned. I will do this to reduce the load on my computer.
    
    So while considering computational power of my laptop, I will consider the following hyper parameters:
        Learning Rate = 0.3
        # of Estimators = 200
        Max Depth = 2
        
    If we are not considering computational power as a factor, I observed through cross validation of all the models provided here and with minimum variance 
    and high average accuracy score, we can see that the following hyperparameters are chosen below:
        Learning Rate = 0.3
        # of Estimators = 550
        Max Depth = 2
        
    The best performance considering computation and parameter tuning based on our EDA of the dataset is:
        Accuracy: 0.7954
        ROC AUC: 0.8548
        Precision-Recall AUC: 0.6909
    
    Classification Report:
                    precision    recall  f1-score   support

                0       0.83      0.90      0.86      1297
                1       0.69      0.53      0.60       531

            accuracy                        0.80      1828
           macro avg    0.76      0.72      0.73      1828
        weighted avg    0.79      0.80      0.79      1828
        
    This is for the # of Estimators with 550.
'''

# Best: 0.763211 using {'estimator__max_depth': 3, 'learning_rate': 1, 'n_estimators': 600}