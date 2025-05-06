import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from df import X, Y

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_curve,
    auc, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

X = X.drop('spotlight', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
)
grid_search.fit(X_train_scaled, y_train)
best_rf = grid_search.best_estimator_

dt = DecisionTreeClassifier(max_depth=grid_search.best_params_['max_depth'], random_state=42)
voting_clf = VotingClassifier(
    estimators=[('rf', best_rf), ('dt', dt)],
    voting='soft'
)
voting_clf.fit(X_train_scaled, y_train)

y_pred_prob = voting_clf.predict_proba(X_test_scaled)[:, 1]
y_pred_class = voting_clf.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred_class)
roc_auc = roc_auc_score(y_test, y_pred_prob)
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)

print(f"Best Random Forest Params: {grid_search.best_params_}")
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_class))

cm = confusion_matrix(y_test, y_pred_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail', 'Success'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

feat_imp = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(8, 6))
feat_imp.plot(kind='barh')
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

perm = permutation_importance(best_rf, X_test_scaled, y_test, n_repeats=10, random_state=42)
perm_imp = pd.Series(perm.importances_mean, index=X.columns).sort_values()
plt.figure(figsize=(8, 6))
perm_imp.plot(kind='barh')
plt.title("Permutation Importance")
plt.xlabel("Mean Importance (shuffled)")
plt.tight_layout()
plt.show()

