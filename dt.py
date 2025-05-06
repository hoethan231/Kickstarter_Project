import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from df import X, Y

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, 
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

X = X.drop('spotlight', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'max_depth': [4, 6, 8, 10],
    'min_samples_leaf': [5, 10, 20],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

best_tree = grid_search.best_estimator_
y_pred_prob = best_tree.predict_proba(X_test_scaled)[:, 1]
y_pred_class = best_tree.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred_class)
roc_auc = roc_auc_score(y_test, y_pred_prob)
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)

print(f"Best Params: {grid_search.best_params_}")
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

feat_imp = pd.Series(best_tree.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(8, 6))
feat_imp.plot(kind='barh')
plt.title("Decision Tree Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

perm = permutation_importance(best_tree, X_test_scaled, y_test, n_repeats=10, random_state=42)
perm_imp = pd.Series(perm.importances_mean, index=X.columns).sort_values()
plt.figure(figsize=(8, 6))
perm_imp.plot(kind='barh')
plt.title("Permutation Importance")
plt.xlabel("Mean Importance (shuffled)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 8))
plot_tree(best_tree, filled=True, feature_names=X.columns, class_names=['Fail', 'Success'], fontsize=9)
plt.title("Best Decision Tree Structure")
plt.tight_layout()
plt.show()
