import warnings
warnings.filterwarnings("ignore")
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from df import X, Y, x, y

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.3, stratify=Y)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(random_state=42))
])

param_grid = {
    "svc__kernel": ['linear', 'rbf', 'poly', 'sigmoid'],
    "svc__C": [0.001, 0.01, 0.1, 1, 10, 100, 500],
    "svc__gamma": np.logspace(-1, 1, 13),
    "svc__max_iter" : [1000, 100000, 1000000]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)
grid.fit(x_train, y_train)

grid.cv_results_
grid.best_params_
res = pd.DataFrame(grid.cv_results_)
res.head()
res = res[['param_svc__kernel', 'param_svc__C', 'param_svc__gamma', 'mean_test_score']]

# Plot for each kernel
kernels = res['param_svc__kernel'].unique()
for kernel in kernels:
    subset = res[res['param_svc__kernel'] == kernel]
    
    # Use pivot_table with mean to handle duplicates due to multiple max_iter values
    pivot = subset.pivot_table(
        index='param_svc__C',
        columns='param_svc__gamma',
        values='mean_test_score',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f'Accuracy for Kernel = {kernel}')
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.tight_layout()
    plt.show()
top_10 = res.sort_values(by='mean_test_score', ascending=False).head(10)

# Print selected columns for readability
print(top_10[['mean_test_score', 'mean_train_score', 'param_svc__kernel', 
              'param_svc__C', 'param_svc__gamma', 'param_svc__max_iter']])

grid.best_params_


# # Get best model and score
# print("Best C:", grid.best_params_['svc__C'])
# print("Best cross-validation accuracy:", grid.best_score_)

# # Final evaluation on test set
# final_accuracy = grid.score(x_test, y_test)
# print("Test set accuracy:", final_accuracy)
# # Testing the loss function on raw data so it can regularize and eliminate features on its own
# x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, random_state=42, test_size=0.3, stratify=y)

# loss_models = []
# loss_models.append(
#     ('Lasso Linear SVC', 
#     LinearSVC(C=100, penalty="l1", loss='squared_hinge', dual=False, random_state=42, max_iter=1000000))
# )
# loss_models.append(
#     ('Ridge Linear SVC', 
#     LinearSVC(C=100, penalty="l2", loss="squared_hinge", dual=True, random_state=42, max_iter=1000000)) # by default is l2 already
# )
# loss_results = []
# loss_names = []

# for name, model in loss_models:
#     kfold = KFold(n_splits=3, random_state=42, shuffle=True)
#     cv_result = cross_val_score(model, x_train2, y_train2, cv=kfold, scoring="accuracy")
#     loss_results.append(cv_result)
#     loss_names.append(name)
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(loss_results)
# ax.set_xticklabels(loss_names)
# plt.show()