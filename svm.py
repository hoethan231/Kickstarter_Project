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
    "svc__gamma": np.logspace(-1, 1, 5),
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

print(grid.best_params_)

res = pd.DataFrame(grid.cv_results_)
res.head()
res = res[['param_svc__kernel', 'param_svc__C', 'param_svc__gamma', 'mean_test_score', 'param_svc__max_iter']]

kernels = res['param_svc__kernel'].unique()
for kernel in kernels:
    subset = res[res['param_svc__kernel'] == kernel]
    
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
    
top_model_params = res.sort_values(by='mean_test_score', ascending=False)
print(top_model_params)
grid.best_params_
best_rbf = res[
    (res['param_svc__kernel'] == 'rbf') &
    (res['param_svc__C'] == 1.0) &
    (res['param_svc__gamma'] == 3.1622776601683795)
]

best_rbf = best_rbf.sort_values(by='param_svc__max_iter')

plt.figure(figsize=(8, 5))
plt.plot(best_rbf['param_svc__max_iter'], best_rbf['mean_test_score'], marker='o')
plt.title("Accuracy vs. max_iter for Best RBF Parameters")
plt.xlabel("max_iter")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# Testing the loss function on raw data so it can regularize and eliminate features on its own
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, random_state=42, test_size=0.3, stratify=y)

loss_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LinearSVC(loss='squared_hinge', random_state=42, max_iter=1000000))
])

param_grid_l1 = {
    'clf__penalty': ['l1'],
    'clf__dual': [False],
    'clf__C': [0.01, 0.1, 1, 10]
}

param_grid_l2 = {
    'clf__penalty': ['l2'],
    'clf__dual': [True, False],
    'clf__C': [0.01, 0.1, 1, 10]
}

grid_l1 = GridSearchCV(loss_pipeline, param_grid=param_grid_l1, scoring='accuracy', cv=3, n_jobs=-1)
grid_l2 = GridSearchCV(loss_pipeline, param_grid=param_grid_l2, scoring='accuracy', cv=3, n_jobs=-1)
grid_l1.fit(x_train2, y_train2)
grid_l2.fit(x_train2, y_train2)

coef_l1 = grid_l1.best_estimator_.named_steps['clf'].coef_.ravel()
coef_l2 = grid_l2.best_estimator_.named_steps['clf'].coef_.ravel()

plt.figure(figsize=(10, 5))
plt.plot(coef_l1, label="L1 penalty", marker='o')
plt.plot(coef_l2, label="L2 penalty", marker='x')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Feature Coefficients: L1 vs L2 Regularization")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.legend()
plt.grid(True)
plt.show()

df_l1 = pd.DataFrame(grid_l1.cv_results_)
df_l2 = pd.DataFrame(grid_l2.cv_results_)

df_l1["Penalty"] = "L1"
df_l2["Penalty"] = "L2"

df_all = pd.concat([df_l1, df_l2], ignore_index=True)

df_plot = df_all[["param_clf__C", "mean_test_score", "Penalty"]]

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_plot, x="param_clf__C", y="mean_test_score", hue="Penalty")
plt.title("LinearSVC Accuracy by C Value and Penalty Type")
plt.xlabel("C Value")
plt.ylabel("Mean Cross-Validated Accuracy")
plt.legend(title="Penalty")
plt.grid(True)
plt.tight_layout()
plt.show()

df_l1_coef = pd.DataFrame({
    "Feature": x.columns.tolist(),
    "Coefficient": coef_l1
})

df_l1_coef["AbsCoeff"] = df_l1_coef["Coefficient"].abs()
df_l1_top = df_l1_coef.sort_values(by="AbsCoeff", ascending=False)

print("Top 10 L1 features by importance:")
print(df_l1_top.head(10))
df_l2_coef = pd.DataFrame({
    "Feature": x.columns.tolist(),
    "Coefficient": coef_l2
})

df_l2_coef["AbsCoeff"] = df_l2_coef["Coefficient"].abs()
df_l2_top = df_l2_coef.sort_values(by="AbsCoeff", ascending=False)

print("\nTop 10 L2 features by importance:")
print(df_l2_top.head(10))