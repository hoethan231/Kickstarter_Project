import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from df import X, Y, x, y
scaler = StandardScaler().fit(X)
scaledX = scaler.transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.3, stratify=Y)

models = []
models.append((
    'Linear SVC', 
    SVC(kernel = 'linear', C=100, random_state=42, max_iter=1000000)))
# Good for linear relationships

models.append((
    'Radical Kernel SVC', 
    SVC(kernel = 'rbf', degree = 2, C=100, random_state=42, max_iter =1000000)))
# Good default choice for non-linear data

models.append((
    'Poly Kernal SVC',
    SVC(kernel="poly", degree=2, C=100, random_state=42, max_iter=1000000)
))
# Also good for non linear data

models.append((
    'Sigmoid Kernal SVC',
    SVC(kernel="sigmoid", degree=2, C=100, random_state=42, max_iter=1000000)
))
# Similar to logistic regression and good for binary classification
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=3, random_state=42, shuffle=True)
    cv_result = cross_val_score(model, x_train, y_train, cv=kfold, scoring="accuracy")
    results.append(cv_result)
    names.append(name)
    
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

radical = models[1][1].fit(x_train, y_train)
radical_pred_proba = radical.predict(x_test)
radical_score = roc_auc_score(y_test, radical_pred_proba, average=None, multi_class='ovr')
print(radical_score)

fpr, tpr, _ = roc_curve(y_test, radical_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"NN Model (AUC = {radical_score:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Neural Network")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# The Radical SVC seems to have slightly better results with the decent speard and higher mean. 
# I will use this to explore the hyperparameters and test it with different penalty functions
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = 'rbf', random_state=42, max_iter=1000000))
])

param_grid = {
    "svc__C" : [0.001, 0.01, 0.1, 1, 10, 100, 500]
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
results = grid.cv_results_

mean_scores = results['mean_test_score']
c_values = results['param_svc__C'].data
mean_val_scores   = grid.cv_results_['mean_test_score']
mean_train_scores = grid.cv_results_['mean_train_score']
C_vals            = grid.cv_results_['param_svc__C'].data

plt.figure()
plt.semilogx(C_vals, mean_train_scores, marker='o', label='train')
plt.semilogx(C_vals, mean_val_scores,   marker='s', label='cv-validation')
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Training vs. CV accuracy – RBF-SVC')
plt.legend(); plt.grid(True); plt.show()

# Get best model and score
print("Best C:", grid.best_params_['svc__C'])
print("Best cross-validation accuracy:", grid.best_score_)

# Final evaluation on test set
final_accuracy = grid.score(x_test, y_test)
print("Test set accuracy:", final_accuracy)

# Testing the loss function on raw data so it can regularize and eliminate features on its own
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, random_state=42, test_size=0.3, stratify=y)

loss_models = []
loss_models.append(
    ('Lasso Linear SVC', 
    LinearSVC(C=100, penalty="l1", loss='squared_hinge', dual=False, random_state=42, max_iter=1000000))
)
loss_models.append(
    ('Ridge Linear SVC', 
    LinearSVC(C=100, penalty="l2", loss="squared_hinge", dual=True, random_state=42, max_iter=1000000)) # by default is l2 already
)

loss_results = []
loss_names = []

for name, model in loss_models:
    kfold = KFold(n_splits=3, random_state=42, shuffle=True)
    cv_result = cross_val_score(model, x_train2, y_train2, cv=kfold, scoring="accuracy")
    loss_results.append(cv_result)
    loss_names.append(name)
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(loss_results)
ax.set_xticklabels(loss_names)
plt.show()

def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k')
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal', adjustable='box')

gamma_range = np.logspace(-3, 3, 13)
param_grid   = {'gamma': gamma_range}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    SVC(kernel='rbf'),
    param_grid,
    cv=cv,
    return_train_score=True,
    n_jobs=-1
)

grid.fit(x_train, y_train)
print(f"Best γ = {grid.best_params_['gamma']}, "
      f"mean CV accuracy = {grid.best_score_:.3f}")

plt.figure(figsize=(6, 4))
plt.semilogx(gamma_range,
             grid.cv_results_['mean_test_score'],
             marker='o')
plt.axvline(grid.best_params_['gamma'], linestyle='--')
plt.xlabel('γ (log scale)')
plt.ylabel('Mean CV accuracy')
plt.title('Radial SVM – Cross-validated accuracy vs γ')
plt.grid(True)
plt.tight_layout()
plt.show()

n_γ  = len(gamma_range)
cols = 4
rows = int(np.ceil(n_γ / cols))

plt.figure(figsize=(cols * 3.5, rows * 3))
for idx, γ in enumerate(gamma_range):
    svm = SVC(kernel='rbf', gamma=γ).fit(X, y)
    plt.subplot(rows, cols, idx + 1)
    plot_decision_boundary(svm, x_test, y_test, f'γ={γ:g}')
plt.suptitle('Decision boundaries for each γ', y=1.02, fontsize=14)
plt.tight_layout()
plt.show()