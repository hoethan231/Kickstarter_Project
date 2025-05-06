import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from df import X, Y, x, y
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.3, stratify=Y)
models = []
models.append((
    'Linear SVC', 
    LinearSVC(C=100, loss = 'hinge', random_state=42, max_iter=1000000)))
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
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_result = cross_val_score(model, x_train, y_train, cv=kfold, scoring="accuracy")
    results.append(cv_result)
    names.append(name)
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results[:-1]) # Exclude the Sigmoid it was so poo poo
ax.set_xticklabels(names[:-1])
plt.show()
linearSVC = models[0][1].fit(x_train, y_train)
polySCE = models[2][1].fit(x_train, y_train)
print(linearSVC.predict(y_test))
print(polySCE.predict(y_test))
# The Linear SVC seems to have slightly better results with the smaller speard and higher mean. 
# I will use this to explore the hyperparameters but will check again later with the poly kernal as well.
# Now testing the penalty function
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", LinearSVC(random_state=42, max_iter=1000000))
])

param_grid = {
    "svc__C" : [0.001, 0.01, 0.1, 1, 10, 100, 500]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1
)

grid.fit(x_train, y_train)
results = grid.cv_results_

mean_scores = results['mean_test_score']
c_values = results['param_svc__C'].data

plt.figure()
plt.semilogx(c_values, mean_scores, marker='o')
plt.xlabel('C (log scale)')
plt.ylabel('Mean CV Accuracy')
plt.ylim(0,1)
plt.title('Validation Curve for LinearSVC')
plt.grid(True)
plt.show()
# Get best model and score
print("Best C:", grid.best_params_['svc__C'])
print("Best cross-validation accuracy:", grid.best_score_)

# Final evaluation on test set
final_accuracy = grid.score(x_test, y_test)
print("Test set accuracy:", final_accuracy)
# Testing the loss function on raw data so it can regularize and eliminate features on its own
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, random_state=42, test_size=0.3, stratify=Y)

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
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    cv_result = cross_val_score(model, x_train2, y_train2, cv=kfold, scoring="accuracy")
    loss_results.append(cv_result)
    loss_names.append(name)
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(loss_results)
ax.set_xticklabels(loss_names)
plt.show()