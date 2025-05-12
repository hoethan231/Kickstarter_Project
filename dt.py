import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    precision_recall_curve, auc, f1_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# === Load and preprocess
filepath = "/Users/prathamsaxena/Downloads/SJSU/CMPE 188/ML Code/ML Data/kickstarter_data_full.csv"

cols = ['goal', 'state', 'country', 'static_usd_rate', 'category', 'created_at_yr',
        'create_to_launch_days', 'launch_to_deadline_days', 'blurb_len_clean', 'SuccessfulBool']
df = pd.read_csv(filepath, usecols=cols)

df = df.dropna()
df = df[df['state'] != 'live'].drop(columns='state')
df['goal'] = df['goal'].astype(float) * df['static_usd_rate'].astype(float)
df = df.drop(columns='static_usd_rate')

encoder = LabelEncoder()
for col in ['category', 'country']:
    df[col] = encoder.fit_transform(df[col].astype(str))

# === Feature Engineering
df['goal_log'] = np.log1p(df['goal'])
df['goal_log_bucket'] = pd.qcut(df['goal_log'], q=4, labels=False)
df['blurb_density'] = df['blurb_len_clean'] / (df['launch_to_deadline_days'] + 1)
df['goal_per_day'] = df['goal_log'] / (df['launch_to_deadline_days'] + 1)
df['create_to_launch_log'] = np.log1p(df['create_to_launch_days'])
df['launch_to_deadline_log'] = np.log1p(df['launch_to_deadline_days'])
df['goal_blurb_ratio'] = df['goal_log'] / (df['blurb_len_clean'] + 1)
df['goal_to_blurb_log'] = np.log1p(df['goal_blurb_ratio'])
df['create_deadline_ratio'] = df['create_to_launch_log'] / (df['launch_to_deadline_log'] + 1)
df['interaction1'] = df['goal_per_day'] * df['goal_log_bucket']
df['interaction2'] = df['blurb_density'] * df['goal_log_bucket']
df['goal_duration_ratio'] = df['goal_log'] / (df['launch_to_deadline_log'] + 1)
df['blurb_launch_ratio'] = df['blurb_density'] / (df['create_to_launch_log'] + 1)
df = df.drop(columns='goal')

X = df.drop(columns='SuccessfulBool')
y = df['SuccessfulBool'].astype(int)

# === Scaling and SMOTE balancing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(X_scaled, y)

# === Feature pruning
rf_temp = RandomForestClassifier(n_estimators=200, random_state=42)
rf_temp.fit(X_bal, y_bal)
perm = permutation_importance(rf_temp, X_bal, y_bal, n_repeats=5, random_state=42)
top_indices = perm.importances_mean.argsort()[::-1][:25]
X_top = X_bal[:, top_indices]
feature_names = np.array(X.columns)[top_indices]

# === Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_top, y_bal, test_size=0.3, random_state=42)

# === Extended Hyperparameter Grid
param_grid = {
    'max_depth': [10, 15, 20, 25, 30, None],
    'min_samples_leaf': [1, 5, 10, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_impurity_decrease': [0.0, 0.0001, 0.001],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced'],
    'ccp_alpha': [0.0, 0.0001, 0.0005, 0.001]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# === Threshold Optimization
best_tree = grid_search.best_estimator_
y_pred_prob = best_tree.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0.3, 0.7, 50)
scores = [roc_auc_score(y_test, y_pred_prob > t) for t in thresholds]
best_thresh = thresholds[np.argmax(scores)]
y_pred = (y_pred_prob > best_thresh).astype(int)

# === Evaluation
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
f1 = f1_score(y_test, y_pred)

# === Print Results
print("\n=== Final Upgraded Decision Tree ===")
print(f"Best Params       : {grid_search.best_params_}")
print(f"Best Threshold    : {best_thresh:.4f}")
print(f"Accuracy          : {acc:.4f}")
print(f"ROC AUC           : {roc_auc:.4f}")
print(f"PR AUC            : {pr_auc:.4f}")
print(f"F1 Score          : {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
