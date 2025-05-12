import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(X_scaled, y)

rf_temp = RandomForestClassifier(n_estimators=200, random_state=42)
rf_temp.fit(X_bal, y_bal)
importance = permutation_importance(rf_temp, X_bal, y_bal, n_repeats=5, random_state=42)
top_indices = importance.importances_mean.argsort()[::-1][:25]
X_bal_pruned = X_bal[:, top_indices]
X_pruned = X_scaled[:, top_indices]

rf = RandomForestClassifier(n_estimators=400, max_depth=20, min_samples_leaf=2, min_samples_split=4,
                            class_weight='balanced_subsample', max_features='sqrt', random_state=42)

et = ExtraTreesClassifier(n_estimators=400, max_depth=20, min_samples_leaf=2, min_samples_split=4,
                          class_weight='balanced', bootstrap=True, max_samples=0.85,
                          max_features='sqrt', random_state=42)

xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.037, subsample=0.85,
                    colsample_bytree=0.9, scale_pos_weight=2, reg_alpha=0.5, gamma=0.35,
                    min_child_weight=3, use_label_encoder=False, eval_metric='logloss',
                    verbosity=0, random_state=42)

cat = CatBoostClassifier(iterations=400, learning_rate=0.035, depth=6, l2_leaf_reg=4,
                         random_seed=42, verbose=0)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
meta_features = np.zeros((X_bal_pruned.shape[0], 4))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_bal_pruned, y_bal)):
    X_tr, X_val = X_bal_pruned[train_idx], X_bal_pruned[val_idx]
    y_tr, y_val = y_bal[train_idx], y_bal[val_idx]

    rf.fit(X_tr, y_tr)
    et.fit(X_tr, y_tr)
    xgb.fit(X_tr, y_tr)
    cat.fit(X_tr, y_tr)

    meta_features[val_idx, 0] = rf.predict_proba(X_val)[:, 1]
    meta_features[val_idx, 1] = et.predict_proba(X_val)[:, 1]
    meta_features[val_idx, 2] = xgb.predict_proba(X_val)[:, 1]
    meta_features[val_idx, 3] = cat.predict_proba(X_val)[:, 1]

meta_nn = CalibratedClassifierCV(MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, alpha=1e-3, random_state=42), method='isotonic', cv=3)
meta_xgb = CalibratedClassifierCV(XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                                subsample=0.9, colsample_bytree=0.9,
                                                use_label_encoder=False, eval_metric='logloss',
                                                verbosity=0, random_state=42), method='isotonic', cv=3)

meta_nn.fit(meta_features, y_bal)
meta_xgb.fit(meta_features, y_bal)

X_meta = np.column_stack([
    rf.predict_proba(X_pruned)[:, 1],
    et.predict_proba(X_pruned)[:, 1],
    xgb.predict_proba(X_pruned)[:, 1],
    cat.predict_proba(X_pruned)[:, 1]
])
nn_preds = meta_nn.predict_proba(X_meta)[:, 1]
xgb_preds = meta_xgb.predict_proba(X_meta)[:, 1]

best_score, best_weight = 0, 0.5
for w in np.linspace(0.3, 0.7, 21):
    y_prob = w * nn_preds + (1 - w) * xgb_preds
    score = 0.6 * f1_score(y, y_prob > 0.5) + 0.4 * roc_auc_score(y, y_prob)
    if score > best_score:
        best_score = score
        best_weight = w

y_prob = best_weight * nn_preds + (1 - best_weight) * xgb_preds
thresholds = np.linspace(0.2, 0.6, 50)
scores = [0.6 * f1_score(y, y_prob > t) + 0.4 * roc_auc_score(y, y_prob) for t in thresholds]
best_thresh = thresholds[np.argmax(scores)]
y_pred = (y_prob > best_thresh).astype(int)

acc = accuracy_score(y, y_pred)
roc = roc_auc_score(y, y_prob)
f1 = f1_score(y, y_pred)
precision, recall, _ = precision_recall_curve(y, y_prob)
pr_auc = auc(recall, precision)

print(f"\n=== Final Blended Meta Ensemble (Calibrated + Pruned) ===")
print(f"Best Threshold        : {best_thresh:.4f}")
print(f"Blend Weight (NN)     : {best_weight:.2f}")
print(f"Accuracy              : {acc:.4f}")
print(f"ROC AUC               : {roc:.4f}")
print(f"F1 Score              : {f1:.4f}")
print(f"PR AUC                : {pr_auc:.4f}")
print("\nClassification Report:\n", classification_report(y, y_pred))

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Failed", "Successful"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

importances = rf_temp.feature_importances_[top_indices]
feature_names = np.array(X.columns)[top_indices]
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
plt.title("Top 25 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend()
plt.grid(True)
plt.show()

if hasattr(meta_nn, 'estimator_') and hasattr(meta_nn.estimator_, 'loss_curve_'):
    plt.figure(figsize=(8, 6))
    plt.plot(meta_nn.estimator_.loss_curve_, label="MLP Loss", color="red")
    plt.title("MLPClassifier Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("Loss curve not available (meta_nn uses calibration wrapper).")
