import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, log_loss
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/prathamsaxena/Downloads/SJSU/CMPE 188/ML Code/ML Data/kickstarter_data_full.csv", usecols=[
    'goal', 'state', 'country', 'static_usd_rate', 'category', 'created_at_yr',
    'create_to_launch_days', 'launch_to_deadline_days', 'blurb_len_clean', 'SuccessfulBool'])

df = df.dropna()
df = df[df['state'] != 'live'].drop(columns='state')
df['goal'] = df['goal'].astype(float) * df['static_usd_rate'].astype(float)
df = df.drop(columns='static_usd_rate')

for col in ['category', 'country']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

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
df['interaction3'] = df['goal_per_day'] * df['blurb_density']
df['interaction4'] = df['create_to_launch_log'] * df['launch_to_deadline_log']
df['interaction5'] = df['goal_log_bucket'] * df['country']
df['goal_duration_ratio'] = df['goal_log'] / (df['launch_to_deadline_log'] + 1)
df['blurb_launch_ratio'] = df['blurb_density'] / (df['create_to_launch_log'] + 1)
df = df.drop(columns='goal')

X = df.drop(columns='SuccessfulBool')
y = df['SuccessfulBool'].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_scaled, y)

ensemble = GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=5, random_state=42)
ensemble.fit(X_bal, y_bal)
ensemble_probs = ensemble.predict_proba(X_bal)[:, 1]

mask_conf = (ensemble_probs < 0.3) | (ensemble_probs > 0.7)
mask_amb = (ensemble_probs >= 0.3) & (ensemble_probs <= 0.7)
mask_dis = ((ensemble_probs > 0.5) & (y_bal == 0)) | ((ensemble_probs < 0.5) & (y_bal == 1))

X_conf, y_conf = X_bal[mask_conf], y_bal[mask_conf]
X_amb, y_amb = X_bal[mask_amb], y_bal[mask_amb]
X_dis, y_dis = X_bal[mask_dis], y_bal[mask_dis]

sfm = SelectFromModel(RandomForestClassifier(n_estimators=300, random_state=42), threshold="median")
X_conf_sel = sfm.fit_transform(X_conf, y_conf)
X_amb_sel = sfm.transform(X_amb)
X_dis_sel = sfm.transform(X_dis)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_conf_sel, y_conf, stratify=y_conf, test_size=0.3, random_state=42)
Xa_train, Xa_test, ya_train, ya_test = train_test_split(X_amb_sel, y_amb, stratify=y_amb, test_size=0.3, random_state=42)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(X_dis_sel, y_dis, stratify=y_dis, test_size=0.3, random_state=42)

tree1 = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
tree2 = DecisionTreeClassifier(max_depth=14, class_weight='balanced', random_state=42)
tree3 = DecisionTreeClassifier(max_depth=18, class_weight='balanced', random_state=42)
tree1.fit(Xc_train, yc_train)
tree2.fit(Xa_train, ya_train)
tree3.fit(Xd_train, yd_train)

yc_prob = tree1.predict_proba(Xc_test)[:, 1]
ya_prob = tree2.predict_proba(Xa_test)[:, 1]
yd_prob = tree3.predict_proba(Xd_test)[:, 1]
yc_true = yc_test
ya_true = ya_test
yd_true = yd_test

y_true = np.concatenate([yc_true, ya_true, yd_true])
tree1_probs = np.concatenate([yc_prob, np.zeros_like(ya_prob), np.zeros_like(yd_prob)])
tree2_probs = np.concatenate([np.zeros_like(yc_prob), ya_prob, np.zeros_like(yd_prob)])
tree3_probs = np.concatenate([np.zeros_like(yc_prob), np.zeros_like(ya_prob), yd_prob])

ensemble_sub = np.concatenate([
    ensemble_probs[mask_conf][len(yc_train):],
    ensemble_probs[mask_amb][len(ya_train):],
    ensemble_probs[mask_dis][len(yd_train):]
])

X_meta = np.stack([
    tree1_probs,
    tree2_probs,
    tree3_probs,
    np.abs(tree1_probs - tree2_probs),
    np.abs(tree1_probs - tree3_probs),
    np.abs(tree2_probs - tree3_probs),
    np.max(np.stack([tree1_probs, tree2_probs, tree3_probs]), axis=0),
    np.min(np.stack([tree1_probs, tree2_probs, tree3_probs]), axis=0),
    ensemble_sub
], axis=1)

base_meta = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42, class_weight='balanced')
meta_clf = CalibratedClassifierCV(base_meta, method='isotonic', cv=3)
meta_clf.fit(X_meta, y_true)

y_meta_prob = meta_clf.predict_proba(X_meta)[:, 1]
y_meta_pred = (y_meta_prob > 0.5).astype(int)

precision_meta, recall_meta, _ = precision_recall_curve(y_true, y_meta_prob)
acc_meta = accuracy_score(y_true, y_meta_pred)
roc_meta = roc_auc_score(y_true, y_meta_prob)
pr_meta = auc(recall_meta, precision_meta)
f1_meta = f1_score(y_true, y_meta_pred)

print(f"Accuracy   : {acc_meta:.4f}")
print(f"ROC AUC    : {roc_meta:.4f}")
print(f"PR AUC     : {pr_meta:.4f}")
print(f"F1 Score   : {f1_meta:.4f}")

cm = confusion_matrix(y_true, y_meta_pred)
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail', 'Success']).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()

meta_model = meta_clf.calibrated_classifiers_[0].estimator
meta_features = [
    "tree1_prob", "tree2_prob", "tree3_prob",
    "abs_diff_1_2", "abs_diff_1_3", "abs_diff_2_3",
    "max_prob", "min_prob", "ensemble_prob"
]
if hasattr(meta_model, 'feature_importances_'):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=meta_model.feature_importances_, y=meta_features)
    plt.title("Meta Tree Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

thresholds = np.linspace(0.2, 0.8, 100)
losses = [log_loss(y_true, (y_meta_prob > t).astype(int)) for t in thresholds]
accuracies = [(y_true == (y_meta_prob > t).astype(int)).mean() for t in thresholds]

plt.figure(figsize=(8, 5))
plt.plot(thresholds, losses, label='Log Loss')
plt.plot(thresholds, accuracies, label='Accuracy')
plt.xlabel('Threshold')
plt.title('Loss vs Accuracy (Threshold Tuning)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(meta_model, filled=True, feature_names=meta_features, class_names=['Fail', 'Success'], rounded=True, fontsize=10)
plt.title("Decision Tree Structure")
plt.tight_layout()
plt.show()
