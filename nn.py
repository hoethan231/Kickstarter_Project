import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from df import X, Y

X = X.drop('spotlight', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

nn_model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

history = nn_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    batch_size=64,
    epochs=50,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

nn_preds_proba = nn_model.predict(X_test_scaled).flatten()
nn_preds = (nn_preds_proba > 0.5).astype(int)

acc = accuracy_score(y_test, nn_preds)
auc = roc_auc_score(y_test, nn_preds_proba)

print(f"Neural Network Accuracy: {acc:.4f}")
print(f"Neural Network AUC: {auc:.4f}")

print(" Classification Report:")
print(classification_report(y_test, nn_preds))

fpr, tpr, _ = roc_curve(y_test, nn_preds_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"NN Model (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Neural Network")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
