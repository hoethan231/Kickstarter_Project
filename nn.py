import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
<<<<<<< HEAD
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

=======
>>>>>>> d53aaa0 (random forest updated, trying stuff)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
<<<<<<< HEAD
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

from df import X, Y

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.3, stratify=Y)

model1 = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(64, activation='sigmoid'),
    # Dropout(0.5),
    Dense(32, activation='sigmoid'),
    # Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model1.summary()

model1.compile(optimizer=SGD(),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

history = model1.fit(
    x_train, y_train,
    validation_split=0.2,
    batch_size=128,
    epochs=15,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)
model1_preds_proba = model1.predict(x_test).flatten()
model1_preds = (model1_preds_proba > 0.5).astype(int)

acc = accuracy_score(y_test, model1_preds)
auc = roc_auc_score(y_test, model1_preds_proba)
=======
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from df import X, Y

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights_array))

optimizers = {
    'Adam': Adam(learning_rate=0.0005),
    'RMSprop': RMSprop(learning_rate=0.0005),
    'SGD': SGD(learning_rate=0.01, momentum=0.9),
}

results = []

for name, optimizer in optimizers.items():
    model = Sequential([
        Dense(256, input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.01),
        Dropout(0.2),
        Dense(128),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.01),
        Dropout(0.1),
        Dense(64),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.01),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
>>>>>>> d53aaa0 (random forest updated, trying stuff)

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=0)
    ]

<<<<<<< HEAD
print(" Classification Report:")
print(classification_report(y_test, model1_preds))

fpr, tpr, _ = roc_curve(y_test, model1_preds_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"NN Model 1 (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Neural Network")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

model2 = Sequential()
model2.add(Flatten(input_shape=(x_train.shape[1],)))
model2.add(Dense(50, activation='relu'))
model2.add(Dense(36, activation='relu'))
model2.add(Dense(1, activation='relu'))
model2.summary()

model2.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history2 = model2.fit(x_train, y_train, epochs=8, batch_size=64, validation_split=0.2)
model2_preds_proba = model2.predict(x_test).flatten()
model2_preds = (model2_preds_proba > 0.5).astype(int)

acc = accuracy_score(y_test, model2_preds)
auc = roc_auc_score(y_test, model2_preds_proba)

print(f"Neural Network Accuracy: {acc:.4f}")
print(f"Neural Network AUC: {auc:.4f}")

print(" Classification Report:")
print(classification_report(y_test, model2_preds))

fpr, tpr, _ = roc_curve(y_test, model2_preds_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"NN Model 2 (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Neural Network")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def build_model(n_layers=1, activation='relu', optimizer='adam', learning_rate=0.001, neurons=64):
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    model = Sequential()
    model.add(Dense(neurons, input_dim=x_train.shape[1], activation=activation))
    for _ in range(n_layers - 1):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
model = KerasClassifier(model=build_model)

param_grid = {
    "model__n_layers": [1, 2],
    "model__activation": ['relu', 'tanh'],
    "model__optimizer": ['adam', 'sgd'],
    "model__learning_rate": [0.001, 0.01],
    "model__neurons": [32, 64],
    "batch_size": [32, 64],
    "epochs": [10, 50]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(x_train, y_train)

print(f"Best: {grid_result.best_score_:.4f} using {grid_result.best_params_}")

results = pd.DataFrame(grid_result.cv_results_)

plt.figure()
for act in results['param_activation'].unique():
    subset = results[results['param_activation'] == act]
    plt.plot(subset['param_n_layers'], subset['mean_test_score'], label=f'activation={act}', marker='o')

plt.xlabel('Number of Layers')
plt.ylabel('Mean CV Accuracy')
plt.title('Accuracy vs Number of Layers for Each Activation')
plt.legend()
plt.grid(True)
plt.show()

pivot = results.pivot_table(values='mean_test_score',
                            index='param_activation',
                            columns='param_optimizer')

plt.figure()
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
plt.title('Accuracy Heatmap: Activation vs Optimizer')
plt.xlabel('Optimizer')
plt.ylabel('Activation')
plt.show()

plt.figure()
for opt in results['param_optimizer'].unique():
    subset = results[results['param_optimizer'] == opt]
    plt.plot(subset['param_learning_rate'], subset['mean_test_score'], label=opt, marker='x')

plt.xscale('log')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Mean CV Accuracy')
plt.title('Accuracy vs Learning Rate by Optimizer')
plt.legend()
plt.grid(True)
plt.show()

best_neuron_scores = results.groupby('param_neurons')['mean_test_score'].max()

plt.figure()
best_neuron_scores.plot(kind='bar')
plt.title('Best Accuracy by Neuron Count')
plt.xlabel('Neurons per Layer')
plt.ylabel('Max Mean CV Accuracy')
plt.grid(True)
plt.show()
=======
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        batch_size=64,
        epochs=100,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )

    preds_proba = model.predict(X_test_scaled).flatten()
    precisions, recalls, thresholds = precision_recall_curve(y_test, preds_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    preds = (preds_proba > best_threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_proba)
    report = classification_report(y_test, preds, output_dict=True)

    results.append({
        'Optimizer': name,
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'F1': round(report['1']['f1-score'], 4),
        'Precision': round(report['1']['precision'], 4),
        'Recall': round(report['1']['recall'], 4),
        'Threshold': round(best_threshold, 4)
    })

df_results = pd.DataFrame(results)
print(df_results)
>>>>>>> d53aaa0 (random forest updated, trying stuff)
