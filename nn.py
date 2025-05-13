import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

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
earlystop = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)

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

    callbacks = [
        earlystop,
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=0)
    ]

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
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))
    for _ in range(n_layers - 1):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
model = KerasClassifier(
    model=build_model,
    callbacks=[earlystop]
    )

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
grid_result = grid.fit(X_train, y_train)

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

best = grid_result.best_params_

def build_deep_model(n_layers=4, dropout_rate=0.2,
                     activation=best['activation'],
                     optimizer=best['optimizer'],
                     learning_rate=best['learning_rate'],
                     neurons=best['neurons']):
    
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")
    
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))
    
    for _ in range(n_layers - 1):
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

model2 = KerasClassifier(
    model=build_deep_model,
    batch_size=best['batch_size'],
    epochs=best['epochs'],
    callbacks=[earlystop]
)

param_grid2 = {
    "model__n_layers": [4, 5, 6, 7],
    "model__dropout_rate": [0.0, 0.2, 0.4, 0.5]
}

grid2 = GridSearchCV(estimator=model2, param_grid=param_grid2, cv=3)
grid2_result = grid2.fit(X_train, y_train)

print("Best params for deep model:", grid2_result.best_params_)

results2 = pd.DataFrame(grid2_result.cv_results_)

pivot = results2.pivot_table(values='mean_test_score',
                             index='param_model__n_layers',
                             columns='param_model__dropout_rate')

plt.figure()
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="magma")
plt.title("Accuracy Heatmap: Dropout Rate vs Number of Layers")
plt.xlabel("Dropout Rate")
plt.ylabel("Number of Layers")
plt.show()