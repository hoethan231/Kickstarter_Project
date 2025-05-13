import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, classification_report
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from df import X, Y, x, y

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.3, stratify=Y)

x2_train, x2_test, y2_train, y2_test = train_test_split(x, y, random_state=42, test_size=0.3, stratify=y)

model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(64, activation='sigmoid'),
    Dense(32, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])
model.summary()

model.compile(optimizer=SGD(),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

earlystop = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    batch_size=128,
    epochs=10,
    callbacks=[earlystop],
    verbose=1
)

model_preds_proba = model.predict(x_test).flatten()
model_preds = (model_preds_proba > 0.5).astype(int)

acc = accuracy_score(y_test, model_preds)
auc = roc_auc_score(y_test, model_preds_proba)

print(f"Neural Network Accuracy: {acc:.4f}")
print(f"Neural Network AUC: {auc:.4f}")

print(" Classification Report:")
print(classification_report(y_test, model_preds))

fpr, tpr, _ = roc_curve(y_test, model_preds_proba)
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

model = Sequential()
model.add(Flatten(input_shape=(x_train.shape[1],)))
model.add(Dense(50, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()
model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history2 = model.fit(x_train, y_train, epochs=8, batch_size=64, validation_split=0.2)

model_preds_proba = model.predict(x_test).flatten()
model_preds = (model_preds_proba > 0.5).astype(int)

acc = accuracy_score(y_test, model_preds)
auc = roc_auc_score(y_test, model_preds_proba)

print(f"Neural Network Accuracy: {acc:.4f}")
print(f"Neural Network AUC: {auc:.4f}")

print(" Classification Report:")
print(classification_report(y_test, model_preds))

fpr, tpr, _ = roc_curve(y_test, model_preds_proba)
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights_array))

optimizers = {
    'Adam': Adam(learning_rate=0.0005),
    'RMSprop': RMSprop(learning_rate=0.0005),
    'SGD': SGD(learning_rate=0.01, momentum=0.9),
}
earlystop = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
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
    model.add(Dense(neurons, input_dim=x_train.shape[1], activation=activation))
    for _ in range(n_layers - 1):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def build_model2(n_layers=1, activation='relu', optimizer='adam', learning_rate=0.001, neurons=64, dropout_rate=0.2):
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    model = Sequential()
    model.add(Dense(neurons, input_dim=x2_train.shape[1], activation=activation))

    if n_layers > 3:
      for _ in range(n_layers - 1):
          model.add(Dense(neurons, activation=activation))
    else:
      for _ in range(n_layers - 1):
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

model = KerasClassifier(
    model=build_model,
    callbacks=[earlystop]
    )
model2 = KerasClassifier(
    model=build_model2,
    callbacks=[earlystop]
    )

param_grid = {
    "model__n_layers": [1, 2],
    "model__activation": ['relu', 'sigmoid'],
    "model__optimizer": ['adam', 'sgd'],
    "model__learning_rate": [0.001, 0.01],
    "model__neurons": [32, 64],
    "batch_size": [64],
    "epochs": [10]
}
# param_grid = {
#     "model__n_layers": [1, 2],
#     "model__activation": ['relu', 'sigmoid'],
#     "model__optimizer": ['adam', 'sgd'],
#     "model__learning_rate": [0.001, 0.01],
#     "model__neurons": [32, 64],
#     "batch_size": [32, 64],
#     "epochs": [10, 50]
# }


grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
grid_result = grid.fit(x_train, y_train)

print(f"Best: {grid_result.best_score_:.4f} using {grid_result.best_params_}")


grid2 = GridSearchCV(estimator=model2, param_grid=param_grid, cv=3, n_jobs=-1)
grid_result2 = grid2.fit(x2_train, y2_train)

print(f"Best2: {grid_result2.best_score_:.4f} using {grid_result2.best_params_}")


results = pd.DataFrame(grid_result.cv_results_)
results.columns

plt.figure()
for act in results['param_model__activation'].unique():
    subset = results[results['param_model__activation'] == act]
    plt.plot(subset['param_model__n_layers'], subset['mean_test_score'], label=f'activation={act}', marker='o')

plt.xlabel('Number of Layers')
plt.ylabel('Mean CV Accuracy')
plt.title('Accuracy vs Number of Layers for Each Activation')
plt.legend()
plt.grid(True)
plt.show()


pivot = results.pivot_table(values='mean_test_score',
                            index='param_model__activation',
                            columns='param_model__optimizer')

plt.figure()
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
plt.title('Accuracy Heatmap: Activation vs Optimizer')
plt.xlabel('Optimizer')
plt.ylabel('Activation')
plt.show()


plt.figure()
for opt in results['param_model__optimizer'].unique():
    subset = results[results['param_model__optimizer'] == opt]
    plt.plot(subset['param_model__learning_rate'], subset['mean_test_score'], label=opt, marker='x')

plt.xscale('log')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Mean CV Accuracy')
plt.title('Accuracy vs Learning Rate by Optimizer')
plt.legend()
plt.grid(True)
plt.show()


best_neuron_scores = results.groupby('param_model__neurons')['mean_test_score'].max()

plt.figure()
best_neuron_scores.plot(kind='bar')
plt.title('Best Accuracy by Neuron Count')
plt.xlabel('Neurons per Layer')
plt.ylabel('Max Mean CV Accuracy')
plt.grid(True)
plt.show()

best = grid_result.best_params_
print(best)

# Build a deep model off best parameters from previous NN


model2 = KerasClassifier(
    model=build_model2,
    batch_size=best['batch_size'],
    epochs=best['epochs'],
    callbacks=[earlystop]
)

param_grid2 = {
    "model__n_layers": [4, 5, 6, 7],
    "model__dropout_rate": [0.0, 0.2, 0.4, 0.5]
}

grid2 = GridSearchCV(estimator=model2, param_grid=param_grid2, cv=3, n_jobs=-1)
grid2_result = grid2.fit(x_train, y_train)

best_model = grid2_result.best_estimator_

# Get predicted probabilities and class labels
y_pred_proba = best_model.predict_proba(x_test)[:, 1]
y_pred = best_model.predict(x_test)

# Compute metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"ROC AUC: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


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

results2.columns


cols = [
    'param_model__activation',
    'param_model__optimizer',
    'param_model__learning_rate',
    'param_model__neurons',
    'param_model__n_layers',
    'mean_test_score'
]

results2 = results2[cols]

# Rename for clarity
results2.columns = ['Activation', 'Optimizer', 'Learning Rate', 'Neurons', 'Layers', 'Mean CV Accuracy']

# Create a pivot table for heatmap (e.g. optimizer vs activation)
pivot = results2.pivot_table(
    values='Mean CV Accuracy',
    index='Activation',
    columns='Optimizer',
    aggfunc='mean'
)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Grid Search Cross-Validation Results: Optimizer vs Activation")
plt.xlabel("Optimizer")
plt.ylabel("Activation")
plt.tight_layout()
plt.show()


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model3 = KerasClassifier(
    model=build_model2,
    callbacks=[early_stop],
    validation_split=0.2,
    verbose=0
)

random_param_dist = {
    "model__n_layers": [1, 2, 3, 4, 5, 6],
    "model__activation": ['relu', 'tanh', 'sigmoid'],
    "model__optimizer": ['adam', 'sgd', 'rmsprop'],
    "model__learning_rate": [0.0001, 0.001, 0.01],
    "model__neurons": [16, 32, 64, 128],
    "batch_size": [32, 64],
    "epochs": [10, 20, 30]
}

random_search = RandomizedSearchCV(
    estimator=model3,
    param_distributions=random_param_dist,
    n_iter=10,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_result = random_search.fit(x_train, y_train)


print("Best Params:", random_result.best_params_)
print("Best Score:", random_result.best_score_)

bayes_param_dist = {
    "model__n_layers": Integer(1, 8),
    "model__activation": Categorical(['relu', 'tanh', 'sigmoid']),
    "model__optimizer": Categorical(['adam', 'sgd', 'rmsprop']),
    "model__learning_rate": Real(1e-4, 1e-2, prior='log-uniform'),
    "model__neurons": Integer(32, 128),
    "batch_size": Categorical([32, 64]),
    "epochs": Integer(10, 30)
}

opt = BayesSearchCV(
    estimator=model3,
    search_spaces=bayes_param_dist,
    n_iter=20,
    cv=3,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

opt_result = opt.fit(x_train, y_train)

print("Best parameters:", opt_result.best_params_)
print("Best score:", opt_result.best_score_)

opt_df = pd.DataFrame(opt_result.cv_results_)
opt_df.columns

opt_df = opt_df[['param_batch_size', 'param_epochs', 'param_model__activation','param_model__learning_rate', 'param_model__n_layers','param_model__neurons', 'param_model__optimizer','mean_test_score']]
opt_df = opt_df.sort_values(by='mean_test_score', ascending=False)
print(opt_df)