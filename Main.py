import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("arrhythmia.data", header=None)

# Replace '?' with NaN
data.replace('?', np.nan, inplace=True)

# Assuming the last column contains the class labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Convert to numeric values
X = X.astype(float)

# Impute missing values with column means
X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Create a base deep learning model
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# With feature selection
num_features_to_select = 120  # Choose the number of features to select
feature_selector = SelectKBest(f_classif, k=num_features_to_select)
X_train_selected = feature_selector.fit_transform(X_train, y_train)
X_test_selected = feature_selector.transform(X_test)

model_with_feature_selection = create_model((X_train_selected.shape[1],), len(np.unique(y_encoded)))

# Train the single model with feature selection
history_with_feature_selection = model_with_feature_selection.fit(X_train_selected, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)

# Train and record training history for each ensemble model
num_models = 5
ensemble_train_history = []
ensemble_val_history = []

for i in range(num_models):
    ensemble_model = create_model((X_train_selected.shape[1],), len(np.unique(y_encoded)))
    history = ensemble_model.fit(X_train_selected, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)
    ensemble_train_history.append(history.history['accuracy'])
    ensemble_val_history.append(history.history['val_accuracy'])

# Single model training accuracy
single_model_train_accuracy = history_with_feature_selection.history['accuracy']
single_model_val_accuracy = history_with_feature_selection.history['val_accuracy']



# Plot the ensemble training history vs single model training accuracy
plt.figure(figsize=(10, 5))
for i in range(num_models):
    plt.plot(ensemble_train_history[i], label=f'Ensemble Model {i+1} Train Accuracy', linestyle='dashed')
    plt.plot(ensemble_val_history[i], label=f'Ensemble Model {i+1} Validation Accuracy', linestyle='dashed')

plt.plot(single_model_train_accuracy, label='Single Model Train Accuracy', marker='o')
plt.plot(single_model_val_accuracy, label='Single Model Validation Accuracy', marker='o')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Ensemble Training History vs Single Model Training Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
