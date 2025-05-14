import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# Ensure eager execution is enabled
tf.config.run_functions_eagerly(True)

# 1. Load Dataset
df = pd.read_csv('creditcard.csv')

# 2. Convert relevant columns to numeric and handle errors
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
df['Class'] = pd.to_numeric(df['Class'], errors='coerce')

# 3. Fill missing values with column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# 4. Feature Engineering: Add Hour of Day from 'Time'
df['Hour'] = (df['Time'] % (3600 * 24)) // 3600

# 5. Normalize 'Amount' and 'Time'
scaler = StandardScaler()
df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])

# 6. Features and Labels
X = df.drop('Class', axis=1)
y = df['Class'].astype(int)

# 7. Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42, k_neighbors=2) # Set k_neighbors to a lower value
X_resampled, y_resampled = smote.fit_resample(X, y)

# 8. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42)

# 9. Build Model Function
def build_model(activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(16, input_dim=X_train.shape[1], activation=activation))
    model.add(Dense(8, activation=activation))
    model.add(Dense(1, activation='sigmoid'))

    # Create a new optimizer instance inside the function
    if optimizer == 'adam':
        optimizer = Adam()
    elif optimizer == 'SGD':
        optimizer = SGD()
    elif optimizer == 'RMSprop':
        optimizer = RMSprop()

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# 10. Train with Different Activation Functions and Optimizers
activations = ['relu', 'tanh']
optimizers = ['SGD', 'Adam', 'RMSprop']  # Use optimizer names as strings
results = {}

for act in activations:
    for opt_name in optimizers:
        print(f"\n🔧 Training: Activation = {act}, Optimizer = {opt_name}")
        model = build_model(activation=act, optimizer=opt_name)  # Pass optimizer name
        history = model.fit(X_train, y_train, epochs=5, batch_size=2048,
                            validation_split=0.2, verbose=0)

        y_pred = (model.predict(X_test) > 0.5).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[f'{act}_{opt_name}'] = {'history': history, 'report': report}

        print(classification_report(y_test, y_pred))

# 11. Plot Accuracy & Loss Comparison
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for key in results:
    plt.plot(results[key]['history'].history['val_accuracy'], label=key)
plt.title('Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
for key in results:
    plt.plot(results[key]['history'].history['val_loss'], label=key)
plt.title('Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()