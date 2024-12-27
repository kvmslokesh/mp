import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.python.keras.models import load_model

from tensorflow.python.keras.engine import data_adapter
import tensorflow as tf 
import tensorflow.python.keras as keras # Monkey patching `tensorflow.python.keras` 
keras.__version__ = tf.__version__
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

# Load data from Excel file
data = pd.read_excel('network_data.xlsx') # change the name 

# Separate features and labels
X = data.drop(columns=['label'])  # Adjust if 'label' column name differs
y = data['label']

# Convert IP addresses and other non-numeric columns to numeric or drop them
# If 'src' and 'dst' are IP addresses, you might want to encode them or drop them
X = X.apply(pd.to_numeric, errors='coerce')  # This will convert any non-numeric columns to NaN

# Alternatively, drop columns that are non-numeric (like 'src' and 'dst')
# X = X.drop(columns=['src', 'dst'])  # Uncomment if you don't need these columns

# Check if there are NaN values (from non-numeric data)
X = X.fillna(0)  # Fill NaN values with 0 or any other appropriate value

# One-hot encode the labels using Pandas get_dummies()
y_categorical = pd.get_dummies(y).values  # Converts categorical labels into one-hot encoded format

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Reshape data for CNN model input (needed for Conv1D)
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build CNN model
cnn_model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    Conv1D(32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')  # Assuming categorical output
])

# Compile CNN model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train CNN model
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test))

# Save CNN model
cnn_model.save("save_model/")
cnn_model.save('cnn_ddos_model.h5')
print("CNN model saved as 'cnn_ddos_model.h5'.")

# Build fully connected deep learning model
dl_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')  # Adjust based on output class count
])

# Compile deep learning model
dl_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train deep learning model
dl_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save deep learning model
dl_model.save('dl_ddos_model.h5')
print("Deep Learning model saved as 'dl_ddos_model.h5'.")

# Example: Load and use saved models
def load_models():
    cnn_loaded_model = load_model('cnn_ddos_model.h5')
    dl_loaded_model = load_model('dl_ddos_model.h5')
    print("Models loaded successfully.")
    return cnn_loaded_model, dl_loaded_model

# Uncomment below to load models when needed
# cnn_model, dl_model = load_models()