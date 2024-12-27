import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

# Load the data
data = pd.read_csv('dataset_sdn.csv')  # Change to your file format (e.g., Excel)

# Handle IP addresses - Convert IPs into numerical format
# You can use any method for converting IP to numeric. One approach is using a hashing method:
data['src'] = data['src'].apply(lambda x: int.from_bytes(map(ord, x), 'big'))
data['dst'] = data['dst'].apply(lambda x: int.from_bytes(map(ord, x), 'big'))

# Handle labels - Convert the label column into one-hot encoded format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['label'])
y_categorical = pd.get_dummies(y).values  # One-hot encoded labels

# Drop non-numeric columns (if needed)
X = data.drop(columns=['label'])

# Convert IP addresses and other non-numeric columns to numeric or drop them
# If 'src' and 'dst' are IP addresses, you might want to encode them or drop them
X = X.apply(pd.to_numeric, errors='coerce')  # This will convert any non-numeric columns to NaN

# Alternatively, drop columns that are non-numeric (like 'src' and 'dst')
# X = X.drop(columns=['src', 'dst'])  # Uncomment if you don't need these columns

# Check if there are NaN values (from non-numeric data)
X = X.fillna(0)  # Fill NaN values with 0 or any other appropriate value


# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Reshape for Conv1D model (batch_size, steps, features)
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the CNN model
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

# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test))

# Save the model
cnn_model.save('cnn_ddos_model.h5')
print("CNN model saved as 'cnn_ddos_model.h5'.")
