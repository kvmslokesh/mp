from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import load_model
import numpy as np


from tensorflow.python.keras.engine import data_adapter
import tensorflow as tf 
import tensorflow.python.keras as keras # Monkey patching `tensorflow.python.keras` 
keras.__version__ = tf.__version__
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset


# Load models
cnn_model = load_model('cnn_ddos_model.h5')
dl_model = load_model('dl_ddos_model.h5')

# Initialize Flask app
app = Flask(__name__)

# Load scaler used for scaling the input data
scaler = StandardScaler()


# Ensure that the scaler and cnn_model are already defined

def convert_int64_to_int(data):
    """Recursively converts np.int64 types to Python int"""
    if isinstance(data, dict):
        return {key: convert_int64_to_int(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_int64_to_int(item) for item in data]
    elif isinstance(data, np.int64):
        return int(data)  # Convert np.int64 to native int
    else:
        return data


# Home route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # We'll create this HTML file next

# Endpoint to predict whether an attack is happening
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert incoming data to DataFrame
        df = pd.DataFrame([data])  # Assuming incoming data is a dictionary
        
        # Preprocess the data (converting all columns to numeric and filling NaN values with 0)
        X = df.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)
        
        # Scale the input data
        X_scaled = scaler.fit_transform(X)
        
        # Check input dimensions and reshape if necessary for CNN
        if X_scaled.shape[1] < 3:
            # If the data is too small in one of the dimensions, pad it (example: padding in width)
            X_scaled = np.pad(X_scaled, ((0, 0), (0, 1)), mode='constant')  # Padding in the second dimension

        # Reshape data for CNN model (3D shape required: batch_size, features, channels)
        X_scaled_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        
        
        
        # Predict using both models
        cnn_prediction = cnn_model.predict(X_scaled_reshaped)
        dl_prediction = dl_model.predict(X_scaled_reshaped)
        # Convert the predictions into a readable format (binary: attack or no attack)
        cnn_attack = np.argmax(cnn_prediction, axis=1)[0]  # 0 for no attack, 1 for attack
        dl_attack = np.argmax(dl_prediction, axis=1)[0]    # 0 for no attack, 1 for attack
        
        # return str(cnn_attack)
        # You can merge both predictions for a more robust decision
        result = {
            'cnn_attack': str(cnn_attack),
            'dl_attack':str(dl_attack),
        }
        
        # Return response as JSON
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
