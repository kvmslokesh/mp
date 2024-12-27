import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import load_model

# Initialize Flask app and load models
app = Flask(__name__)
cnn_model = load_model('.//cnn_ddos_model.h5')
scaler = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Log the received data for debugging
        print("Received Data:", data)
        
        # Convert incoming data to DataFrame
        df = pd.DataFrame([data])  # Convert the JSON data into a DataFrame
        
        # Preprocess the data (convert all columns to numeric, fill NaNs with 0)
        X = df.apply(pd.to_numeric, errors='coerce')  # Coerce non-numeric values to NaN
        X = X.fillna(0)  # Replace NaNs with zeros
        
        # Log the preprocessed data
        print("Preprocessed Data:", X)
        
        # Scale the input data (feature scaling)
        X_scaled = scaler.fit_transform(X)
        
        # Reshape data for CNN model (ensure it has 3 dimensions)
        X_scaled_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        
        # Predict using the CNN model
        cnn_prediction = cnn_model.predict(X_scaled_reshaped)
        
        # Convert predictions to binary (0 for no attack, 1 for attack)
        cnn_attack = np.argmax(cnn_prediction, axis=1)[0]
        
        # Log the prediction result
        print("CNN Attack Prediction:", cnn_attack)
        
        # Return response as JSON
        return jsonify({
            'cnn_attack': str(cnn_attack)
        })
    
    except Exception as e:
        # Return an error message if something goes wrong
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
