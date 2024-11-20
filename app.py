from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained machine learning model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the dataset
data = pd.read_csv('data.csv')  # Replace 'data.csv' with the name of your dataset file

# Define the home route
@app.route('/')
def home():
    # Extract unique machine IDs from the dataset
    machine_ids = data['UDI'].unique()
    return render_template('index.html', machine_ids=machine_ids)

# Define the route to check machine status
@app.route('/check_status', methods=['POST'])
def check_status():
    try:
        # Get the selected Machine ID from the frontend
        machine_id = request.form.get('machine_id')

        # Filter the dataset to get the row corresponding to the Machine ID
        machine_data = data[data['UDI'] == machine_id]
        
        if machine_data.empty:
            return jsonify({"error": "Machine ID not found."}), 400
        
        # Drop unnecessary columns (like UDI) before prediction
        machine_features = machine_data.drop(columns=['UDI', 'Product ID', 'Failure Type', 'Target'])

        # Make predictions using the trained model
        prediction = model.predict(machine_features)[0]
        prediction_proba = model.predict_proba(machine_features)[0]

        # Interpret the prediction (example: 0 = No Failure, 1 = Failure)
        status = "No Failure" if prediction == 0 else "Failure"
        confidence = round(prediction_proba[prediction] * 100, 2)

        # Return the result
        return jsonify({"status": status, "confidence": f"{confidence}%"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
