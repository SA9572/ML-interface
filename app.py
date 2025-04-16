# app.py
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained ML model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        tensile_strength = float(request.form['tensile_strength'])
        proof_stress = float(request.form['proof_stress'])
        elongation = float(request.form['elongation'])

        # Prepare input for the model
        input_data = np.array([[tensile_strength, proof_stress, elongation]])

        # Make prediction (assuming model outputs 1 = Good, 0 = Not Suitable)
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data).max() * 100  # Confidence score

        # Generate result and suggestion
        if prediction == 1:
            result = "Good for Industry Use"
            suggestion = "No improvements needed."
        else:
            result = "Not Suitable for Industry Use"
            suggestion = "Consider increasing tensile strength or adjusting heat treatment."

        return jsonify({
            'result': result,
            'confidence': f"{confidence:.2f}%",
            'suggestion': suggestion,
            'tensile_strength': tensile_strength,
            'proof_stress': proof_stress,
            'elongation': elongation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)