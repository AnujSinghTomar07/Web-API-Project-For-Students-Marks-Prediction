import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('Model/Student_Marks.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form data
        study_hours = float(request.form['study_hours'])
        sleep_hours = float(request.form['sleep_hours'])
        extracurricular_hours = float(request.form['extracurricular_hours'])

        # Prepare the input for the model
        features = np.array([[study_hours, sleep_hours, extracurricular_hours]])

        # Make prediction
        prediction = model.predict(features)

        # Round the prediction result
        output = round(prediction[0], 2)

        # Render the result on the index.html page
        return render_template('index.html', prediction_text=f'Predicted Marks: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run()
