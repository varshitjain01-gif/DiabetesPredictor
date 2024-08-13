# app.py

from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('diabetes_module.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)

    result = "You are likely diabetic." if prediction[0] == 1 else "You are not likely diabetic."
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
