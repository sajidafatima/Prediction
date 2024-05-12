import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("lstm_model_final_2.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    # Extract features from form input
    float_features = [float(x) for x in request.form.values()]
    # Reshape features for LSTM input
    features = np.array(float_features).reshape((1, 1, len(float_features)))
    # Make prediction
    prediction = model.predict(features)
    # Format prediction text
    prediction_text = "The Demand of Electricity is {} (kWh)".format(prediction)
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    flask_app.run(debug=True)
