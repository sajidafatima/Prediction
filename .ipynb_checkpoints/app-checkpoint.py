import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle 

# Create flask app
flask_app = Flask(__name__)
#model = pickle.load(open("lstm_model_final_2.pkl", "rb"))

from tensorflow.keras.models import load_model

# Load the model
model = load_model("model_final.h5")


@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    # Extract features from form input
    float_features = [float(x) for x in request.form.values()]
    # Convert features to numpy array
    features_array = np.array(features)
    # Reshape features for LSTM input
    reshaped_features = np.reshape(features_array, (features_array.shape[0], features_array.shape[1], 1))
    # Make prediction
    prediction = model.predict(features)
    # Format prediction text
    prediction_text = "The Demand of Electricity is {} (kWh)".format(prediction)
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    flask_app.run(debug=True)
