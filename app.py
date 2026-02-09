from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route("/")
def home():
    return "Fraud Detection API is Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["features"]

    data = np.array(data).reshape(1, -1)

    # Scale Time and Amount (last 2 columns)
    data[:, [-2, -1]] = scaler.transform(data[:, [-2, -1]])

    # Get probability
    prob = model.predict_proba(data)[0][1]

    # Apply threshold
    prediction = int(prob >= 0.9)

    return jsonify({
        "fraud_probability": round(float(prob), 4),
        "fraud_prediction": prediction
    })


if __name__ == "__main__":
    app.run(debug=True)
