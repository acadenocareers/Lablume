from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = joblib.load("lablume_final_model.pkl")
scaler = joblib.load("lablume_scaler.pkl")

@app.route("/", methods=["GET"])
def home():
    return {"message": "LabLume ML API is running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = [
        data["Age"],
        data["Sugar"],
        data["BP"],
        data["Cholesterol"],
        data["Hemoglobin"],
        data["Gender"]
    ]

    scaled = scaler.transform([features])
    prediction = model.predict(scaled)[0]

    labels = {
        0: "Normal",
        1: "Attention Required",
        2: "Critical"
    }

    return jsonify({
        "prediction": int(prediction),
        "result": labels[prediction]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
