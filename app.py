from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("lablume_final_model.pkl")
scaler = joblib.load("lablume_scaler.pkl")

HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>LabLume Prediction</title>
    <style>
        body { font-family: Arial; background:#f4f6f8; }
        .box {
            width: 400px; margin: 50px auto; padding: 20px;
            background: white; border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        input, button {
            width: 100%; padding: 10px; margin: 8px 0;
        }
        button { background:#2563eb; color:white; border:none; }
    </style>
</head>
<body>
<div class="box">
    <h2>LabLume Health Risk Prediction</h2>
    <form method="post">
        <input name="Age" placeholder="Age" required>
        <input name="Sugar" placeholder="Sugar" required>
        <input name="BP" placeholder="BP" required>
        <input name="Cholesterol" placeholder="Cholesterol" required>
        <input name="Hemoglobin" placeholder="Hemoglobin" required>
        <input name="Gender" placeholder="Gender (0=F, 1=M)" required>
        <button type="submit">Predict</button>
    </form>
    {% if result %}
        <h3>Result: {{ result }}</h3>
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return "LabLume ML API is running"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None

    if request.method == "POST":
        data = [
            float(request.form["Age"]),
            float(request.form["Sugar"]),
            float(request.form["BP"]),
            float(request.form["Cholesterol"]),
            float(request.form["Hemoglobin"]),
            float(request.form["Gender"])
        ]

        scaled = scaler.transform([data])
        pred = model.predict(scaled)[0]

        labels = {0:"Normal", 1:"Attention Required", 2:"Critical"}
        result = labels[pred]

    return render_template_string(HTML_FORM, result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
