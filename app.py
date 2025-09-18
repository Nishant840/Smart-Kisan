import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import pickle
import numpy as np
import pandas as pd



load_dotenv()
app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "❗️ Environment variable GROQ_API_KEY is not set. "
        "Export it before running the app, e.g.: "
        "`export GROQ_API_KEY=your_key_here`"
    )

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


with open("crop_model.pkl", "rb") as f:
    crop_model = pickle.load(f)
with open("scaler_crop.pkl", "rb") as f:
    crop_scaler = pickle.load(f)
with open("label_encoder_crop.pkl", "rb") as f:
    crop_le = pickle.load(f)

with open("Fertilizer_Recommendation_model.pkl", "rb") as f:
    fert_model = pickle.load(f)
with open("Fertilizer_label_encoder.pkl", "rb") as f:
    fert_le = pickle.load(f)



def crop_recommendation(N, P, K, temp, hum, ph, rain):
    features = np.array([[N, P, K, temp, hum, ph, rain]])
    features_scaled = crop_scaler.transform(features)
    pred_encoded = crop_model.predict(features_scaled)[0]
    return crop_le.inverse_transform([pred_encoded])[0]


@app.route("/")
def home():
    return render_template("home.html")

@app.route('/crop')
def crop_Recommend():
    return render_template('crop.html')

@app.route('/soil')
def soil_page():
    return render_template("soil.html")

@app.route("/predictFertilizer", methods=["POST"])
def predict_fertilizer():
    data = request.get_json()

    try:
        custom_input = {
            "Temperature": [float(data["Temperature"])],
            "Humidity": [float(data["Humidity"])],
            "Moisture": [float(data["Moisture"])],
            "Nitrogen": [float(data["Nitrogen"])],
            "Phosphorous": [float(data["Phosphorous"])],
            "Potassium": [float(data["Potassium"])],
            "Soil_Type": [data["Soil_Type"]],
            "Crop_Type": [data["Crop_Type"]],
        }

        custom_df = pd.DataFrame(custom_input)

        custom_encoded = pd.get_dummies(custom_df, columns=["Soil_Type", "Crop_Type"], drop_first=True)

        for col in fert_model.feature_names_in_:
            if col not in custom_encoded.columns:
                custom_encoded[col] = 0
        custom_encoded = custom_encoded[fert_model.feature_names_in_]

        pred = fert_model.predict(custom_encoded)
        fert_name = fert_le.inverse_transform(pred)[0]

        return jsonify({"fertilizer": fert_name})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400


@app.route("/predictCrop", methods=["POST"])
def predict_crop():
    data = request.get_json()
    try:
        N = float(data["N"])
        P = float(data["P"])
        K = float(data["K"])
        temp = float(data["temp"])
        hum = float(data["hum"])
        ph = float(data["ph"])
        rain = float(data["rain"])
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid input data"}), 400

    crop_name = crop_recommendation(N, P, K, temp, hum, ph, rain)
    return jsonify({"crop": crop_name})



@app.route("/askAnything", methods=["GET", "POST"])
def ask_anything():
    if request.method == "GET":
        return render_template("askAnything.html")

    payload = request.get_json()
    question = payload.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    groq_payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "in the same language it was asked, using a concise but complete response."
                ),
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            GROQ_ENDPOINT, headers=headers, json=groq_payload, timeout=12
        )
        resp.raise_for_status()
    except requests.HTTPError as http_err:
        try:
            err_body = resp.json()
            err_msg = err_body.get("error", {}).get("message", str(err_body))
        except Exception:
            err_msg = resp.text or str(http_err)

        if resp.status_code == 401:
            return (
                jsonify(
                    {
                        "error": (
                            "Groq authentication failed (401). "
                            "Check that GROQ_API_KEY is correct, not expired, "
                            "and has the required permissions."
                        )
                    }
                ),
                401,
            )
        return jsonify({"error": f"Groq request failed: {err_msg}"}), resp.status_code
    except requests.RequestException as e:
        return jsonify({"error": f"Network error contacting Groq: {e}"}), 502

    try:
        answer = resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return jsonify({"error": "Unexpected response format from Groq"}), 502

    return jsonify({"answer": answer})

@app.route('/weather')
def weatherInfo():
    return render_template('weatherInfo.html')

if __name__ == "__main__":
    app.run(debug=True)
