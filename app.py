import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

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


@app.route("/")
def home():
    return render_template("home.html")


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


if __name__ == "__main__":
    app.run(debug=True)
