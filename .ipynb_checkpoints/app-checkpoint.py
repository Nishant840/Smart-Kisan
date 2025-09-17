from flask import Flask, render_template, request, jsonify
import openai
from langdetect import detect
from gtts import gTTS
import base64
import io

app = Flask(__name__)

openai.api_key = 'your-openai-api-key'

LANGUAGE_MAP = {
    "en": "en", "hi": "hi", "bn": "bn", "ta": "ta", "te": "te",
    "gu": "gu", "kn": "kn", "ml": "ml", "mr": "mr", "pa": "pa", "ur": "ur"
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/askAnything')
def ask_ui():
    return render_template('askAnything.html')

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "⚠️ Please ask something.", "audio": ""})

    try:
        detected_lang = detect(question)
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=150
        )

        answer_text = response.choices[0].message["content"]

        tts_lang = LANGUAGE_MAP.get(detected_lang, "en")
        audio_base64 = ""
        try:
            mp3_fp = io.BytesIO()
            tts = gTTS(text=answer_text, lang=tts_lang)
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            audio_base64 = base64.b64encode(mp3_fp.read()).decode("utf-8")
        except Exception as e:
            print("TTS error:", e)

        return jsonify({"answer": answer_text, "audio": audio_base64})

    except Exception as e:
        return jsonify({"answer": f"⚠️ Error: {str(e)}", "audio": ""})

if __name__ == "__main__":
    app.run(debug=True)
