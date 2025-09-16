from flask import Flask, render_template, request
from googletrans import Translator

app = Flask(__name__)
translator = Translator()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/askAnything')
def askUI():
    return render_template('askAnything.html')

@app.route("/processQuestion", methods=["POST"])
def process_question():
    question = request.form.get("question")
    lang = request.form.get("lang", "en")

    # detect and translate to English
    detected = translator.detect(question).lang
    translated_q = translator.translate(question, src=detected, dest="en").text

    # dummy AI response (replace with real AI/ML logic)
    ai_response_en = f"This is a demo response for: {translated_q}"

    # translate back to original language
    final_response = translator.translate(ai_response_en, src="en", dest=detected).text

    return render_template("askAnything.html", question=question, response=final_response)


if __name__ == "__main__":
    app.run(debug=True)