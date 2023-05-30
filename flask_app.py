from flask import Flask, request, render_template
from pathlib import Path
import tensorflow as tf

from clean_text import clean_texts

CWD = Path(__file__).parent.resolve()

model = tf.keras.models.load_model(CWD / 'models/cyberbullying-bdlstm.h5')

with open(CWD / 'models/tokenizer.json') as file:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(file.read())

app = Flask(__name__)

@app.route("/cyberbully", methods=['GET'])
def cyberbully():
    messages = clean_texts([request.args.get("msg")], tokenizer)
    return {"score":model.predict(messages).tolist()[0][0]}
convos = []
@app.route('/get')
def get():
    userText = request.args.get('msg')
    import requests
    botText = requests.post("https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill", headers={"Authorization": "Bearer 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"}, json={"inputs": {"past_user_inputs": [i[0] for i in convos], "generated_responses": [i[1] for i in convos], "text": userText}, "parameters": {"repetition_penalty": 1.33}}).json()["generated_text"]
    convos.append((userText, botText))
    return {"bot": botText.strip()}

@app.route('/', methods=['GET'])
def main():
    return render_template('chat.html')

@app.route('/echo', methods=['GET', 'POST'])
def echo():
    if 'echo' in request.args:
        return request.args.get('echo')
    else:
        return "Param not found", 404



app.run(debug=True)