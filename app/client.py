import os
import sys
import requests
from flask import Flask, jsonify, request
from gtts import gTTS
from playsound import playsound
from datetime import datetime
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

SERVER = "http://127.0.0.1:5000/?query={}"

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
@app.route('/', methods=['GET'])
def user():
    if request.method == "GET":
        query = request.args['query']
        response = requests.get(url=SERVER.format(query))
        text = response.text
        tts = gTTS(text=text, lang="ko")
        now = str(datetime.now())
        PATH = "../data/log/{}.mp3".format(now).replace(":", ".")
        tts.save(PATH.format(now))
        playsound(PATH)
        return jsonify(text)

if __name__ == '__main__':
    app.run('127.0.0.1', 4999)