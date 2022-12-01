import os
import sys
import torch
import soundfile as sf
from datetime import datetime
from flask import Flask, jsonify, request
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from predict import Predict
from core.action import define_action, trigger_action
from core.state_tracker import DialogueStateTracker
from core.agent import Agent

LOG_FILENAME = "../data/log/query.log"
SPEECH_MODEL = "kresnik/wav2vec2-large-xlsr-korean"

def get_prediction(batch):
    inputs = processor(batch["speech"], sampling_rate=16000, return_tensors="pt", padding="longest")
    input_values = inputs.input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch


def query_logging(current, ip, intent, query, action_type):
    """ 현재 시간, 사용자 IP, 쿼리, 의도 분류 결과 로그 수집 """
    with open(LOG_FILENAME, mode='a+', encoding='utf-8') as f:
        f.writelines(' '.join([current, ip, str(intent), query, str(action_type), '\n']))


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
@app.route('/', methods=['GET', 'POST'])
def bot():
    if request.method == "GET":
        current = str(datetime.utcnow())
        ip      = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        query   = request.args['query']
        # message = Translate.translate_ko_en(query)
        intent  = P.predict_intent(query)
        # entity  = P.predict_entity(query)
        # state = DST.fill_slot(intent, entity)
        # action = Agent.predict_next_action(state)
        intent, action_type = define_action(intent)
        message = trigger_action(intent, action_type)
        query_logging(current, ip, intent, query, action_type)
        return jsonify(message)
    
    elif request.method == "POST":
        f = request.files['file']
        f.save((f.filename))
        dirname = os.path.dirname(__name__)
        speech, _ = sf.read(dirname + f.filename)
        batch = {}
        batch['speech'] = speech
        result = get_prediction(batch)
        query = result['transcription']
        intent = P.predict_intent(query)
        message = trigger_action(intent, action_type)
        return jsonify(message)

if __name__ == '__main__':
    P = Predict()
    DST = DialogueStateTracker()
    AGT = Agent()
    # processor = Wav2Vec2Processor.from_pretrained(SPEECH_MODEL)
    # model = Wav2Vec2ForCTC.from_pretrained(SPEECH_MODEL)
    app.run('0.0.0.0', 5000)