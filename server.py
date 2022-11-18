from datetime import datetime
from flask import Flask, jsonify, request
from predict import Predict
from core.action import define_action, trigger_action

def query_logging(current, ip, intent, query, action_type):
    """ 현재 시간, 사용자 IP, 쿼리, 의도 분류 결과 로그 수집 """
    with open('./data/log/query.log', mode='a+', encoding='utf-8') as f:
        f.writelines(' '.join([current, ip, str(intent), query, str(action_type), '\n']))


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == "GET":
        current = str(datetime.utcnow())
        ip      = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        query   = request.args['query']
        intent  = P.predict_intent(query)
        intent, action_type = define_action(intent)
        message = trigger_action(intent, action_type)
        query_logging(current, ip, intent, query, action_type)
        return message

if __name__ == '__main__':
    P = Predict()
    app.run('127.0.0.1', 5000)