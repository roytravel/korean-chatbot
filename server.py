from datetime import datetime
from flask import Flask, jsonify, request
from predict import Predict

LABEL = {
    0: 'weather',
    1: 'dust',
    2: 'travel',
    3: 'restaurant'
}

def query_logging(current, ip, intent, query):
    """ 현재 시간, 사용자 IP, 쿼리, 의도 분류 결과 로그 수집 """
    with open('./data/log/query.log', mode='a+', encoding='utf-8') as f:
        f.writelines(' '.join([current, ip, intent, query, '\n']))

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == "GET":
        current = str(datetime.utcnow())
        ip      = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        query   = request.args['query']
        intent  = LABEL[P.predict_intent(query)]        
        query_logging(current, ip, intent, query)
        return intent
    else:
        query = request.form['query']
        intent = P.predict_intent(query)
        return LABEL[intent]

if __name__ == '__main__':
    P = Predict()
    app.run('127.0.0.1', 5000)