from flask import Flask, jsonify, request
from predict import Predict

text = "아 배고프다 오늘 날씨에 따라 밥 뭐먹을지 골라야징"
# text = "아 여행가고 싶다 서울 여행지 추천해줘"
# text = "아 식당 어디 가지 배고픈데 결정하기 어려워."
# text = "미세먼지 심해?"

LABEL = {
    0: 'weather',
    1: 'dust',
    2: 'travel',
    3: 'restaurant'
}

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == "GET":
        query = request.args['query']
        intent = P.predict_intent(query)
        return LABEL[intent]
    else:
        query = request.form['query']
        intent = P.predict_intent(query)
        return LABEL[intent]

if __name__ == '__main__':
    P = Predict()
    app.run('127.0.0.1', 5000)