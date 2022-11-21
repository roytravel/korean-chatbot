# Korean Chatbot
This Korean chatbot is a task-oriented Korean dialogue system. it orients a multi-turn and an open vocabulary(ontology-free) dialogue system.

## Environment and Installation
- Windows 10
- CUDA 11.2
- python 3.6
- pip install -r requirements.txt

## Model Training
Run following two scripts to fine-tune the model.
```
    python train_intent.py
    python train_entity.py
```
if you run a script, then it makes fine-tuned model. BERT model is used to train both an intent classification model and entity recognition model for now. other model will be updated. 

## Model Inference
If you want to do inference, then you can run the following script. import Predict class from `predict.py` then create an instance method with a sentence that you want to query.
```
    from predict import Predict

    P = Predict()

    sentence = ""
    intent = P.predict_intent(sentence)
    entity = P.predict_entity(sentence)

    print (intent, entity)

```

## Server
Run following script to run an API server. 
```
    python server.py
```