from transformers import BertForSequenceClassification

class BertForSequenceClassification(BertForSequenceClassification):
    """ BertForSequenceClassification Wrapper 클래스 """
    def __init__(self, config):
        super().__init__(config)