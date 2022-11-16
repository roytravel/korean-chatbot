import torch
import numpy as np
from utils.decorators import data
from transformers import AutoTokenizer
from transformers import BertConfig, BertForSequenceClassification

@data
class Predict:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = BertConfig.from_json_file(self.OUTPUT_DIR+self.CONFIG)
        self.model = BertForSequenceClassification.from_pretrained(self.OUTPUT_DIR+self.MODEL_FILE_NAME, config=config)
        
    def predict_intent(self, sentence):
        input = self.tokenizer(sentence, max_length=128, truncation=True, add_special_tokens=True, return_tensors="pt")
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input['input_ids'],
                            attention_mask=input['attention_mask'],
                            token_type_ids=input['token_type_ids'])
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        intent = np.argmax(logits)
        return intent