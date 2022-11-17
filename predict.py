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
        intent_config = BertConfig.from_json_file(self.INTENT_OUTPUT_DIR+self.CONFIG)
        entity_config = BertConfig.from_json_file(self.ENTITY_OUTPUT_DIR+self.CONFIG)
        self.intent_model = BertForSequenceClassification.from_pretrained(self.INTENT_OUTPUT_DIR+self.MODEL_FILE_NAME, config=intent_config)
        self.entity_model = BertForSequenceClassification.from_pretrained(self.ENTITY_OUTPUT_DIR+self.MODEL_FILE_NAME, config=entity_config)
        
    def predict_intent(self, sentence: str) -> int:
        input = self.tokenizer(sentence, max_length=128, truncation=True, return_tensors="pt")
        self.intent_model.eval()
        with torch.no_grad():
            outputs = self.intent_model(input_ids=input['input_ids'],
                            attention_mask=input['attention_mask'],
                            token_type_ids=input['token_type_ids'])
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        intent = np.argmax(logits)
        return intent
    
    def predict_entity(self, sentence: str) -> list:
        sentence = sentence.split()
        input = self.tokenizer(sentence, max_length=128, truncation=True, padding=True, return_tensors="pt")
        self.entity_model.eval()
        entities = []
        with torch.no_grad():
            for i in range(input['input_ids'].shape[0]):
                outputs = self.entity_model(input_ids=input['input_ids'][i].unsqueeze(0),
                                attention_mask=input['attention_mask'][i].unsqueeze(0),
                                token_type_ids=input['token_type_ids'][i].unsqueeze(0))
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                entity = np.argmax(logits)
                entities.append(entity)
        return entities