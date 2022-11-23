import torch
import numpy as np
from utils.decorators import data
from transformers import AutoTokenizer
from transformers import BertConfig, BertForSequenceClassification
from model.model import CustomBertForQuestionAnswering

@data
class Predict:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.klue_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        intent_config = BertConfig.from_json_file(self.INTENT_OUTPUT_DIR+self.CONFIG)
        entity_config = BertConfig.from_json_file(self.ENTITY_OUTPUT_DIR+self.CONFIG)
        question_config = BertConfig.from_pretrained(self.QUEST_OUTPUT_DIR+self.CONFIG)
        question_config.max_length = self.QA_MAX_SEQ_LEN
    
        self.intent_model = BertForSequenceClassification.from_pretrained(self.INTENT_OUTPUT_DIR+self.MODEL_FILE_NAME, config=intent_config)
        self.entity_model = BertForSequenceClassification.from_pretrained(self.ENTITY_OUTPUT_DIR+self.MODEL_FILE_NAME, config=entity_config)
        self.quest_model = CustomBertForQuestionAnswering.from_pretrained(self.QUEST_OUTPUT_DIR+self.MODEL_FILE_NAME, config=question_config)
        
    def predict_intent(self, sentence: str) -> int:
        input = self.tokenizer(sentence, max_length=128, truncation=True, return_tensors="pt")
        self.intent_model.eval()
        with torch.no_grad():
            outputs = self.intent_model(input_ids=input['input_ids'],
                            attention_mask=input['attention_mask'],
                            token_type_ids=input['token_type_ids'])
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        intent = int(np.argmax(logits))
        return intent
    
    def predict_entity(self, sentence: str) -> list:
        entities = []
        sentence = sentence.split()
        input = self.tokenizer(sentence, max_length=128, truncation=True, padding=True, return_tensors="pt")
        self.entity_model.eval()
        with torch.no_grad():
            for i in range(input['input_ids'].shape[0]):
                outputs = self.entity_model(input_ids=input['input_ids'][i].unsqueeze(0),
                                attention_mask=input['attention_mask'][i].unsqueeze(0),
                                token_type_ids=input['token_type_ids'][i].unsqueeze(0))
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                entity = np.argmax(logits)
                entities.append(entity)
        entities = zip(sentence, entities)
        return entities
    
    def predict_question(self, context: str, query: str) -> str:
        input = self.klue_tokenizer(query, context, add_special_tokens=True, padding="longest", max_length=512, truncation=True, return_tensors="pt")
        self.quest_model.eval()
        with torch.no_grad():
            outputs = self.quest_model(**input)
            preds = np.argmax(outputs, axis=1)
            start = preds[0][0]
            end = preds[0][1]
            return context[start:end]