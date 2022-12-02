import kss
import torch
import numpy as np
from utils.decorators import hyperparameter
from transformers import AutoTokenizer
from transformers import BertConfig, BertForSequenceClassification, BertModel
from model import CustomBertForQuestionAnswering
from model import BertForExtSum

@hyperparameter
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
        self.summary_model = torch.load(self.SUMMARY_OUTPUT_DIR + self.MODEL_FILE_NAME).to('cpu')
        # self.summary_model = model.load_state_dict(torch.load(self.SUMMARY_OUTPUT_DIR+self.MODEL_FILE_NAME))
        
        
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
        
    def predict_summary(self, sentences: list) -> list:
        tokens = []
    
        for sentence in sentences:
            tokens.append(self.tokenizer(text=sentence, add_special_tokens=True))
            
        src, segments, clss = [], [], []
        flag = 0
        
        for token in tokens:
            if flag > 1:
                flag = 0
            clss = clss + [len(src)]
            src = src + token["input_ids"]
            segments = segments + [flag] * len(token["input_ids"])

            flag += 1
            
            if len(src) == self.SUMMARY_MAX_SEQ_LEN:
                break
            
            elif len(src) > self.SUMMARY_MAX_SEQ_LEN:
                src = src[:self.SUMMARY_MAX_SEQ_LEN:-1] + [src[-1]]
                segments = segments[:self.SUMMARY_MAX_SEQ_LEN:]
                break
        
        if len(src) < self.SUMMARY_MAX_SEQ_LEN:
            src = src + [0] * (self.SUMMARY_MAX_SEQ_LEN - len(src))
            segments = segments + [0] * (self.SUMMARY_MAX_SEQ_LEN - len(segments))
        
        if len(clss) < self.SUMMARY_MAX_SEQ_LEN:
            clss = clss + [-1] * (self.SUMMARY_MAX_SEQ_LEN - len(clss))
        
        data = dict(
            src = torch.tensor(src),
            segments = torch.tensor(segments),
            clss = torch.tensor(clss),
        )
        
        self.summary_model.eval()
        with torch.no_grad():
            src = data["src"].unsqueeze(0).to('cpu')
            segments = data["segments"].unsqueeze(0).to('cpu')
            clss = data["clss"].unsqueeze(0).cpu().to('cpu')
            _, output = self.summary_model(src, segments, clss)
        
        output = output.squeeze(0)
        output_sort, idx = output.sort(descending=True)
        output_sort = output_sort.tolist()
        idx = idx.tolist()
        end_idx = output_sort.index(0)
        output_sort = output_sort[:end_idx]
        idx = idx[:end_idx]
        if len(idx) > 3:
            result = idx[:3]
        else:
            result = idx
        
        sents = []
        for r in result:
            sents.append(sentences[r])
        return sents