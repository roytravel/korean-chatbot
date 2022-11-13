import torch
import numpy as np
from transformers import AutoTokenizer

class Predict:
    def __init__(self) -> None:
        MODEL_NAME = "bert-base-multilingual-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = torch.load('./data/output/model.pth')

    def predict_intent(self, sentence):
        self.model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenized_sentnece = self.tokenizer(sentence, max_length=128, truncation=True, add_special_tokens=True, return_tensors="pt")
        tokenized_sentnece.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=tokenized_sentnece['input_ids'],
                            attention_mask=tokenized_sentnece['attention_mask'],
                            token_type_ids=tokenized_sentnece['token_type_ids'])
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits)
        return result