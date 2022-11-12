
import json
from torch.utils.data import Dataset

LABEL = {'greeting': 0,
        'get_weather': 1,
        'get_fine_dust': 2}

class MyDataset(Dataset):
    """ Pytorch Dataset 모듈 상속 """
    def __init__(self, text, label) -> None:
        self.text = text
        self.label = label
    
    def __getitem__(self, index):
        """ 인덱스에 해당하는 데이터 반환 """
        return self.text[index], self.label[index]
    
    def __len__(self):
        """ 데이터셋 개수 반환 """
        return len(self.text)

def load_dataset():
    texts, labels = [], []
    with open('data.json', mode='r', encoding='utf-8') as f:
        json_data = json.load(f)
        data = json_data['kochat_nlu_data']['common_examples']
        for idx in data:
            text = idx['text']
            label = LABEL[idx['intent']]
            texts.append(text)
            labels.append(label)
    return texts, labels