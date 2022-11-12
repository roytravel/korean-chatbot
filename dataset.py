import json
import pandas as pd
from torch.utils.data import Dataset

class TestDataset(Dataset):
    """ Pytorch Dataset 모듈 상속 """
    def __init__(self) -> None:
        self.LABEL = {'greeting': 0, 'get_weather': 1, 'get_fine_dust': 2}
        data = json.load(open('data.json', mode='r', encoding='utf-8'))
        self.data = data['kochat_nlu_data']['common_examples']
    
    def __getitem__(self, index):
        """ 인덱스에 해당하는 데이터 반환 """
        return self.data[index]['text'], self.LABEL[self.data[index]['intent']]
    
    def __len__(self):
        """ 데이터셋 개수 반환 """
        return len(self.data)

class GeneralDataset(Dataset):
    def __init__(self) -> None:
        self.LABEL = {'weather':0, 'dust':1, 'travel':2, 'restaurant':3}
        self.df = pd.read_csv('data/intent_data.csv', encoding='utf-8')
    
    def __getitem__(self, index):
        return self.df['question'][index], self.LABEL[self.df['label'][index]]
    
    def __len__(self):
        return len(self.df)