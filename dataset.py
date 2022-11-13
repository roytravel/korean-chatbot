import json
import pandas as pd
import datasets
import torch
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

class General(Dataset):
    def __init__(self, encodings, label) -> None:
        self.encodings = encodings
        self.label = label

    def __getitem__(self, index):
        """ 일반적으로 Numpy 배열이나 Tensor 형식으로 반환 + input/output을 튜플 형식으로 반환 """
        #item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        encodings = {key: value[index].clone().detach() for key, value in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels[index])
        label = torch.tensor(self.label[index])
        return encodings, label
    
    def __len__(self):
        return len(self.encodings)

class NSMC(Dataset):
    def __init__(self, encodings, labels) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        """ 일반적으로 Numpy 배열이나 Tensor 형식으로 반환 + input/output을 튜플 형식으로 반환 """
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item
    
    def __len__(self):
        return len(self.labels)