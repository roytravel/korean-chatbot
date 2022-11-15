import random
import pandas as pd
import torch
from torch import Tensor
from typing import Tuple, List
from utils.decorators import data
from utils.data.tokenizer import AutoTokenizer

@data
class Dataset:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        # self.intent = self.bring_intent()
        self.entity = self.bring_entity()

    def bring_intent(self) -> Tuple[Tensor, Tensor]:
        """ intent 분류 모델 학습을 위해 문장과 라벨 가져오기. """
        sequence = self._bring_intent_sequence()
        label = self._bring_intent_label()
        return sequence, label
        
    def _bring_intent_sequence(self):
        raise NotImplementedError
    
    def _bring_intent_label(self):
        raise NotImplementedError
    
    def bring_entity(self) -> Tuple[Tensor, Tensor]:
        """ entity 식별 모델 학습을 위해 문장과 라벨 가져오기. """
        self.df = pd.read_csv(self.ENTITY_DIR)
        sequences = self.__bring_entity_sequence()
        labels = self.__bring_entity_label()
        
        train_dataset, test_dataset = self.__split_dataset(sequences, labels)
        
        train_dataset = self.__list_to_tensor(train_dataset)
        test_dataset = self.__list_to_tensor(test_dataset)
        return train_dataset, test_dataset
           
    def __bring_entity_sequence(self):
        """ 토큰화 """
        sequence = self.df['question']
        sequence = [seq.split() for seq in sequence]
        return sequence
    
    def __bring_entity_label(self) -> dict:
        """ entity 사전 생성 """
        label = self.df['label']
        label_dict = self.__make_label_dict(label)
        labels = self.__map_label(label, label_dict)
        return labels

    def __split_dataset(self, sequences, labels):
        """ 데이터셋 랜덤 셔플 후 학습, 테스트 데이터셋 분할 """
        dataset = list(zip(sequences, labels))
        random.shuffle(dataset)
        num = int(len(dataset) * self.SPLIT_RATIO)
        train_dataset = dataset[:num]
        test_dataset = dataset[num:]
        return train_dataset, test_dataset
    
    def __make_label_dict(self, label):
        """ 전체 라벨을 구하고 고유 인덱스 부여 예: {''B-DATE':0, 'B-LOCATION':1, ...} """
        label_set, label_dict = set(), dict()
        [[label_set.add(t) for t in tag.split(' ')] for tag in label]
        label_set = sorted(list(label_set))
        for idx, tag in enumerate(label_set):
            label_dict[tag] = idx
        return label_dict
    
    def __map_label(self, label, label_dict) -> List[List]:
        """ 레이블에 대해 정수 매핑 """
        labels = [[label_dict[t] for t in lb.split()] for lb in label]
        return labels
    
    def __list_to_tensor(self, dataset) -> Tuple[Tensor, Tensor]:
        """ 발화 문장과 라벨을 텐서화 """
        sequence, label = zip(*dataset)
        label, sequence = list(label), list(sequence)
        for i in range(len(label)):
            sequence[i] = self.tokenizer(sequence[i], max_length=128, padding=True, truncation=True, return_tensors="pt")
            label[i] = torch.tensor(label[i])
        dataset = list(zip(sequence, label))
        return dataset