import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Tuple, List

from utils.decorators import hyperparameter
from utils.data.tokenizer import AutoTokenizer

@hyperparameter
class Dataset:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

    def bring_intent(self) -> Tuple[DataLoader, DataLoader]:
        """ intent 분류 모델 학습을 위해 문장과 라벨 가져오기. """
        self.intent_df = pd.read_csv(self.INTENT_FILE)
        sequences = self.__bring_intent_sequence()
        labels = self.__bring_intent_label()
        
        train_dataset, test_dataset = self.__split_dataset(sequences, labels)
        
        train_dataset = self.__tensorize_intent(train_dataset)
        test_dataset = self.__tensorize_intent(test_dataset)
        
        train_dataset = self.__make_batch(train_dataset)
        test_dataset = self.__make_batch(test_dataset)
        return train_dataset, test_dataset
        
    def __bring_intent_sequence(self) -> pd.DataFrame:
        sequences = self.intent_df['question']
        return sequences
    
    def __bring_intent_label(self):
        self.intent_df['label'] = self.intent_df['label'].replace('weather', 0)
        self.intent_df['label'] = self.intent_df['label'].replace('dust', 1)
        self.intent_df['label'] = self.intent_df['label'].replace('travel', 2)
        self.intent_df['label'] = self.intent_df['label'].replace('restaurant', 3)
        label = self.intent_df['label'].values
        return label
    
    def __tensorize_intent(self, dataset) -> Tuple[Tensor, Tensor]:
        """ 문장과 라벨 텐서화 """
        sequence, label = zip(*dataset)
        label = list(label)
        sequences = []
        for seq in sequence:
            sequences.append([seq])
        
        for i in range(len(label)):
            sequences[i] = self.tokenizer(sequences[i], max_length=self.MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
            label[i] = torch.tensor(label[i])
        
        dataset = list(zip(sequences, label))
        return dataset
    
    def bring_entity(self) -> Tuple[Tensor, Tensor]:
        """ entity 식별 모델 학습을 위해 문장과 라벨 가져오기. """
        self.entity_df = pd.read_csv(self.ENTITY_FILE)
        sequences = self.__bring_entity_sequence()
        labels = self.__bring_entity_label()

        train_dataset, test_dataset = self.__split_dataset(sequences, labels)
        
        train_dataset = self.__tensorize_entity(train_dataset)
        test_dataset = self.__tensorize_entity(test_dataset)
        
        train_dataset = self.__make_batch(train_dataset)
        test_dataset = self.__make_batch(test_dataset)
        return train_dataset, test_dataset
           
    def __bring_entity_sequence(self):
        """ 문장 토큰화 """
        sequence = self.entity_df['question']
        sequence = [seq.split() for seq in sequence]
        return sequence
    
    def __bring_entity_label(self) -> dict:
        """ entity에 대한 정수 인코딩된 라벨 생성 """
        label = self.entity_df['label']
        label_dict = self.__make_label_dict(label)
        labels = self.__map_label(label, label_dict)
        return labels

    def __split_dataset(self, sequences, labels) -> Tuple[List, List]:
        """ 학습, 테스트 데이터셋 분할 """
        dataset = list(zip(sequences, labels))
        # random.shuffle(dataset)
        num = int(len(dataset) * self.SPLIT_RATIO)
        train_dataset = dataset[:num]
        test_dataset = dataset[num:]
        return train_dataset, test_dataset
    
    def __make_label_dict(self, label) -> dict:
        """ 전체 라벨을 구하고 고유 인덱스 부여 예: {'B-DATE':0, 'B-LOCATION':1, ...} 
            만약 1부터 시작할 경우 학습 도중 에러 발생 가능 """
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
    
    def __tensorize_entity(self, dataset) -> Tuple[Tensor, Tensor]:
        """ 문장과 라벨 텐서화 """
        sequence, label = zip(*dataset)
        label, sequence = list(label), list(sequence)
        
        for i in range(len(label)):
            sequence[i] = self.tokenizer(sequence[i], max_length=self.MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
            label[i] = torch.tensor(label[i])

        dataset = list(zip(sequence, label))
        return dataset
    
    def __make_batch(self, dataset) -> DataLoader:
        dataset = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True, drop_last=False, pin_memory=True, collate_fn=self.__collate_fn)
        return dataset
        
    def __collate_fn(self, batch):
        return tuple(zip(*batch))