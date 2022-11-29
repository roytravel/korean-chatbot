import argparse
import json
import zipfile
from glob import glob
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader, Dataset

SUMMARY_TRAIN_DIR = ""
SUMMARY_VALID_DIR = ""

class SummaryDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        return 

def make_dataset():
    train_folder = glob(SUMMARY_TRAIN_DIR, recursive=True)
    valid_folder = glob(SUMMARY_VALID_DIR, recursive=True)
    
    # 본문 최대 길이 검증 과정 필요
    train_sentence, train_label = [], []
    for path in train_folder: # train_folder에 있는 모든 파일에 대해 (3개)
        with open(path, mode="r", encoding="utf-8") as f:
            file = json.load(f)
            docs = file["documents"] # documents 부분만 추출
            for doc in docs:
                text = doc["text"] # sentences
                extractive = doc["extractive"] # label
                num_paragraph = len(text) # 문단 개수
                
                sentences, labels = [], []
                for i in range(num_paragraph):
                    num_sentence = len(text[i]) # 문단 속 문장 개수
                    for j in range(num_sentence):
                        sentence = text[i][j]["sentence"]
                        sentences.append(sentence)
                        
                for i in range(len(sentences)): # 라벨링
                    if i in extractive:
                        labels.append(1)
                    else:
                        labels.append(0)
                        
                train_sentence.append(sentences)
                train_label.append(labels)
                break
    
    train_dataset = None
    valid_dataset = None
    return train_dataset, valid_dataset
       
                
if __name__ == "__main__":
    # 1. 하이퍼파라미터 설정
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int, default=32)
    # args = parser.parse_args()
    
    # 2. 데이터셋 생성 & 데이터로더
    train_dataset, valid_dataset = make_dataset()
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)
    # valid_dataloader = DataLoader(dataset=valid_dataset, batch_sampler=args.batch_size)

    # 3. 모델 & 토크나이저 로딩
    # config = BertConfig.from_pretrained("bert-base-multilingual-cased")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    # model = BertModel.from_pretrained("bert-base-multilingual-cased")
    
    # 4. 모델 학습 & 모델 평가

    # 5. 모델 저장