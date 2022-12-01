import torch
import torch.nn as nn
import argparse
import json
import zipfile
import torch
from glob import glob
from transformers import AutoTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils.ealry_stopping import EarlyStopping
from utils.decorators import data


class Bert(nn.Module):
    def __init__(self):
        super(Bert).__init__()
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased")
        
    def forward(self, x, segments, mask):
        encoded_layers, _ = self.model(x, segments, attention_mask = mask)
        top_vector = encoded_layers[-1]
        return top_vector


class SimpleClassifier:
    def __init__(self, hidden_size) -> None:
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask_cls):
        h = self.linear(x).squeeze(-1)
        sentence_scores = self.sigmoid(h) * mask_cls.float()
        return sentence_scores


class BertForExtSum(nn.Module):
    def __init__(self):
        super(BertForExtSum).__init__()
        self.bert = Bert()
        self.encoder = SimpleClassifier(self.bert.model.config.hidden_size)
        
        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)

    def forward(self, x, segments, clss, mask, mask_cls):
        """
        Args:
            x: input_ids
            segments: segment embedding
            clss: position embedding
            mask: _description_
            mask_cls: _description_
        """
        top_vector = self.bert(x, segments, mask)
        sentences_vector = top_vector[torch.arange(top_vector.size(0)).unsqueeze(1), clss]
        sentences_vector = sentences_vector * mask_cls[:, :, None].float()
        sentences_scores = self.encoder(sentences_vector, mask_cls).squeeze(-1)
        return sentences_scores, mask_cls
        

@data
class SummaryDataset(Dataset):
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        train_folder = glob(self.SUMMARY_TRAIN_DIR, recursive=True)
        valid_folder = glob(self.SUMMARY_VALID_DIR, recursive=True)
        train_sentence, train_label = self.make_dataset(train_folder)
        valid_sentence, valid_label = self.make_dataset(valid_folder)
        
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        return 
    
    def make_dataset(self, folder: str):
        # 본문 최대 길이 검증 과정 필요
        train_sentence, train_label = [], []
        for path in folder: # train|valid folder에 있는 모든 파일에 대해 (3개)
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
                            
                    input_tensors = []
                    for sentence in sentences:
                        input_tensor = self.tokenizer(sentence).input_ids
                        input_tensors += (input_tensor)
                    train_sentence.append(sentences)
                    train_label.append(labels)
                    
        return train_dataset, valid_dataset
       
                
if __name__ == "__main__":
    # 1. 하이퍼파라미터 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--patience", type=float, default=15)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--param_init", type=float, default=0)
    parser.add_argument("--param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("--seed", type=int, default=521)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # 2. 데이터셋 생성 & 데이터로더
    train_dataset, valid_dataset = SummaryDataset()
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)
    # valid_dataloader = DataLoader(dataset=valid_dataset, batch_sampler=args.batch_size)

    # 3. 모델 & 토크나이저 로딩
    model = BertForExtSum()
    # config = BertConfig.from_pretrained("bert-base-multilingual-cased")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    # model = BertModel.from_pretrained("bert-base-multilingual-cased")
    
    # 4. 학습 파라미터 설정
    # criterion = torch.nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=0, num_warmup_steps=0)
    
    # 5. 모델 학습 & 모델 평가
    # ES = EarlyStopping(patience=args.patience)
    # WS = SummaryWriter(log_dir=args.output_dir)
    # 6. 모델 저장