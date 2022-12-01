import gc
import json
import argparse
import torch
import torch.nn as nn
import pandas as pd
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from utils.ealry_stopping import EarlyStopping
from utils.decorators import data
torch.cuda.empty_cache()


class SimpleClassifier(nn.Module):
    def __init__(self, hidden_size) -> None:
        super(SimpleClassifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sentence_scores = self.sigmoid(h) * mask_cls.float()
        return sentence_scores


@data
class BertForExtSum(nn.Module):
    def __init__(self):
        super(BertForExtSum, self).__init__()
        self.bert = BertModel.from_pretrained(self.MODEL_NAME)
        self.encoder = SimpleClassifier(self.bert.config.hidden_size)

    def forward(self, src, segments, clss):
        """
        Args:
            src: input_ids
            segments: segment embedding
            clss: position embedding
        """
        mask_src = ~(src == 0)
        mask_cls = ~(clss == -1)
        
        top_vector = self.bert(input_ids=src, token_type_ids=segments, attention_mask=mask_src)
        top_vector = top_vector.last_hidden_state
        
        sentences_vector = top_vector[torch.arange(top_vector.size(0)).unsqueeze(1), clss]
        sentences_vector = sentences_vector * mask_cls[:, :, None].float()
        
        sentences_scores = self.encoder(sentences_vector, mask_cls).squeeze(-1)
        return sentences_scores, mask_cls


@data
class CreateDataset:
    def format_json_to_df(self, prefix: str) -> None:
        """ prefix: `train` or `valid` """
        dataframe = []
        
        if prefix == "train":
            folder = glob(self.SUMMARY_TRAIN_DIR, recursive=True)
        elif prefix == "valid":
            folder = glob(self.SUMMARY_VALID_DIR, recursive=True)
            
        for path in folder:
            with open(path, mode="r", encoding="utf-8") as f:
                file = json.load(f)
                docs = file["documents"]
                for doc in docs:
                    data = []
                    data.append(doc["category"])
                    data.append(doc["title"])
                    article = []
                    for paragraph in doc["text"]:
                        for sentence in paragraph:
                            article.append(sentence["sentence"])
                    data.append(article)
                    data.append(doc["abstractive"][0])
                    data.append(doc["extractive"])
                    data.append(doc["char_count"])
                    dataframe.append(data)
        
        df = pd.DataFrame(dataframe)
        df.columns = ["category", "title", "article", "abstractive", "extractive", "char_count"]
        df.to_csv(f"data/summary/{prefix}.csv", encoding="utf-8-sig")
        return df


@data
class SummaryDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.df = dataframe
        
    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        tokens = []
        
        for sentence in row.article:
            tokens.append(self.tokenizer(text=sentence, add_special_tokens=True))
            
        src, labels, segments, clss = [], [], [], []
        flag = 0
        
        for token in tokens:
            if flag > 1:
                flag = 0
            clss = clss + [len(src)]
            src = src + token["input_ids"]
            segments = segments + [flag] * len(token["input_ids"])

            if tokens.index(token) in (row.extractive):
                labels.append(1)
            else:
                labels.append(0)

            flag += 1
            
            if len(src) == args.max_seq_len:
                break
            
            elif len(src) > args.max_seq_len:
                src = src[:args.max_seq_len-1] + [src[-1]]
                segments = segments[:args.max_seq_len]
                break
        
        if len(src) < args.max_seq_len:
            src = src + [0] * (args.max_seq_len - len(src))
            segments = segments + [0] * (args.max_seq_len - len(segments))
        
        if len(clss) < args.max_seq_len:
            clss = clss + [-1] * (args.max_seq_len - len(clss))
        
        if len(labels) < args.max_seq_len:
            labels = labels + [0] * (args.max_seq_len - len(labels))
        
        return dict(
            src = torch.tensor(src),
            segments = torch.tensor(segments),
            clss = torch.tensor(clss),
            labels = torch.FloatTensor(labels)
        )
    
    def __len__(self):
        return len(self.df)
        
                
if __name__ == "__main__":
    # 1. 하이퍼파라미터 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--patience", type=float, default=15)
    parser.add_argument("--output_dir", type=str, default="./data/output/summary/")
    # parser.add_argument("--param_init", type=float, default=0)
    # parser.add_argument("--param_init_glorot", type=bool, nargs='?',const=True, default=True)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--max_seq_len", type=int, default=512)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    # 2. 데이터셋 생성 & 데이터 로더
    train_df = CreateDataset().format_json_to_df("train")
    valid_df = CreateDataset().format_json_to_df("valid")
    train_dataset = SummaryDataset(train_df)
    valid_dataset = SummaryDataset(valid_df)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=0)
    valid_dataloader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)
    del train_df, valid_df, train_dataset, valid_dataset
    gc.collect()

    # 3. 모델 & 토크나이저 로딩
    model = BertForExtSum().to(device)
    
    # 4. 학습 파라미터 설정
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=0, num_warmup_steps=0)
    
    # 5. 모델 학습 & 모델 평가
    ES = EarlyStopping(patience=args.patience)
    WS = SummaryWriter(log_dir=args.output_dir)
    for epoch in range(args.num_epochs):
        model.train()
        train_loss, correct, count = 0, 0, 0
        for train_idx, batch in enumerate(tqdm(train_dataloader)):
            src = batch['src'].to(device)
            segments = batch['segments'].to(device)
            clss = batch['clss'].to(device)
            labels = batch['labels'].to(device)
            sentences_scores, mask_cls = model(src, segments, clss)
            loss = criterion(sentences_scores, labels)
            loss = (loss * mask_cls.float().sum() / len(labels))
            WS.add_scalar("loss/train", loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # prevent exploding gradient.
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            _, preds = torch.max(sentences_scores, dim=1)
            count += labels.size(1)
            correct += preds.eq(labels.size(1)).sum().item()
        train_accuracy = round((correct/count), 4)
        avg_train_loss = round(train_loss / len(train_dataloader), 4)
        
        model.eval()
        valid_loss, correct, count = 0, 0, 0
        for valid_idx, batch in enumerate(tqdm(valid_dataloader)):
            src = batch['src'].to(device)
            segments = batch['segments'].to(device)
            clss = batch['clss'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                sentences_scores, mask_cls = model(src, segments, clss)
            loss = criterion(sentences_scores, labels)
            loss = (loss * mask_cls.float().sum() / len(labels))
            WS.add_scalar("loss/valid", loss, epoch)
            valid_loss += loss.item()
            _, preds = torch.max(sentences_scores, dim=1)
            count += labels.size(1)
            correct += preds.eq(labels.size(1)).sum().item()
        valid_accuracy = round((correct/count), 4)
        avg_valid_loss = round(valid_loss / len(valid_dataloader), 4)
        
        print (f"[*] epoch: {epoch} | train accuracy: {train_accuracy} | train loss: {avg_train_loss} | valid accuracy: {valid_accuracy} | valid loss: {avg_valid_loss}")
        if ES.is_stop(avg_valid_loss):
            break
    
    # 6. 모델 저장
    torch.save(model.state_dict(), args.output_dir + "pytorch_model.bin")
    WS.close()