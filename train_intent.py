# -*- coding:utf-8 -*-
import os
import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertConfig
from glob import glob
from tqdm import tqdm

from utils.data import Dataset
from model.model import BertForSequenceClassification

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()

CHECKPOINT_FILE_PATH = "data/output/"
MODEL_NAME = "bert-base-multilingual-cased"
MODEL_PATH = "./data/output/model.pth"

if __name__ == "__main__":
    # 1. 하이퍼 파라미터 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type = int, default=100)
    parser.add_argument("--batch_size", type = int, default=32)
    parser.add_argument("--learning_rate", type = float, default=5e-5)
    parser.add_argument("--max_seq_len", type = int, default=128)
    parser.add_argument("--dropout_rate", type = float, default=0.2)
    parser.add_argument('--num_classes', type = int, default=4)
    parser.add_argument('--seed', type = int, default=521)
    args = parser.parse_args()
    
    seed = torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # 2. 데이터셋 로딩
    D = Dataset()
    train_dataset, test_dataset = D.bring_intent()
    
    # 3. 모델 로드
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=args.num_classes).to(device)
    
    # 4. 모델 하이퍼파라미터 설정
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.1, verbose=True, patience=3)

    # 5. 학습 & 평가
    for epoch in range(args.num_epoch):
        model.train()
        train_loss = 0
        correct, count = 0, 0
        for train_idx, (text, label) in enumerate(tqdm(train_dataset)):
            input_ids = text[0]['input_ids'].to(device)
            attention_mask = text[0]['attention_mask'].to(device)
            token_type_ids = text[0]['token_type_ids'].to(device)
            label = label[0].unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) 
            loss = criterion(outputs[0], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs[0], dim=1)
            
            count += label.size(0)
            correct = preds.eq(label).sum().item()
            accuracy = round((correct/count), 4)
            train_losss = round((train_loss/count), 4)
            if train_idx % 10 == 0:
                print (f"[*] Epoch: {epoch} \t Step: {train_idx}/{len(train_dataset)} train accuracy: {accuracy} \t train loss: {train_losss}")
        
        # model.eval()
        # valid_loss = 0
        # correct, count = 0, 0
        # with torch.no_grad():    
        #     for valid_idx, (text, label) in enumerate(test_dataset):
        #         input_ids = text['input_ids'].unsqueeze(0)
        #         attention_mask = text['attention_mask'].unsqueeze(0)
        #         token_type_ids = text['token_type_ids'].unsqueeze(0)
        #         label = label.unsqueeze(0)
        #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #         loss = criterion(outputs[0], label)
        #         valid_loss += loss.item()
        #         _, preds = torch.max(outputs[0], dim=1)
        #         count += label.size(0)
        #         correct = preds.eq(label).sum().item()
        #         accuracy = round((correct/count), 4)
        #         valid_losss = round((valid_loss/count), 4)
        #         break
        #         if valid_idx % 100 == 0:
        #             print (f"[*] Epoch: {epoch} \t Step: {valid_idx}/{len(test_label)} valid accuracy: {accuracy} \t valid loss: {valid_losss}")
        # lr_scheduler.step(metrics=valid_loss)