import os
import numpy as np
import argparse
import torch
from transformers import get_linear_schedule_with_warmup
from glob import glob
from tqdm import tqdm
from utils.data import Dataset
from model.model import BertForSequenceClassification
from utils.ealry_stopping import EarlyStopping

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()

if __name__ == "__main__":
    # 1. 하이퍼 파라미터 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="data/output/intent/")
    parser.add_argument('--model_name', type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_training_steps', type=int, default=1000)
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=521)
    parser.add_argument('--eps', type=float, default=1e-6) # for Adam
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()
    
    # 2. 랜덤 시드 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
   
    # 3. 데이터셋 로딩
    D = Dataset()
    train_dataloader, test_dataloader = D.bring_intent()
    
    # 4. 모델 로드 
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_classes).to(device)
    
    # 5. 학습 하이퍼파라미터 설정
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=args.num_training_steps, num_warmup_steps=args.num_warmup_steps)
    
    # 6. 학습 & 평가
    ES = EarlyStopping(patience=args.patience)
    for epoch in range(1, args.num_epochs+1):
        model.train()
        train_loss, correct, count = 0, 0, 0
        for train_idx, (text, label) in enumerate(tqdm(train_dataloader)):
            input_ids = text[0]['input_ids'].to(device)
            attention_mask = text[0]['attention_mask'].to(device)
            token_type_ids = text[0]['token_type_ids'].to(device)
            label = label[0].unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) 
            loss = criterion(outputs[0], label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm) # prevent exploding gradient.
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs[0], dim=1)
            count += label.size(0)
            correct += preds.eq(label).sum().item()
            
        train_accuracy = round((correct/count), 4)
        avg_train_loss = round(train_loss / len(train_dataloader), 4)

        model.eval()
        valid_loss, correct, count = 0, 0, 0
        for valid_idx, (text, label) in enumerate(test_dataloader):
            input_ids = text[0]['input_ids'].to(device)
            attention_mask = text[0]['attention_mask'].to(device)
            token_type_ids = text[0]['token_type_ids'].to(device)
            label = label[0].unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(outputs[0], label)
            valid_loss += loss.item()
            _, preds = torch.max(outputs[0], dim=1)
            count += label.size(0)
            correct += preds.eq(label).sum().item()
            
        valid_accuracy = round((correct/count), 4)
        avg_valid_loss = round(valid_loss / len(test_dataloader), 4)
        print (f"[*] epoch: {epoch} | train accuracy: {train_accuracy} | train loss: {avg_train_loss} | valid accuracy: {valid_accuracy} | valid loss: {avg_valid_loss}")
        if ES.is_stop(avg_valid_loss):
            break
    
    # 7. 모델 저장
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.output_dir)
    # torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))