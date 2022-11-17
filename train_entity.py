import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from model.model import BertForSequenceClassification
from utils.data.dataset import Dataset
from utils.ealry_stopping import EarlyStopping
torch.cuda.empty_cache()

if __name__ == "__main__":
    # 1. 하이퍼파라미터 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="./data/output/entity/")
    parser.add_argument('--model_name', type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_training_steps', type=int, default=1000)
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--seed', type=int, default=521)
    parser.add_argument('--eps', type=float, default=1e-6) # for Adam
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    
    # 2. 랜덤 시드 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # 3. 데이터셋 로드
    dataset = Dataset()
    train_dataloader, test_dataloader = dataset.bring_entity()
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=args.num_training_steps, num_warmup_steps=args.num_warmup_steps)
    
    ES = EarlyStopping(patience=args.patience)
    writer = SummaryWriter(args.output_dir)
    for epoch in range(args.num_epochs):
        model.train()
        train_loss, correct, count = 0, 0, 0
        for train_idx, (text, label) in enumerate(tqdm(train_dataloader)):
            input_ids = text[0]['input_ids'].to(device) # 3 x 32
            attention_mask = text[0]['attention_mask'].to(device)
            token_type_ids = text[0]['token_type_ids'].to(device)
            label = label[0].to(device) # [3]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(outputs[0], label)
            writer.add_scalar("loss/train", loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # prevent exploding gradient.
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
            label = label[0].to(device)
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
    
    model.save_pretrained(args.output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.output_dir)
    writer.close()
    
    # tokenizer.save_pretrained(args.output_dir)
    # label_map = {i : label for i, label in enumerate(LABELS, start=1)}
    # model_config = {"max_seq_length":128, "num_labels":len(LABELS)+1, "label_map":label_map}
    # json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))