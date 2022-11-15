import argparse
import numpy as np
import torch
from tqdm import tqdm

from model.model import BertForSequenceClassification
from utils.data.tokenizer import AutoTokenizer
from utils.data import Dataset
torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=521)
    parser.add_argument('--output_dir', type=str, default='./data/output/entity/')
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--logging_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    dataset = Dataset()
    train_dataset, test_dataset = dataset.bring_entity()
    MODEL_NAME = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=15).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, verbose=True, patience=3)
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        correct, count = 0, 0
        for train_idx, (text, label) in enumerate(tqdm(train_dataset)):
            input_ids = text[0]['input_ids'].to(device)
            attention_mask = text[0]['attention_mask'].to(device)
            token_type_ids = text[0]['token_type_ids'].to(device)
            label = label[0].to(device)
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
                print (f" [*] Epoch: {epoch} \t Step: {train_idx}/{len(train_dataset)}\t train accuracy: {accuracy} \t train loss: {train_losss}")
    
    model.save_pretrained(args.output_dir)
    # model_to_save = model.module if hasattr(model, 'module') else model
    # model_to_save.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)

    # label_map = {i : label for i, label in enumerate(LABELS, start=1)}
    # model_config = {"max_seq_length":128, "num_labels":len(LABELS)+1, "label_map":label_map}
    # json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))