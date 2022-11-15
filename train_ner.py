import argparse
import numpy as np
import torch
from tqdm import tqdm, trange

from model.model import BertForSequenceClassification
from utils.data.tokenizer import BertTokenizer, AutoTokenizer
from preprocess.preprocessor import preprocess_ner_dataset
from utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=521)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./data/output/ner/')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    dataset = Dataset()
    train_dataset, test_dataset = dataset.bring_entity()

    MODEL_NAME = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=15).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, verbose=True, patience=3)
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        correct, count = 0, 0
        for train_idx, (text, label) in enumerate(tqdm(train_dataset)):
            input_ids = text['input_ids'].to(device)
            attention_mask = text['attention_mask'].to(device)
            token_type_ids = text['token_type_ids'].to(device)
            label = label.to(device)
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
                print (f"[*] Epoch: {epoch} \t Step: {train_idx}/{len(train_dataset)}\t train accuracy: {accuracy} \t train loss: {train_losss}")

    # model.eval()
    # for epoch in trange(args.num_epochs):
    #     train_loss = 0
    #     nb_tr_examples, nb_tr_steps = 0, 0
    #     for batch_idx, batch in enumerate(tqdm(train_dataloader)):
    #         batch = tuple(t.to(device) for t in batch)
    #         input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
    #         loss = model(input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch)
    #         loss.backward()
    #         torch.nn.utils.clip_grad(model.parameters(), args.max_grad_norm)
    #         train_loss += loss.item()
    #         nb_tr_examples += input_ids.size(0)
    #         nb_tr_steps += 1
    #         if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
    #             optimizer.step()
    #             scheduler.step()
    #             model.zero_grad()
    #             global_step += 1
                
    # model_to_save = model.module if hasattr(model, 'module') else model
    # model_to_save.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)

    # label_map = {i : label for i, label in enumerate(LABELS, start=1)}
    # model_config = {"max_seq_length":128, "num_labels":len(LABELS)+1, "label_map":label_map}
    # json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))