# -*- coding:utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import GeneralDataset
from tokenizer import Tokenizer
from model.model import Model, Convolution

def evaluate(text: str) -> str:
    raise NotImplementedError()


if __name__ == "__main__":
    MODEL_FILE_PATH = "./test.pth"
    CHECKPOINT_FILE_PATH = "data/checkpoint/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type = int, default=1000)
    parser.add_argument("--batch_size", type = int, default=32)
    parser.add_argument("--learning_rate", type = float, default=0.00001)
    parser.add_argument("--max_seq_len", type = int, default=128)
    parser.add_argument("--dropout_rate", type = float, default=0.2)
    parser.add_argument('--vocab_size', type = int, default=32000)
    parser.add_argument('--d_model', type = int, default=128)
    parser.add_argument('--d_ff', type = int, default=512)
    parser.add_argument('--num_layers', type = int, default=6)
    parser.add_argument('--num_heads', type = int, default=8)
    parser.add_argument('--num_classes', type = int, default=4)
    parser.add_argument('--seed', type = int, default=521)
    args = parser.parse_args()
    
    seed = torch.initial_seed(args.seed)

    T = Tokenizer()
    dataset = GeneralDataset()
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(input_size=args.max_seq_len, hidden_size=64, num_classes=args.num_classes).to(device)
    # model = Convolution(in_channel=args.max_seq_len, out_channel=args.d_model, kernel_size=1, residual=True)
    
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.1, verbose=True, patience=3)

    for epoch in range(args.num_epoch):
        model.train()
        train_loss = 0
        correct, count = 0, 0
        for batch_idx, (text, label) in enumerate(train_dataloader):
            text, label = T.tokenize(text[0]).to(device), label.to(device)
            output = model(text).unsqueeze(0)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(output, 1)
            count += label.size(0)
            correct = preds.eq(label).sum().item()
            print (f"[*] Epoch: {epoch} \tStep: {batch_idx}/{len(train_dataloader)}\tTrain accuracy: {round((correct/count), 4)} \tTrain Loss: {round((train_loss/count), 4)}")
        lr_scheduler.step(metrics=loss)
        
        # with torch.no_grad():
        #     valid_loss = 0
        #     for batch_idx, (text, label) in enumerate(test_dataloader):
        #         text, label = tokenize(text).to(device), label.to(device)
        #         output = model(text)
        #         output.unsqueeze(0)
        #         optimizer.zero_grad()
        #         loss = criterion(output, label)
        #         loss.backward()
        
    torch.save(model.state_dict(), MODEL_FILE_PATH)
    
    # model.load_state_dict(torch.load(MODEL_FILE_PATH))
    # model.eval()
    # os.path.join(CHECKPOINT_FILE_PATH, f"{time.time().pth}")
    # state = {
    #     'epoch': epoch,
    #     'optimizer': optimizer.state_dict(),
    #     'model': model.state_dict(),
    #     'seed': seed,
    # }