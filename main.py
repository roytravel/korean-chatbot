# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader

from dataset import MyDataset, load_dataset
from tokenizer import tokenize
from model import Model


if __name__ == "__main__":
    PATH = "./test.pth"
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type = int, default=1000)
    parser.add_argument("--batch_size", type = int, default=32)
    parser.add_argument("--learning_rate", type = float, default=0.0001)
    args = parser.parse_args()

    texts, labels = load_dataset()
    dataset = MyDataset(texts, labels)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))

    for epoch in range(args.num_epoch):
        model.train()
        train_loss = 0
        correct, count = 0, 0
        for batch_idx, (text, label) in enumerate(train_dataloader):
            text, label = tokenize(text).to(device), label.to(device)
            output = model(text).unsqueeze(0)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(output, 1)
            count += label.size(0)
            correct = preds.eq(label).sum().item()
            print (f"[*] Epoch: {epoch} \tStep: {batch_idx}/{len(train_dataloader)}\tTrain accuracy: {round((correct/count)*100, 4)} \tTrain Loss: {round((train_loss/count)*100, 4)}")
        
        # with torch.no_grad():
        #     valid_loss = 0
        #     for batch_idx, (text, label) in enumerate(test_dataloader):
        #         text, label = tokenize(text).to(device), label.to(device)
        #         output = model(text)
        #         output.unsqueeze(0)
        #         optimizer.zero_grad()
        #         loss = criterion(output, label)
        #         loss.backward()

    torch.save(model.state_dict(), PATH)

    loaded_model = torch.load_state_dict(torch.load(PATH))