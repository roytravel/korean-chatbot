""" 테스트용 """
import torch
from collections import defaultdict
dic = defaultdict(int)
count = 1
MAX_SEQ_LEN = 5

def tokenize(text):
    global count
    text = text[0].split()
    lists = []
    for t in text:
        if not dic[t]:
            dic[t] = count
            count += 1
        lists.append(dic[t])

    while len(lists) < MAX_SEQ_LEN:
        lists.append(0)
    
    lists = torch.Tensor(lists)
    return lists