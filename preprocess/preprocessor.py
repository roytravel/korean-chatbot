import datasets
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-multilingual-cased"

def preprocess_nsmc_dataset():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # NSMC 데이터셋을 DataFrame 형식으로 생성
    dataset = datasets.load_dataset('nsmc')
    train_data = pd.DataFrame({"document" : dataset['train']['document'], "label" : dataset['train']['label']})
    test_data = pd.DataFrame({"document" : dataset['test']['document'], "label" : dataset['test']['label']})
    
    # 중복 레코드 제거
    train_data.drop_duplicates(subset=['document'], inplace= True)
    test_data.drop_duplicates(subset=['document'], inplace= True)
    
    # 결측 레코드 제거 (null인 레코드)
    train_data['document'].replace('', np.nan, inplace=True)
    test_data['document'].replace('', np.nan, inplace=True)
    train_data = train_data.dropna(how = 'any')
    test_data = test_data.dropna(how = 'any')
    
    train_data = train_data[:100]
    test_data = test_data[:100]
    
    train_label = train_data['label'].values
    test_label = test_data['label'].values
    
    train_data = tokenizer(list(train_data['document']), padding=True, truncation=True, 
                                            add_special_tokens=True, return_tensors="pt")
    test_data = tokenizer(list(test_data['document']), padding=True, truncation=True,
                                         add_special_tokens=True, return_tensors="pt")
    
    return train_data, test_data, train_label, test_label

def preprocess_general_dataset():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = pd.read_csv('./data/intent_data.csv')
    df['label'] = df['label'].replace('weather', 0)
    df['label'] = df['label'].replace('dust', 1)
    df['label'] = df['label'].replace('travel', 2)
    df['label'] = df['label'].replace('restaurant', 3)
    
    df['label'] = np.array(df['label'], dtype=np.int64)
    
    train_label = df['label'][:16000].values
    test_label = df['label'][16000:].values
    
    train_data = df[:16000]
    test_data = df[16000:]
    
    train_data = tokenizer(list(train_data['question']), padding=True, truncation=True, 
                                            add_special_tokens=True, return_tensors="pt")
    test_data = tokenizer(list(test_data['question']), padding=True, truncation=True,
                                         add_special_tokens=True, return_tensors="pt")
    
    return train_data, test_data, train_label, test_label