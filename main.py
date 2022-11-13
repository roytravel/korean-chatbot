# -*- coding:utf-8 -*-
import os
import argparse
import torch
from transformers import Trainer, TrainingArguments

from dataset import NSMC, General
from preprocess.tokenizer import AutoTokenizer
from preprocess.preprocessor import preprocess_nsmc_dataset, preprocess_general_dataset
from model.model import BertForSequenceClassification
from utils.metrics import compute_metrics
from predict import Predict

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_FILE_PATH = "./test.pth"
CHECKPOINT_FILE_PATH = "data/checkpoint/"
MODEL_NAME = "bert-base-multilingual-cased"
MODEL_PATH = "./data/output/model.pth"

if __name__ == "__main__":
    # 1. 하이퍼 파라미터 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type = int, default=1000)
    parser.add_argument("--batch_size", type = int, default=32)
    parser.add_argument("--learning_rate", type = float, default=0.00001)
    parser.add_argument("--max_seq_len", type = int, default=128)
    parser.add_argument("--dropout_rate", type = float, default=0.2)
    parser.add_argument('--vocab_size', type = int, default=32000)
    parser.add_argument('--num_classes', type = int, default=4)
    parser.add_argument('--seed', type = int, default=521)
    args = parser.parse_args()
    
    seed = torch.manual_seed(args.seed)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
   
    # 2. 토크나이저 로딩
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 3. 데이터셋 로딩
    # train_data, test_data, train_label, test_label = preprocess_nsmc_dataset()
    # train_dataset = NSMC(train_data, train_label)
    # test_dataset = NSMC(test_data, test_label)
    
    train_data, test_data, train_label, test_label = preprocess_general_dataset()
    train_dataset = General(train_data, train_label)
    test_dataset = General(test_data, test_label)
    
    # 4. 학습 인자 설정
    training_args = TrainingArguments(
    output_dir='./data/output',
    num_train_epochs=3,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./data/logs',
    logging_steps=500,
    save_steps=500,
    save_total_limit=3
    )
    
    # 5. 모델 로드
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4).to(device)

    # 6. 학습
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    
    # 7. 평가
    trainer = Trainer(model=model,args=training_args, compute_metrics=compute_metrics)
    trainer.evaluate(eval_dataset=test_dataset)
    
    # 8. 추론
    P = Predict()    
    print(P.predict_intent("아 여행 가고 싶다 서울 여행지 추천해줘!"))