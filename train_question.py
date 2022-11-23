import os
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings(action='ignore')
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, BertPreTrainedModel, BertModel, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.ealry_stopping import EarlyStopping

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()


class KorQuadDataset(Dataset):
    def __init__(self, dataset):
        data = self.make_dataset(dataset)
        self.question = data[0]
        self.context = data[1]
        self.answer_text = data[2]
        self.answer_start = data[3]
        
    def make_dataset(self, dataset):
        context, question, answer_start, answer_text = [], [], [], []
        for idx, data in enumerate(dataset):
            start = data['answers']['answer_start']
            _, start = self.get_text(data['context'], start[0])
            context.append(data['context'])
            question.append(data['question'])
            answer_text.append(data['answers']['text'][0])
            answer_start.append(start)
        
        return question, context, answer_text, answer_start
    
    def get_text(self, context, start_loc):
        contexts = context.split('. ')
        len_text_answer_idx = -1
        len_text = [0]
        for idx, text in enumerate(contexts):
            len_text.append(len_text[-1] + len(text)+2) # +2: "' "
            # if len_text_answer_idx == -1 and start_loc < len_text[-1]:
            #     len_text_answer_idx = idx - 1
        len_text[-1] -= 2 # context의 마지막 문장이므로 -2

        start, end = 0, len(len_text)-1 # index로 리스트 탐색하므로 -1
        
        while len_text[end] - len_text[start] > 160:
            if (start_loc - len_text[start]) > (len_text[end] - start_loc):
                start += 1
            else:
                end -= 1
        answer_start = start_loc - len_text[start]
        return context[len_text[start]:len_text[end]], answer_start
    
    def __getitem__(self, index):
        return self.question[index], self.context[index], self.answer_start[index], self.answer_text[index]
    
    def __len__(self):
        return len(self.question)


def custom_collate_fn(batch):
    questions, contexts, answer_start, answer_text, afters = [], [], [], [], []

    for _question, _context, _start, _text in batch:
        questions.append(_question)
        contexts.append(_context)
        afters.append(_context[_start:])
        answer_start.append(_start)
        answer_text.append(_text)

    tensorized_input = tokenizer(
        questions, contexts,
        add_special_tokens=True,
        padding="longest",
        max_length=512,
        truncation=True,
        return_tensors='pt'
    )

    after_text = tokenizer(afters, return_tensors=None).input_ids
    answer_tokens = tokenizer(answer_text, return_tensors=None).input_ids

    tensorized_label_ones = np.ones(tensorized_input.input_ids.shape)
    tensorized_label_zero = np.zeros(tensorized_input.input_ids.shape)

    for i, zipped in enumerate(zip(after_text, answer_tokens)):
        text, answer = zipped
        padding_start = (tensorized_input['attention_mask'][i] == 1).nonzero()[-1].item() + 1
        tensorized_label_ones[i, padding_start-len(text): padding_start-len(text)+len(answer)] = 0
        tensorized_label_zero[i, padding_start-len(text): padding_start-len(text)+len(answer)] = 1

    tensorized_label_ones = torch.from_numpy(tensorized_label_ones)
    tensorized_label_zero = torch.from_numpy(tensorized_label_zero)
    tensorized_label = torch.stack([tensorized_label_ones, tensorized_label_zero], dim=2)
    return tensorized_input, tensorized_label


def make_dataloader(dataset, batch_size: int, signature='train') -> DataLoader:
    sampler = RandomSampler(dataset) if signature == 'train' else SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size, sampler = sampler, collate_fn = custom_collate_fn)
    return dataloader


class CustomBertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.labels_type = [0, 1]
        self.num_labels = len(self.labels_type)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_output = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                      nn.GELU(),
                                      nn.Linear(config.hidden_size, 128),
                                      nn.GELU(),
                                      nn.Linear(128, self.num_labels))
        # self.dropout = nn.Dropout(config.dripout_rate)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['last_hidden_state']
        logits = self.qa_output(outputs)
        return logits


def validate(model, valid_dataloader, f1_metric, em_metric, acc_metric):
    criterion = torch.nn.MSELoss()
    model.eval()
    model.to(device)
    
    total_loss, total_em, total_f1, total_acc= 0, 0, 0, 0
        
    for step, batch in enumerate(tqdm(valid_dataloader)):
        batch = tuple(item.to(device) for item in batch)
        batch_input, batch_label = batch
        with torch.no_grad():
            outputs = model(**batch_input)

        loss = criterion(outputs.float(), batch_label.float())
        em_value = 0
        for output, label in zip(outputs, batch_label):
            em = em_metric.compute(predictions=output, references=label)['exact_match']
            em_value += em
        em_value /= len(outputs)
        em = em_value

        f1 = f1_metric.compute(predictions=outputs.view([-1, 1]).squeeze(), references=batch_label.view([-1, 1]).squeeze())['f1']
        acc = acc_metric.compute(predictions=outputs.view([-1, 1]).squeeze(), references=batch_label.view([-1, 1]).squeeze())['accuracy']
        
        total_loss += loss.item()
        total_f1 += f1
        total_em += em
        total_acc += acc

    total_loss = total_loss/(step+1)
    total_em = total_em/(step+1)
    total_f1 = total_f1/(step+1)
    total_acc = total_acc/(step+1)
    return total_loss, total_em, total_f1, total_acc


if __name__ == "__main__":
    # 1. 하이퍼 파라미터 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="data/output/question/")
    parser.add_argument('--model_name', type=str, default="klue/bert-base")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5) # 5e-3
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01) # 4e-5
    parser.add_argument('--seed', type=int, default=521)
    parser.add_argument('--eps', type=float, default=1e-8) # for AdamW
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()
    
    # 2. 랜덤 시드 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    # 3. 토크나이저 로드 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 4. 데이터셋 준비
    dataset = load_dataset("squad_kor_v1") 
    train_dataset = KorQuadDataset(dataset['train'])
    valid_dataset = KorQuadDataset(dataset['validation'])
    train_dataloader = make_dataloader(train_dataset, args.train_batch_size, 'train')
    valid_dataloader = make_dataloader(valid_dataset, args.valid_batch_size, 'valid')
    
    # 5. 학습 모델 준비
    config = BertConfig.from_pretrained(args.model_name)
    config.max_length = args.max_seq_len
    model = CustomBertForQuestionAnswering(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.eps, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_epochs)
    del train_dataset, valid_dataset, dataset
    gc.collect()

    # 6. 학습 & 평가
    ES = EarlyStopping(patience=args.patience)
    writer = SummaryWriter(args.output_dir)
    criterion = torch.nn.MSELoss()
    em_metric = load_metric('exact_match')
    f1_metric = load_metric('f1')
    acc_metric = load_metric('accuracy')

    train_dict = {'loss' : [], 'f1' : []}
    valid_dict = {'loss' : [], 'f1' : [], 'em' : []}

    for epoch in range(1, args.num_epochs+1):
        total_loss, total_acc, batch_acc, total_f1, train_f1, batch_em, total_em, batch_count = 0, 0, 0, 0, 0, 0, 0, 0
        train_loss, valid_loss = 0, 0
        
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader), start=1):
            batch_count += 1
            batch = tuple(item.to(device) for item in batch)
            batch_input, batch_label = batch
            model.zero_grad()
            outputs = model(**batch_input)
            loss = criterion(outputs.float(), batch_label.float())
            writer.add_scalar("loss/train", loss, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm) # prevent exploding gradient.
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            train_loss += loss.item()
            total_loss += loss.item()
            em_value = 0
            for output, label in zip(outputs, batch_label):
                em = em_metric.compute(predictions=output, references=label)['exact_match']
                em_value += em
            em_value /= len(outputs)
            em = em_value

            f1 = f1_metric.compute(predictions=outputs.view([-1, 1]).squeeze(), references=batch_label.view([-1, 1]).squeeze(), average='macro')['f1']
            accuracy = acc_metric.compute(predictions=outputs.view([-1, 1]).squeeze(), references=batch_label.view([-1, 1]).squeeze())['accuracy']

            batch_em += em
            total_em += em

            train_f1 += f1
            total_f1 += f1
            
            batch_acc += accuracy
            total_acc += accuracy
            
            if (step != 0 and step % 100 == 0):
                print(f"epoch: {epoch}, step : {step}, train loss : {train_loss / batch_count:.4f}, f1-score : {train_f1 / batch_count:.4f}")    
                train_loss, train_f1, batch_count = 0,0,0

        print(f"Epoch {epoch} total loss : {total_loss/step:.4f} total f1 : {total_f1/step:.4f}")

        train_dict['f1'].append(total_f1/(step))
        train_dict['loss'].append(total_loss/(step))
        
        if valid_dataloader != None:
            valid_loss, valid_em, valid_f1, valid_acc = validate(model, valid_dataloader, f1_metric, em_metric, acc_metric)
            print(f"Epoch {epoch} valid loss : {valid_loss:.4f} valid f1 : {valid_f1:.4f} valid em : {valid_em:.4f} valid accuracy : {valid_acc:.4f}")

        valid_dict['f1'].append(valid_f1)
        valid_dict['loss'].append(valid_loss)
        valid_dict['em'].append(valid_em)
        
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    # torch.save({
    #         'epoch': epoch,
    #         'model': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'scheduler': scheduler.state_dict(),
    #         'loss' : loss,
    #         'f1' : f1}, 
    #         "test.ckpt")
    # writer.close()
    
    # train_dict, valid_dict