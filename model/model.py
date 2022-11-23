from torch import Tensor
import torch.nn as nn
from transformers import BertForSequenceClassification, BertPreTrainedModel, BertModel

class BertForSequenceClassification(BertForSequenceClassification):
    """ BertForSequenceClassification Wrapper 클래스 """
    def __init__(self, config):
        super().__init__(config)

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


class Model(nn.Module):
    """ 테스트용 분류 모델 """
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class Convolution(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, residual: bool, num_classes: int) -> None:
        super(Convolution, self).__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        pass
    
    
class LSTM(nn.Module):
    def __init__(self) -> None:
        super(LSTM, self).__init__()
    
    def forawrd(self, x: Tensor) -> Tensor:
        pass


class RNN(nn.Module):
    def __init__(self) -> None:
        super(RNN, self).__init__()
        
    def forward(self, x: Tensor) -> Tensor:
        pass