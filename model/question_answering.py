import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

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