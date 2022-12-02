import torch
import torch.nn as nn
from transformers import BertModel
from utils.decorators import hyperparameter

class SimpleClassifier(nn.Module):
    def __init__(self, hidden_size) -> None:
        super(SimpleClassifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sentence_scores = self.sigmoid(h) * mask_cls.float()
        return sentence_scores


@hyperparameter
class BertForExtSum(nn.Module):
    def __init__(self):
        super(BertForExtSum, self).__init__()
        self.bert = BertModel.from_pretrained(self.MODEL_NAME)
        self.encoder = SimpleClassifier(self.bert.config.hidden_size)

    def forward(self, src, segments, clss):
        """
        Args:
            src: input_ids
            segments: segment embedding
            clss: position embedding
        """
        mask_src = ~(src == 0)
        mask_cls = ~(clss == -1)
        
        top_vector = self.bert(input_ids=src, token_type_ids=segments, attention_mask=mask_src)
        top_vector = top_vector.last_hidden_state
        
        sentences_vector = top_vector[torch.arange(top_vector.size(0)).unsqueeze(1), clss]
        sentences_vector = sentences_vector * mask_cls[:, :, None].float()
        
        sentences_scores = self.encoder(sentences_vector, mask_cls).squeeze(-1)
        return sentences_scores, mask_cls