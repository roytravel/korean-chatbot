from transformers import AutoTokenizer, BertTokenizer

class AutoTokenizer(AutoTokenizer):
    """ AutoTokenizer Wrapper 클래스 """
    def __init__(self):
        super().__init__()

class BertTokenizer(BertTokenizer):
    """ BertTokenizer Wrapper 클래스 """
    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, never_split=None, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]", tokenize_chinese_chars=True, **kwargs):
        super().__init__(vocab_file, do_lower_case, do_basic_tokenize, never_split, unk_token, sep_token, pad_token, cls_token, mask_token, tokenize_chinese_chars, **kwargs)