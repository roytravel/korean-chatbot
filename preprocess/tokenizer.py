from transformers import AutoTokenizer

class AutoTokenizer(AutoTokenizer):
    """ AutoTokenizer Wrapper 클래스 """
    def __init__(self):
        super().__init__()