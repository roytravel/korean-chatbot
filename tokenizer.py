import os
import torch
import sentencepiece as spm

INPUT_FILE_PATH         = "ratings_train.txt"
VOCAB_SIZE              = 32000
SP_MODEL_PATH           = "./data/tokenizer"
USER_DEFINED_SYMBOLS    = '[PAD],[UNK],[CLS],[SEP],[MASK],[BOS],[EOS],[UNK0],[UNK1],[UNK2],[UNK3],[UNK4],[UNK5],[UNK6],[UNK7],[UNK8],[UNK9],[unused0],[unused1],[unused2],[unused3],[unused4],[unused5],[unused6],[unused7],[unused8],[unused9],[unused10],[unused11],[unused12],[unused13],[unused14],[unused15],[unused16],[unused17],[unused18],[unused19],[unused20],[unused21],[unused22],[unused23],[unused24],[unused25],[unused26],[unused27],[unused28],[unused29],[unused30],[unused31],[unused32],[unused33],[unused34],[unused35],[unused36],[unused37],[unused38],[unused39],[unused40],[unused41],[unused42],[unused43],[unused44],[unused45],[unused46],[unused47],[unused48],[unused49],[unused50],[unused51],[unused52],[unused53],[unused54],[unused55],[unused56],[unused57],[unused58],[unused59],[unused60],[unused61],[unused62],[unused63],[unused64],[unused65],[unused66],[unused67],[unused68],[unused69],[unused70],[unused71],[unused72],[unused73],[unused74],[unused75],[unused76],[unused77],[unused78],[unused79],[unused80],[unused81],[unused82],[unused83],[unused84],[unused85],[unused86],[unused87],[unused88],[unused89],[unused90],[unused91],[unused92],[unused93],[unused94],[unused95],[unused96],[unused97],[unused98],[unused99]'
MODEL_TYPE              = "bpe" # unigram, bpe, ...
SP_MODEL_NAME           = f"tokenizer_{VOCAB_SIZE}"
SP_MODEL_PATH           = os.path.join(SP_MODEL_PATH, SP_MODEL_NAME)
CHARACTER_COVERAGE      = 1.0
MAX_SEQ_LEN = 128

class SentencePiece:
    def __init__(self) -> None:
        pass       

    def create_vocab_by_spm(self):
        if not os.path.isdir(SP_MODEL_PATH):
            os.makedirs(SP_MODEL_PATH)

        parameter = '--input={} --model_prefix={} --vocab_size={} --user_defined_symbols={} --model_type={} --character_coverage={}'
        cmd = parameter.format(INPUT_FILE_PATH, SP_MODEL_PATH, VOCAB_SIZE, USER_DEFINED_SYMBOLS, MODEL_TYPE, CHARACTER_COVERAGE)
        spm.SentencePieceTrainer.Train(cmd)
    
    def _test_spm_tokenizer(self) -> None:
        sp = spm.SentencePieceProcessor()
        sp.Load('{}.model'.format(SP_MODEL_PATH))
        text = "나는 오늘 아침밥을 먹었다."
        token = sp.EncodeAsPieces(text)
        ids = sp.EncodeAsIds(text)
        print("Tokens : {}".format(token))
        print("Tokens : {}".format(ids))


class Tokenizer:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load('{}.model'.format(SP_MODEL_PATH))

    def tokenize(self, text):
        # token = self.sp.EncodeAsPieces(text)
        ids = self.sp.EncodeAsIds(text)
        # print("Tokens : {}".format(token))
        # print("Tokens : {}".format(ids))
        ids = self._pad_sequence(ids)
        ids = torch.Tensor(ids)
        return ids

    def _pad_sequence(self, sequence: list) -> list:
        while len(sequence) < MAX_SEQ_LEN:
            sequence.append(0)
        return sequence


# if __name__ == "__main__":
#     SP = SentencePiece()
#     SP.create_vocab_by_spm()
#     SP._test_spm_tokenizer()