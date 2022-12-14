import os
import platform

pf = platform.system()
_ = "\\" if pf == "Windows" else "/" # _ = '\' or '/'

base_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
HYPERPARAMETER = {
    "INTENT_FILE": base_dir + f'{_}data{_}intent_data.csv',
    "ENTITY_FILE": base_dir + f'{_}data{_}entity_data.csv',
    "QA_FILE": base_dir + f"{_}data{_}KorQuAD_v1.0_train",
    "SUMMARY_TRAIN_DIR": base_dir + f"{_}data{_}summary{_}training{_}*",
    "SUMMARY_VALID_DIR": base_dir + f"{_}data{_}summary{_}validation{_}*",
    "MODEL_NAME": "bert-base-multilingual-cased",
    "SPLIT_RATIO": 0.8,
    "BATCH_SIZE": 256,
    "MAX_LENGTH": 32,
    "SUMMARY_MAX_SEQ_LEN": 512,
    "QA_MAX_SEQ_LEN": 512,
    
    "INTENT_OUTPUT_DIR": base_dir + f"{_}data{_}output{_}intent{_}",
    "ENTITY_OUTPUT_DIR": base_dir + f"{_}data{_}output{_}entity{_}",
    "QUEST_OUTPUT_DIR": base_dir + f"{_}data{_}output{_}question_good_save{_}",
    "SUMMARY_OUTPUT_DIR": base_dir + f"{_}data{_}output{_}summary{_}",
    "CONFIG": "config.json",
    "MODEL_FILE_NAME": f"{_}pytorch_model.bin",
    "DOMAIN_FILENAME": base_dir + f"{_}config{_}domain.json",
    "IP_ADDR": "127.0.0.1",
    "PORT": 3000
}