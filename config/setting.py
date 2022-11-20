import os
import platform

pf = platform.system()
_ = "\\" if pf == "Windows" else "/" # _ = '\' or '/'

base_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATA = {
    "INTENT_FILE": base_dir + f'{_}data{_}intent_data.csv',
    "ENTITY_FILE": base_dir + f'{_}data{_}entity_data.csv',
    "MODEL_NAME": "bert-base-multilingual-cased",
    "SPLIT_RATIO": 0.8,
    "BATCH_SIZE": 256,
    "MAX_LENGTH": 32,
    
    "INTENT_OUTPUT_DIR": base_dir + f"{_}data{_}output{_}intent{_}",
    "ENTITY_OUTPUT_DIR": base_dir + f"{_}data{_}output{_}entity{_}",
    "CONFIG": "config.json",
    "MODEL_FILE_NAME": f"{_}pytorch_model.bin",
    "DOMAIN_FILENAME": base_dir + f"{_}config{_}domain.json",
}