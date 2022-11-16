import os
import platform

pf = platform.system()
_ = "\\" if pf == "Windows" else "/" # _ = '\' or '/'

base_dir = os.path.abspath(os.curdir)

DATA = {
    "INTENT_FILE": base_dir + f'{_}data{_}intent_data.csv',
    "ENTITY_FILE": base_dir + f'{_}data{_}entity_data.csv',
    "MODEL_NAME": "bert-base-multilingual-cased",
    "SPLIT_RATIO": 0.8,
    "BATCH_SIZE": 256,
    "MAX_LENGTH": 32,
    
    "OUTPUT_DIR": f"data{_}output{_}",
    "CONFIG": f"{_}config.json",
    "MODEL_FILE_NAME": f"{_}pytorch_model.bin"
    
}