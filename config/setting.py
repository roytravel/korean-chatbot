import os
import platform

pf = platform.system()
_ = "\\" if pf == "Windows" else "/" # _ = '\' or '/'

base_dir = os.path.abspath(os.curdir)

DATA = {
    "INTENT_DIR": base_dir + f'{_}data{_}intent_data.csv',
    "ENTITY_DIR": base_dir + f'{_}data{_}entity_data.csv',
    "MODEL_NAME": "bert-base-multilingual-cased",
    "SPLIT_RATIO": 0.8
}