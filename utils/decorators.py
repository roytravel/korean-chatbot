from config import setting

def hyperparameter(cls):
    for key, value in setting.HYPERPARAMETER.items():
        setattr(cls, key, value)
    return cls