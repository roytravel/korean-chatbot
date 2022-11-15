from config import setting

def data(cls):
    for key, value in setting.DATA.items():
        setattr(cls, key, value)
    return cls