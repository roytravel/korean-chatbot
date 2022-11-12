from torch import Tensor
import torch.nn as nn

class Model(nn.Module):
    """ 테스트용 분류 모델 """
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 3)
        self.relu1 = nn.ReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

class Convolution(nn.Module):
    def __init__(self) -> None:
        super(Convolution, self).__init__()
    
    def forward(self):
        pass

class LSTM(nn.Module):
    def __init__(self) -> None:
        super(LSTM, self).__init__()
    
    def forawrd(self):
        pass

class RNN(nn.Module):
    def __init__(self) -> None:
        super(RNN, self).__init__()
    
    def forward(self):
        pass

class KoBERT(nn.Module):
    def __init__(self) -> None:
        super(KoBERT, self).__init__()

    def forward(self):
        pass
