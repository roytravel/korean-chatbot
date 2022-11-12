from torch import Tensor
import torch.nn as nn

class Model(nn.Module):
    """ 테스트용 분류 모델 """
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class Convolution(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, residual: bool) -> None:
        super(Convolution, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=4, kernel_size=1, padding=kernel_size//2)
        self.norm2 = nn.BatchNorm1d(out_channel)
        self.residual = residual
    
    def forward(self, x: Tensor) -> Tensor:
        _x = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return _x + x if self.residual else x

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
