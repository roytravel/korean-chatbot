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
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, residual: bool, num_classes: int) -> None:
        super(Convolution, self).__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        pass
    
    
class LSTM(nn.Module):
    def __init__(self) -> None:
        super(LSTM, self).__init__()
    
    def forawrd(self, x: Tensor) -> Tensor:
        pass


class RNN(nn.Module):
    def __init__(self) -> None:
        super(RNN, self).__init__()
        
    def forward(self, x: Tensor) -> Tensor:
        pass