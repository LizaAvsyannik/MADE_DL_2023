import torch
from torch import nn


# class ConvolutionalBlock(nn.Module):
#     def __init__(self):
#         super(ConvolutionalBlock, self).__init__()

class OCRModel(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.conv8 = nn.Sequential(nn.Conv2d(3, 8, (1, 3)),
                                  nn.ReLU(),
                                  nn.Conv2d(8, 8, (1, 3)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((1, 3)))
        
        self.conv16 = nn.Sequential(nn.Conv2d(8, 16, (3, 5)),
                                  nn.ReLU(),
                                  nn.Conv2d(16, 16, (3, 5)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((3, 5)))

        self.lstm = nn.LSTM(240, 64, 2, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(128, 20))
    
    def forward(self, x):
        # print(x.shape)
        conv_output = self.conv16(self.conv8(x))
        # print(conv_output.shape)
        conv_output = conv_output.reshape(conv_output.shape[0], conv_output.shape[1] * conv_output.shape[2], conv_output.shape[3])
        # print(conv_output.shape)
        conv_output = conv_output.permute(0, 2, 1)
        # print(conv_output.shape)
        lstm_output, _ = self.lstm(conv_output)
        return self.fc(lstm_output)