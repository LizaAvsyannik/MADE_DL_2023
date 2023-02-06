import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, n_layers, input_ch, output_ch, kernel_size, activation=nn.ReLU, **kwargs):
        super().__init__()
        kwargs.setdefault('padding', 'same')

        conv_activ = lambda **conv_kwargs: nn.Sequential(nn.Conv2d(**conv_kwargs), activation())

        layers = [conv_activ(in_channels=input_ch, out_channels=output_ch, kernel_size=kernel_size, **kwargs)]
        repeated_convs_kwargs = {'in_channels': output_ch, 'out_channels': output_ch, 'kernel_size': kernel_size}
        repeated_convs_kwargs.update(kwargs)
        for _ in range(n_layers):
            layers.append(conv_activ(**repeated_convs_kwargs))
        self._block = nn.Sequential(*layers)

    def forward(self, x):
        return self._block(x)


class OCRModel(nn.Module):
    def __init__(self, n_output_classes):
        super().__init__()
        self._pooling = nn.MaxPool2d
        self._conv = nn.Sequential(
            ConvBlock(3, 3, 16, 5),
            self._pooling(2),
            ConvBlock(3, 16, 32, 3),
            self._pooling(2),
            nn.BatchNorm2d(32),
            ConvBlock(3, 32, 128, 3),
            self._pooling(2),
            nn.BatchNorm2d(128),
            ConvBlock(3, 128, 128, 3),
            self._pooling(2),
            nn.BatchNorm2d(128),
        )

        self.lstm = nn.LSTM(384, 128, 2, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm([12, 128]),
            nn.Linear(128, n_output_classes))

    def forward(self, x: torch.Tensor):
        conv_output = self._conv(x)
        conv_output = torch.flatten(conv_output, start_dim=1, end_dim=2)
        conv_output = conv_output.permute(0, 2, 1)  # (N, W, C * H)
        lstm_output, _ = self.lstm(conv_output)  # (N, W, encoded)
        fc_output = self.fc(lstm_output).permute(1, 0, 2)  # (W, N, encoded)
        return fc_output.log_softmax(-1)
