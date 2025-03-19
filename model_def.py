import torch
import torch.nn as nn

class Convautoenc(nn.Module):
    def __init__(self, out_channels=32, kernel_size=2048, stride=512):
        super(Convautoenc, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.synconv1 = nn.ConvTranspose1d(in_channels=out_channels, out_channels=1, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

    def encoder(self, x):
        x = self.conv1(x)
        y = torch.tanh(x)
        return y

    def decoder(self, y):
        xrek = self.synconv1(y)
        return xrek

    def forward(self, x):
        y = self.encoder(x)
        xrek = self.decoder(y)
        return xrek
