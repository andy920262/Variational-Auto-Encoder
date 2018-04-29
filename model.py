import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()

    def forward(self, inputs):
        raise NotImplementedError

class BaseDecoder(nn.Module):
    def __init__(self):
        super(BaseDecoder, self).__init__()

    def forward(self, inputs):
        raise NotImplementedError

class Conv2d96x96Encoder(BaseEncoder):
    def __init__(self, encode_size):
        super(Conv2d96x96Encoder, self).__init__()
        self.encode_size = encode_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU())
        self.linear = nn.Linear(128 * 3 * 3, 512)
        self.mu = nn.Linear(512, encode_size)
        self.logvar = nn.Linear(512, encode_size)

    def forward(self, inputs):
        x = self.conv_layers(inputs).view(-1, 128 * 3 * 3)
        x = self.linear(x)
        return self.mu(x), self.logvar(x)

class Conv2d96x96Decoder(BaseDecoder):
    def __init__(self, encode_size):
        super(Conv2d96x96Decoder, self).__init__()
        self.encode_size = encode_size
        self.linear = nn.Linear(encode_size, 128 * 3 * 3)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid())

    def forward(self, inputs):
        x = self.linear(inputs).view(-1, 128, 3, 3)
        x = self.conv_layers(x)
        return x

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, inputs):
        mu, logvar = self.encoder(inputs)
        std = torch.exp(0.5 * logvar)
        # PyTorch 0.4.0
        # eps = torch.randn_like(std)
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_()).cuda()
        reconstruct = self.decoder(eps * std + mu)
        return reconstruct, mu, logvar
    
    def loss(self, inputs, reconstruct, mu, logvar):
        rec_err = F.mse_loss(reconstruct, inputs, size_average=False)
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
        return rec_err + kld
