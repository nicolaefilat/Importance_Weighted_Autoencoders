# The VAE or IWAE network architectures

import torch
import torch.nn as nn


# VAE or IWAE with one stochastic layer
class VAE_1(nn.Module):

    def __init__(self, args):
        super(VAE_1, self).__init__()
        self.k = args.k
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3_mu = nn.Linear(200, 50)
        self.fc3_sigma = nn.Linear(200, 50)
        self.fc4 = nn.Linear(50, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 28 * 28)

    def forward(self, x):
        x = x.view(-1, 1, 28 * 28)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3_mu(x)
        log_sigma = self.fc3_sigma(x)
        eps = torch.randn_like(mu.repeat(1, self.k, 1))
        x = mu + torch.exp(log_sigma) * eps
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = self.fc6(x)
        return x, mu, log_sigma, eps


# VAE or IWAE with two stochastic layers
class VAE_2(nn.Module):

    def __init__(self, args):
        super(VAE_2, self).__init__()
        self.k = args.k
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3_mu = nn.Linear(200, 100)
        self.fc3_sigma = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6_mu = nn.Linear(100, 50)
        self.fc6_sigma = nn.Linear(100, 50)
        self.fc7 = nn.Linear(50, 100)
        self.fc8 = nn.Linear(100, 100)
        self.fc9_mu = nn.Linear(100, 100)
        self.fc9_sigma = nn.Linear(100, 100)
        self.fc10 = nn.Linear(100, 200)
        self.fc11 = nn.Linear(200, 200)
        self.fc12 = nn.Linear(200, 28 * 28)

    def forward(self, x):
        x = x.view(-1, 1, 28 * 28)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu1 = self.fc3_mu(x)
        log_sigma1 = self.fc3_sigma(x)
        eps1 = torch.randn_like(mu1.repeat(1, self.k, 1))
        h1 = mu1 + torch.exp(log_sigma1) * eps1
        x = torch.tanh(self.fc4(h1))
        x = torch.tanh(self.fc5(x))
        mu2 = self.fc6_mu(x)
        log_sigma2 = self.fc6_sigma(x)
        eps2 = torch.randn_like(mu2)
        h2 = mu2 + torch.exp(log_sigma2) * eps2
        x = torch.tanh(self.fc7(h2))
        x = torch.tanh(self.fc8(x))
        mu3 = self.fc9_mu(x)
        log_sigma3 = self.fc9_sigma(x)
        x = torch.tanh(self.fc10(h1))
        x = torch.tanh(self.fc11(x))
        x = self.fc12(x)
        return x, mu1, mu2, mu3, log_sigma1, log_sigma2, log_sigma3, eps1, eps2


