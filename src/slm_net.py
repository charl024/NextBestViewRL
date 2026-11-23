""" Custom neural network for SLM-Lab. CNN used to process images, and a general MLP for rest of data observation """
import torch
from torch import nn
from slm_lab.agent.net.mlp import MLPNet

from slm_lab.agent.net import net_util

class DepthNet(nn.Module):
    def __init__(self, k, img, n_vec, out_dim):
        super().__init__()
        self.k = k
        self.img = img
        self.n_img = k * img * img

        self.cnn = nn.Sequential(
            nn.Conv2d(k, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            n_cnn = self.cnn(torch.zeros(1, k, img, img)).shape[1]
        
        self.mlp = nn.Sequential(nn.Linear(n_vec, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(n_cnn + 64, out_dim), nn.ReLU())

    def forward(self, x):
        depth = x[:, :self.n_img].reshape(-1, self.k, self.img, self.img)
        vec = x[:, self.n_img:]
        return self.head(torch.cat([self.cnn(depth), self.mlp(vec)], dim=1))

class NBVConvNet(MLPNet):
    def __init__(self, net_spec, in_dim, out_dim):
        super().__init__(net_spec, in_dim, out_dim)
        k, img = net_spec["depth_k"], net_spec["depth_img"]

        # set MLPNet model to use custom DepthNet
        self.model = DepthNet(k, img, in_dim - k * img * img, self.hid_layers[-1])
        net_util.init_layers(self, self.init_fn)
        self.to(self.device) 