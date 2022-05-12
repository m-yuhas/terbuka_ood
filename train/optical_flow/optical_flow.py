"""Module for training and functional test of optical flow based OOD detector.
"""


from typing import Tuple


import numpy
import torch


class Encoder(torch.nn.Module):
    """Encoder network."""

    def __init__(self, dimensions: Tuple[int], n_latent: int, lam: float) -> None:
        super(Encoder, self).__init__()
        self.n_latent = n_latent
        self.dimensions = dimensions

        self.conv1 = torch.nn.Conv2d(self.dimensions[2], 32, (5, 5), stride=(3,3), padding=(0, 0), bias=False)
        self.conv1_bn = torch.nn.BatchNorm2d(32)


class AutoEncoder(torch.nn.Module):
    """A single autoencoder network for the detector."""

    def __init__(self) -> None:
        super(AutoEncoder, self).__init__()
        
