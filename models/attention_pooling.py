import torch
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False))

    def forward(self, sequence):
        weights = self.score(sequence).softmax(dim=1)
        return (sequence * weights).sum(dim=1)


class MeanPooling(nn.Module):
    def forward(self, sequence):
        return sequence.mean(dim=1)


def make_pooling(dim, config):
    if config["pooling"] == "attention":
        return AttentionPooling(dim, config["pooling_hidden_dim"])
    if config["pooling"] == "mean":
        return MeanPooling()
    raise ValueError("Unknown pooling")
