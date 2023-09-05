import torch
from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class RandomClassModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x, z):
        batch_size = x.size(0)
        probabilities = torch.rand(batch_size, self.num_classes)
        return probabilities
