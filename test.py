import torch.nn as nn
import numpy as np
import torch
import os
from datasets import ImageNet, ImageNet_semi, get_dataset
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from libs.autoencoder import get_model
import argparse
from tqdm import tqdm
import pickle
torch.manual_seed(0)
np.random.seed(0)

dataset = dset.ImageFolder(root='/mnt/datasets/ImageNet/train', transform=None)
idx_to_class = {}

for i, (k, v) in enumerate(dataset.class_to_idx.items()):
    if i != v:
        print('errorerrorerror')
    idx_to_class[v] = k

with open('idx_to_class.pkl', 'wb') as f:
    pickle.dump(idx_to_class, f)

with open('idx_to_class.pkl', 'rb') as f:
    test_1 = pickle.load(f)

print(type(test_1))
print(test_1)

