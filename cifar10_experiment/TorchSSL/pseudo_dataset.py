import numpy as np
import torchvision
import torch
import os
import pickle
import sys
import pickle
import numpy as np
from torch.utils.data import DataLoader
# 随机打乱targets   
import random 
import tarfile

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/cifar-10-batches-py')
    parser.add_argument('--save_path', type=str, default='./saved_models/freematch_cifar10_40_1')

    args = parser.parse_args()

    labels = [] # save labels
    data_dict = {} # data_1, data_2, data_3, data_4, data_5

    for i in range(5):
        batch = unpickle(os.path.join(args.data_path, 'data_batch_{}'.format(i+1)))
        data_dict['data_{}'.format(i+1)] = batch['data']
        labels.extend(batch['labels'])
    
    pseudo_path = os.path.join(args.save_path, 'pseudo_label.npy')
    pseudo_labels = np.load(pseudo_path)

    # print true labels distribution and pseudo labels distribution
    true_distribution = torch.bincount(torch.tensor(labels))
    pseudo_distribution =  torch.bincount(torch.tensor(pseudo_labels))
    print("true distribution:", true_distribution)
    print("pseudo distribution:", pseudo_distribution)
    # print accuracy
    print(np.sum(labels == pseudo_labels) / 50000)

    for i in range(5):
        data = data_dict['data_{}'.format(i+1)]
        targets = pseudo_labels[i*10000:(i+1)*10000]
        batch_path = os.path.join(args.save_path, 'cifar-10-batches-py/data_batch_{}'.format(i+1))
        os.makedirs(os.path.dirname(batch_path), exist_ok=True)
        with open(batch_path, 'wb') as f:
            pickle.dump({'data': data, 'labels': targets}, f)
    
    with tarfile.open(os.path.join(args.save_path, 'cifar-10-python.tar.gz'), 'w:gz') as f:
        f.add(os.path.join(args.save_path, 'cifar-10-batches-py'), arcname='cifar-10-batches-py')

    
