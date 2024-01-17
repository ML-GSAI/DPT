import torch.nn as nn
import numpy as np
import torch
import os
from datasets import ImageNet
from torch.utils.data import DataLoader
from libs.autoencoder import get_model
import argparse
from tqdm import tqdm
torch.manual_seed(0)
np.random.seed(0)

def mprint(*args):
    print('\n-----------------------------')
    print(*args)
    print('-----------------------------\n')

from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])

def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams

def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    cluster_name = config.model_name + '-' + '-'.join(config.subset_path.split('/')).split('.txt')[0]
    cluster_path = f'pretrained/cluster/{cluster_name}/imagenet_features_preds.npy'
    fnames_path = f'pretrained/cluster/{cluster_name}/imagenet_features_fnames.pth'
    autoencoder_path = config.autoencoder.pretrained_path
    path = config.image_path

    dataset = ImageNet(path=path, resolution=config.resolution, random_flip=False, cluster_path=cluster_path, fnames_path=fnames_path)

    train_dataset = dataset.get_split(split='train', labeled=True)

    train_batch_size = 128
    if config.resolution == 512:
        train_batch_size = 64

    train_dataset_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, drop_last=False,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    model = get_model(autoencoder_path)
    model = nn.DataParallel(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    save_path = f'pretrained/datasets/{cluster_name}'
    save_features_path = os.path.join(save_path, f'imagenet{config.resolution}_features')
    os.system(f'mkdir -p {save_features_path}')

    idx = 0
    for batch in tqdm(train_dataset_loader):
        img, label = batch
        img = torch.cat([img, img.flip(dims=[-1])], dim=0)
        img = img.to(device)
        moments = model(img, fn='encode_moments')
        moments = moments.detach().cpu().numpy()

        label = torch.cat([label, label], dim=0)
        label = label.detach().cpu().numpy()

        for moment, lb in zip(moments, label):
            np.save(os.path.join(save_features_path, f'{idx}.npy'), (moment, lb))
            idx += 1

    mprint(f'save {idx} files')

if __name__ == "__main__":
    app.run(main)
