import ml_collections
import torch
import torch.nn as nn
from sklearn.utils.validation import check_is_fitted
from torch import multiprocessing as mp
from torchvision.utils import make_grid, save_image
from torch.utils._pytree import tree_map
import accelerate
import time
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import tempfile
from absl import logging
import builtins
import os
import argparse
import pprint
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from cyanure.data_processing import preprocess
from cyanure.estimators import Classifier
import pickle
import cyanure as cyan
import src.deit as deit
from src.data_manager import (
    init_data,
)
import utils
from datasets import ImageNet, ImageNet_semi, get_dataset
from torch.utils.data import DataLoader
from libs.autoencoder import get_model

def mprint(*args):
    print('\n-----------------------------')
    print(*args)
    print('-----------------------------\n')

def get_all_features(config):
    torch.backends.cudnn.benchmark = True
    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.pretrained = os.path.join(config.pretrained, config.fname)
    config = ml_collections.FrozenConfigDict(config)

    # -- Function to make train/test dataloader
    def init_pipe_all(training):
        # -- make data transforms
        transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))])
        # -- init data-loaders/samplers
        subset_file = None # remove subset_file 
        data_loader, _ = init_data(
            transform=transform,
            batch_size=16,
            num_workers=4,
            world_size=1,
            rank=0,
            root_path=config.root_path,
            image_folder=config.image_folder,
            training=training,
            copy_data=False,
            drop_last=False,
            subset_file=subset_file)
        return data_loader

    # -- Initialize the model
    encoder = init_model(
        pretrained=config.pretrained,
        model_name=config.model_name)
    encoder.eval()

    # -- Initialize the data-pipeline
    data_loader = init_pipe_all(training=True)

    # -- accelerate 
    encoder, data_loader= accelerator.prepare(encoder, data_loader)
    embs, labs, fnames = make_embeddings_fnames(
        accelerator=accelerator,
        blocks=config.blocks,
        mask_frac=config.mask,
        data_loader=data_loader,
        encoder=encoder)
    features_path = os.path.join('pretrained/features', f'{config.model_name}')
    os.makedirs(name=features_path, exist_ok=True)

    if accelerator.is_main_process:
        train_embs_path = os.path.join(features_path, f'features-label-fnames.pth.tar')
        torch.save({
            'embs': embs,
            'labs': labs,
            'fnames': fnames
        }, train_embs_path)
        logging.info(f'saved all training embs of shape {embs.shape}')

    
    accelerator.wait_for_everyone()



def get_aug_features(config):
    torch.backends.cudnn.benchmark = True
    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.pretrained = os.path.join(config.pretrained, config.fname)
    config = ml_collections.FrozenConfigDict(config)

    cluster_name = config.model_name + '-' + '-'.join(config.subset_path.split('/')).split('.txt')[0]

    aug_root_path = f'{config.dpm_path}/{cluster_name}/samples_for_classifier/'
    aug_image_folder = f'aug_{config.augmentation_K}_samples/'

    # -- Function to make train/test dataloader
    def init_pipe(training):
        # -- make data transforms
        transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))])
        # -- init data-loaders/samplers
        subset_file = None
        data_loader, _ = init_data(
            transform=transform,
            batch_size=16,
            num_workers=4,
            world_size=1,
            rank=0,
            root_path=aug_root_path,
            image_folder=aug_image_folder,
            training=training,
            copy_data=False,
            drop_last=False,
            subset_file=subset_file)
        return data_loader

    # -- Initialize the model
    encoder = init_model(
        pretrained=config.pretrained,
        model_name=config.model_name)
    encoder.eval()

    # -- Initialize the data-pipeline
    data_loader = init_pipe(training=True)

    # -- accelerate 
    encoder, data_loader= accelerator.prepare(encoder, data_loader)
    aug_embs, aug_labs = make_embeddings(
        accelerator=accelerator,
        blocks=config.blocks,
        mask_frac=config.mask,
        data_loader=data_loader,
        encoder=encoder)

    features_path = os.path.join('pretrained/features', f'{cluster_name}')
    os.makedirs(name=features_path, exist_ok=True)

    if accelerator.is_main_process:
        train_aug_embs_path = os.path.join(features_path, f'aug-{config.augmentation_K}-features-label.pth.tar')
        torch.save({
            'embs': aug_embs,
            'labs': aug_labs,
        }, train_aug_embs_path)
        logging.info(f'saved train augment embs of shape {aug_embs.shape}')

    accelerator.wait_for_everyone()

def get_cluster(config):
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    features_path = os.path.join('pretrained/features', f'{config.model_name}')
    train_embs_path = os.path.join(features_path, f'features-label-fnames.pth.tar')

    cluster_name = config.model_name + '-' + '-'.join(config.subset_path.split('/')).split('.txt')[0]
    
    if accelerator.is_main_process:
        checkpoint = torch.load(train_embs_path, map_location='cpu')
        embs, labs, fnames = checkpoint['embs'], checkpoint['labs'], checkpoint['fnames']
        print(f'loaded train embs of shape {embs.shape}')
        subset_tag = '-'.join(config.subset_path.split('/')).split('.txt')[0] + '-accelerate' if config.subset_path is not None else 'imagenet_subses1-100percent'
        classifier = pickle.load(open(os.path.join('pretrained', f'classifier-{subset_tag}-{config.fname}.pkl'),'rb'))
        os.makedirs(f'pretrained/cluster/{cluster_name}', exist_ok=True)
        p_ms = classifier.decision_function(embs.numpy())
        print(p_ms.shape)
        p_ms = torch.tensor(p_ms)
        softmax = nn.Softmax(dim=1)
        p_ms = softmax(p_ms)
        probs, preds = torch.max(p_ms, dim=1)
        probs, preds = probs.numpy(), preds.numpy()
        print(np.sum(preds == labs.numpy())/len(labs))
        torch.save(torch.tensor(probs), f'pretrained/cluster/{cluster_name}/imagenet_features_probs.pth')
        np.save(f'pretrained/cluster/{cluster_name}/imagenet_features_preds.npy', torch.tensor(preds))
        torch.save(torch.tensor(preds), f'pretrained/cluster/{cluster_name}/imagenet_features_preds.pth')
        torch.save(labs, f'pretrained/cluster/{cluster_name}/imagenet_features_labels.pth') 
        torch.save(fnames, f'pretrained/cluster/{cluster_name}/imagenet_features_fnames.pth')
    
    accelerator.wait_for_everyone()


def train_classifier_stage3(config):
    torch.backends.cudnn.benchmark = True
    os.makedirs(name=os.path.dirname(config.output_path), exist_ok=True) # create output log path
    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    lambd = config.lambd
    config = ml_collections.FrozenConfigDict(config)
    
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=config.output_path)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    logging.info(f'config:{config}')

    # -- Define file names used to save computed embeddings (for efficient
    # -- reuse if running the script more than once)
    subset_tag = '-'.join(config.subset_path.split('/')).split('.txt')[0] + '-accelerate' if config.subset_path is not None else 'imagenet_subses1-100percent'
    cluster_name = config.model_name + '-' + '-'.join(config.subset_path.split('/')).split('.txt')[0]

    features_path = os.path.join('pretrained/features', f'{cluster_name}')
    os.makedirs(name=features_path, exist_ok=True)
    

    train_embs_aug_path = os.path.join(features_path, f'aug-{config.augmentation_K}-features-label.pth.tar')
    logging.info(f'train_embs_aug_path:{train_embs_aug_path}')
    assert os.path.exists(train_embs_aug_path) == True

    train_embs_path = os.path.join(config.pretrained, 'accelerate', f'train-features-fnames-{subset_tag}-{config.fname}')
    test_embs_path = os.path.join(config.pretrained, 'accelerate', f'val-features-fnames-{config.fname}')
    logging.info(f'test_embs_path:{test_embs_path}')
    assert os.path.exists(test_embs_path) == True

    train_aug_checkpoint = torch.load(train_embs_aug_path, map_location='cpu')
    embs, labs = train_aug_checkpoint['embs'], train_aug_checkpoint['labs']
    logging.info(f'loaded train_aug_embs of shape {embs.shape}')

    # -- Compute the embeddings
    if os.path.exists(train_embs_path) and config.using_true_label:
        train_checkpoint = torch.load(train_embs_path, map_location='cpu')
        train_embs, train_labs, train_fnames = train_checkpoint['embs'], train_checkpoint['labs'], train_checkpoint['fnames']
        logging.info(f'loaded train_true_embs of shape {train_embs.shape}')
        logging.info('add true embs')
        logging.info(f'train_embs_path:{train_embs_path}')
        embs = torch.cat((embs, train_embs), dim=0)
        labs = torch.cat((labs, train_labs), dim=0)
        logging.info(f'loaded train_aug_add_true_embs_path of shape {embs.shape}')

    if accelerator.is_main_process:
        # -- Normalize embeddings
        preprocess(embs, normalize=config.normalize, columns=False, centering=True)
        # -- Fit Logistic Regression Classifier
        lambd /= len(embs)
        classifier = Classifier(loss='multiclass-logistic', penalty=config.penalty, fit_intercept=False,
            lambda_1=lambd,
            lambda_2=lambd,
            tol=1e-3,
            solver='auto',
            max_iter=100)
        
        classifier.fit(
            embs.numpy(),
            labs.numpy()
            )
        # -- Evaluate and log
        train_score = classifier.score(embs.numpy(), labs.numpy())
        # -- (save train score)
        logging.info(f'train score: {train_score}')

        pickle.dump(classifier, open(os.path.join('pretrained', f'classifier-{subset_tag}-{config.fname}-aug-{config.augmentation_K}.pkl'),'wb'))
    
    # -- If test embeddings already computed, load file, otherwise, compute
    # -- embeddings and save
    
    # -- Compute the test embeddings
    test_checkpoint = torch.load(test_embs_path, map_location='cpu')
    test_embs, test_labs = test_checkpoint['embs'], test_checkpoint['labs']
    logging.info(f'loaded test embs of shape {test_embs.shape}')

    if accelerator.is_main_process:
        # -- Normalize embeddings
        preprocess(test_embs, normalize=config.normalize, columns=False, centering=True)

        # -- Evaluate and log
        test_score = classifier.score(test_embs.numpy(), test_labs.numpy())
        # -- (save test score)
        logging.info(f'test score: {test_score}\n\n')
        
        return test_score

def train_classifier_stage1(config):
    torch.backends.cudnn.benchmark = True
    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=config.output_path)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    # -- Define file names used to save computed embeddings (for efficient
    # -- reuse if running the script more than once)
    subset_tag = '-'.join(config.subset_path.split('/')).split('.txt')[0] + '-accelerate' if config.subset_path is not None else 'imagenet_subses1-100percent'
    train_embs_path = os.path.join(config.pretrained, 'accelerate', f'train-features-fnames-{subset_tag}-{config.fname}')
    test_embs_path = os.path.join(config.pretrained, 'accelerate', f'val-features-fnames-{config.fname}')
    os.makedirs(name=os.path.dirname(train_embs_path), exist_ok=True)
    os.makedirs(name=os.path.dirname(test_embs_path), exist_ok=True)
    logging.info(train_embs_path)
    logging.info(test_embs_path)
    lambd = config.lambd
    config.pretrained = os.path.join(config.pretrained, config.fname)
    config = ml_collections.FrozenConfigDict(config)

    # -- Function to make train/test dataloader
    def init_pipe(training):
        # -- make data transforms
        transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))])
        # -- init data-loaders/samplers
        subset_file = config.subset_path if training else None
        data_loader, _ = init_data(
            transform=transform,
            batch_size=16,
            num_workers=4,
            world_size=1,
            rank=0,
            root_path=config.root_path,
            image_folder=config.image_folder,
            training=training,
            copy_data=False,
            drop_last=False,
            subset_file=subset_file)
        return data_loader

    # -- Initialize the model
    encoder = init_model(
        pretrained=config.pretrained,
        model_name=config.model_name)
    encoder.eval()

    # -- Initialize the data-pipeline
    train_loader = init_pipe(training=True)
    test_loader = init_pipe(training=False)

    # -- accelerate 
    encoder, train_loader, test_loader= accelerator.prepare(encoder, train_loader, test_loader)

    # -- Compute the embeddings
    
    if config.preload and os.path.exists(train_embs_path):
        checkpoint = torch.load(train_embs_path, map_location='cpu')
        embs, labs, fnames = checkpoint['embs'], checkpoint['labs'], checkpoint['fnames']
        logging.info(f'loaded embs of shape {embs.shape}')
    else:
        embs, labs, fnames = make_embeddings_fnames(
            accelerator=accelerator,
            blocks=config.blocks,
            mask_frac=config.mask,
            data_loader=train_loader,
            encoder=encoder)

        if accelerator.is_main_process:
            torch.save({
                'embs': embs,
                'labs': labs,
                'fnames': fnames,
            }, train_embs_path)
            logging.info(f'saved train embs of shape {embs.shape}')
        accelerator.wait_for_everyone()


    if accelerator.is_main_process:
        # -- Normalize embeddings
        preprocess(embs, normalize=config.normalize, columns=False, centering=True)
        # -- Fit Logistic Regression Classifier
        lambd /= len(embs)
        classifier = Classifier(loss='multiclass-logistic', penalty=config.penalty, fit_intercept=False,
            lambda_1=lambd,
            lambda_2=lambd,
            tol=1e-3,
            solver='auto',
            max_iter=10000)

        classifier.fit(
            embs.numpy(),
            labs.numpy()
            )
        # -- Evaluate and log
        train_score = classifier.score(embs.numpy(), labs.numpy())
        # -- (save train score)
        logging.info(f'train score: {train_score}')

        pickle.dump(classifier, open(os.path.join('pretrained', f'classifier-{subset_tag}-{config.fname}.pkl'),'wb'))

    accelerator.wait_for_everyone()

    # -- If test embeddings already computed, load file, otherwise, compute
    # -- embeddings and save
    
    # -- Compute the embeddings
    if config.preload and os.path.exists(test_embs_path):
        checkpoint = torch.load(test_embs_path, map_location='cpu')
        test_embs, test_labs = checkpoint['embs'], checkpoint['labs']
        logging.info(f'loaded test embs of shape {test_embs.shape}')
    else:
        test_embs, test_labs = make_embeddings(
            accelerator=accelerator,
            blocks=config.blocks,
            mask_frac=config.mask,
            data_loader=test_loader,
            encoder=encoder)
        if accelerator.is_main_process:
            torch.save({
                'embs': test_embs,
                'labs': test_labs,
            }, test_embs_path)
            logging.info(f'saved test embs of shape {test_embs.shape}')
            
    if accelerator.is_main_process:
        # -- Normalize embeddings
        preprocess(test_embs, normalize=config.normalize, columns=False, centering=True)

        # -- Evaluate and log
        test_score = classifier.score(test_embs.numpy(), test_labs.numpy())
        # -- (save test score)
        logging.info(f'test score: {test_score}\n\n')
    
    accelerator.wait_for_everyone()

        
    


from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("func", None, "func_name")


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

def load_pretrained(
    encoder,
    pretrained
):
    checkpoint = torch.load(pretrained, map_location='cpu')
    pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logging.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logging.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logging.info(f'loaded pretrained model with msg: {msg}')
    try:
        logging.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]} '
                    f'path: {pretrained}')
    except Exception:
        pass
    del checkpoint
    return encoder

def make_embeddings_fnames(
    accelerator,
    blocks,
    mask_frac,
    data_loader,
    encoder,
    epochs=1,
):
    ipe = len(data_loader)

    z_mem, l_mem, n_mem = [], [], []
    temp_dict = torch.load('fnames_num_dict.pth.tar', map_location='cpu')
    fnames_num_dict, num_fnames_dict = temp_dict['fnames_num_dict'], temp_dict['num_fnames_dict']
    print(len(fnames_num_dict))

    for _ in range(epochs):
        for itr, (imgs, labels, fnames) in enumerate(data_loader):
            with torch.no_grad():
                z = encoder.module.forward_blocks(imgs, blocks, mask_frac) # ddp should add module
            fnames = np.array(fnames)
            nums = torch.tensor([fnames_num_dict[fname] for fname in fnames]).to(z.device)

            all_z = accelerator.gather_for_metrics(z)
            all_l = accelerator.gather_for_metrics(labels)
            all_n = accelerator.gather_for_metrics(nums)

            z_mem.append(all_z.cpu())
            l_mem.append(all_l.cpu())
            n_mem.append(all_n.cpu())

            if itr % 50 == 0:
                logging.info(f'[{itr}/{ipe}]')

    z_mem = torch.cat(z_mem, 0)
    l_mem = torch.cat(l_mem, 0)
    n_mem = torch.cat(n_mem, 0)

    logging.info(z_mem.shape)
    logging.info(l_mem.shape)
    logging.info(n_mem.shape)

    f_mem = [num_fnames_dict[i] for i in n_mem.numpy()]

    return z_mem, l_mem, f_mem

def make_embeddings(
    accelerator,
    blocks,
    mask_frac,
    data_loader,
    encoder,
    epochs=1,
):
    ipe = len(data_loader)

    z_mem, l_mem = [], []

    for _ in range(epochs):
        for itr, (imgs, labels, fnames) in enumerate(data_loader):
            with torch.no_grad():
                z = encoder.module.forward_blocks(imgs, blocks, mask_frac) # ddp should add module
                
            all_z = accelerator.gather_for_metrics(z)
            all_l = accelerator.gather_for_metrics(labels)

            z_mem.append(all_z.cpu())
            l_mem.append(all_l.cpu())

            if itr % 50 == 0:
                logging.info(f'[{itr}/{ipe}]')

    z_mem = torch.cat(z_mem, 0)
    l_mem = torch.cat(l_mem, 0)

    logging.info(z_mem.shape)
    logging.info(l_mem.shape)

    return z_mem, l_mem

def init_model(
    pretrained,
    model_name,
):
    encoder = deit.__dict__[model_name]()
    encoder.fc = None
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained)

    return encoder

def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()

    if FLAGS.func == 'train_classifier_stage1':
        train_classifier_stage1(config)
    elif FLAGS.func == 'get_all_features':
        get_all_features(config)
    elif FLAGS.func == 'get_cluster':
        get_cluster(config)
    elif FLAGS.func == 'get_aug_features':
        get_aug_features(config)
    elif FLAGS.func == 'train_classifier_stage3':
        train_classifier_stage3(config)
    
    


if __name__ == "__main__":
    app.run(main)
