from __future__ import print_function, division
import os

import torch
import numpy as np

from utils import net_builder
from datasets.ssl_dataset import SSL_Dataset
from datasets.data_utils import get_data_loader
import accelerate

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='./saved_models/fixmatch/model_best.pth')
    parser.add_argument('--use_train_model', action='store_true')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)

    parser.add_argument('--save_path', type=str, default='./saved_models/fixmatch')

    args = parser.parse_args()

    accelerator = accelerate.Accelerator()
    
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['train_model'] if args.use_train_model else checkpoint['ema_model']
    
    _net_builder = net_builder(args.net, 
                               args.net_from_name,
                               {'depth': args.depth, 
                                'widen_factor': args.widen_factor,
                                'leaky_slope': args.leaky_slope,
                                'dropRate': args.dropout,
                                'use_embed': False})
    
    net = _net_builder(num_classes=args.num_classes)
    
    _train_dset = SSL_Dataset(args, alg='fullysupervised', name=args.dataset, train=True,
                                num_classes=args.num_classes, data_dir=args.data_dir)
    train_dset = _train_dset.get_label_dset()


    train_loader = get_data_loader(train_dset,
                                  args.batch_size, 
                                  num_workers=1, shuffle=False)
    
    # net, train_loader = accelerator.prepare(net, train_loader)

    # net.load_state_dict(load_model)
    weights_dict = {}
    for k, v in load_model.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    net.load_state_dict(weights_dict)
    # if torch.cuda.is_available():
    #     net.cuda()
    net.eval()

    acc = 0.0
    target_list = []
    pseudo_label_list = []
    with torch.no_grad():
        for (_, image, target) in train_loader:
            print(_)
            image = image.type(torch.FloatTensor)#.cuda()
            logit = net(image)
            print(logit.shape)
            
            target, logit = accelerator.gather(target), accelerator.gather(logit)
            for x, y in zip(target.cpu(), logit.cpu().max(1)[1]):
                target_list.append(x)
                pseudo_label_list.append(y)

            acc += logit.cpu().max(1)[1].eq(target.cpu()).sum().numpy()
    
    accelerator.print(len(train_dset))

    target_list = torch.stack(target_list)
    pseudo_label_list = torch.stack(pseudo_label_list)

    np.save(os.path.join(args.save_path, 'pseudo_label.npy'), pseudo_label_list)

    #accelerator.print(pseudo_label_list[:100],'\n', target_list[:100])

    print(pseudo_label_list[:100], '\n', target_list[:100])
    print(pseudo_label_list[:100] == target_list[:100])

    result1 = torch.bincount(target_list)
    result2 = torch.bincount(pseudo_label_list)

    print(result1)
    print(result2)

    accelerator.print(f"Test Accuracy: {acc/len(train_dset)}")

    

    

    