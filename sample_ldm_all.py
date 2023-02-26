from tools.fid_score import calculate_fid_given_paths
import ml_collections
import torch
from torch import multiprocessing as mp
import accelerate
import utils
import sde
import einops
from datasets import get_dataset
import tempfile
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from absl import logging
import builtins
import libs.autoencoder
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from absl import logging
import pickle

def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=config.output_path)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)

    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    cluster_name = config.model_name + '-' + '-'.join(config.subset_path.split('/')).split('.txt')[0]

    nnet_path = f'{config.dpm_path}/{cluster_name}/ckpts/300000.ckpt/nnet_ema.pth'
    logging.info(f'load nnet from {nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(nnet_path, map_location='cpu'))
    nnet.eval()

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def decode_large_batch(_batch):
        decode_mini_batch_size = 20  # use a small batch size since the decoder is large
        xs = []
        pt = 0
        for _decode_mini_batch_size in utils.amortize(_batch.size(0), decode_mini_batch_size):
            x = decode(_batch[pt: pt + _decode_mini_batch_size])
            pt += _decode_mini_batch_size
            xs.append(x)
        xs = torch.concat(xs, dim=0)
        assert xs.size(0) == _batch.size(0)
        return xs

    if 'cfg' in config.sample and config.sample.cfg and config.sample.scale > 0:  # classifier free guidance
        logging.info(f'Use classifier free guidance with scale={config.sample.scale}')
        def cfg_nnet(x, timesteps, y):
            _cond = nnet(x, timesteps, y=y)
            _uncond = nnet(x, timesteps, y=torch.tensor([dataset.K] * x.size(0), device=device))
            return _cond + config.sample.scale * (_cond - _uncond)
        score_model = sde.ScoreModel(cfg_nnet, pred=config.pred, sde=sde.VPSDE())
    else:
        score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE())

    logging.info(config.sample)
    assert os.path.exists(dataset.fid_stat)
    logging.info(f'sample: each class sample n_samples={config.sample.n_samples}, mode={config.train.mode}, mixed_precision={config.mixed_precision}')

    def amortize(n_samples, batch_size):
        k = n_samples // batch_size
        r = n_samples % batch_size
        return k * [batch_size] if r == 0 else k * [batch_size] + [r]

    def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, class_num=0):
        os.makedirs(path, exist_ok=True)
        idx = 0
        batch_size = mini_batch_size * accelerator.num_processes

        for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
            samples = unpreprocess_fn(sample_fn(mini_batch_size, class_num))
            samples = accelerator.gather(samples.contiguous())[:_batch_size]
            if accelerator.local_process_index == 0:
                for sample in samples:
                    save_image(sample, os.path.join(path, f"{idx}.png"))
                    idx += 1

    def sample_fn(_n_samples, class_num):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        if config.train.mode == 'uncond':
            kwargs = dict()
        elif config.train.mode == 'cond':
            torch_arr = torch.ones(_n_samples // 10, device=device, dtype=int) * torch.tensor(int(class_num))
            kwargs = dict(y=einops.repeat(torch_arr % dataset.K, 'nrow -> (nrow ncol)', ncol=10))
        else:
            raise NotImplementedError

        if config.sample.algorithm == 'euler_maruyama_sde':
            _z = sde.euler_maruyama(sde.ReverseSDE(score_model), _z_init, config.sample.sample_steps, verbose=accelerator.is_main_process, **kwargs)
        elif config.sample.algorithm == 'euler_maruyama_ode':
            _z = sde.euler_maruyama(sde.ODE(score_model), _z_init, config.sample.sample_steps, verbose=accelerator.is_main_process, **kwargs)
        elif config.sample.algorithm == 'dpm_solver':
            noise_schedule = NoiseScheduleVP(schedule='linear')
            model_fn = model_wrapper(
                score_model.noise_pred,
                noise_schedule,
                time_input_type='0',
                model_kwargs=kwargs
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule)
            _z = dpm_solver.sample(
                _z_init,
                steps=config.sample.sample_steps,
                eps=1e-4,
                adaptive_step_size=False,
                fast_version=True,
            )
        else:
            raise NotImplementedError
        return decode_large_batch(_z)
    f_read = open('idx_to_class.pkl', 'rb')
    dict2 = pickle.load(f_read)
    print(dict2)

    for i in range(1000):
        aug_samples_path = f'{config.dpm_path}/{cluster_name}/samples_for_classifier/aug_{config.augmentation_K}_samples'

        path = os.path.join(aug_samples_path, f'train/{dict2[i]}')
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        sample2dir(accelerator, path, config.augmentation_K, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess, class_num = i)


from absl import flags
from absl import app
from ml_collections import config_flags
import os
import sys
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("output_path", None, "The path to output log.")


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
        if argv[i].startswith('--config.'):
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
    config_name = get_config_name()
    hparams = get_hparams()
    config.project = config_name
    config.notes = hparams
    config.output_path = FLAGS.output_path
    
    evaluate(config)


if __name__ == "__main__":
    app.run(main)
