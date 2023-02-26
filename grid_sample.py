import ml_collections
import torch
from torch import multiprocessing as mp
import accelerate
import utils
import sde
from datasets import get_dataset
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from absl import logging
import einops
from torchvision.utils import save_image, make_grid


def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    utils.set_logger(log_level='info')

    dataset = get_dataset(**config.dataset)

    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()
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
    logging.info(f'mode={config.train.mode}, mixed_precision={config.mixed_precision}')

    def sample_fn(_x_init, _kwargs):
        if config.sample.algorithm == 'euler_maruyama_sde':
            rsde = sde.ReverseSDE(score_model)
            _samples = sde.euler_maruyama(rsde, _x_init, config.sample.sample_steps,
                                          verbose=accelerator.is_main_process, **_kwargs)
        elif config.sample.algorithm == 'euler_maruyama_ode':
            rsde = sde.ODE(score_model)
            _samples = sde.euler_maruyama(rsde, _x_init, config.sample.sample_steps,
                                          verbose=accelerator.is_main_process, **_kwargs)
        elif config.sample.algorithm == 'dpm_solver':
            noise_schedule = NoiseScheduleVP(schedule='linear')
            model_fn = model_wrapper(
                score_model.noise_pred,
                noise_schedule,
                time_input_type='0',
                model_kwargs=_kwargs
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule)
            _samples = dpm_solver.sample(
                _x_init,
                steps=config.sample.sample_steps,
                eps=1e-4,
                adaptive_step_size=False,
                fast_version=True,
            )
        else:
            raise NotImplementedError

        return _samples

    if config.train.mode == 'uncond':
        x_init = torch.randn(100, *dataset.data_shape, device=device)
        kwargs = dict()
        idx = 0
        samples = []
        for _batch_size in utils.amortize(len(x_init), config.sample.mini_batch_size):
            samples.append(sample_fn(x_init[idx: idx + _batch_size], kwargs))
            idx += _batch_size

    elif config.train.mode == 'cond':
        x_init = torch.randn(config.nnet.num_classes * 10, *dataset.data_shape, device=device)
        y = einops.repeat(torch.arange(config.nnet.num_classes, device=device), 'nrow -> (nrow ncol)', ncol=10)
        idx = 0
        samples = []
        for _batch_size in utils.amortize(len(x_init), config.sample.mini_batch_size):
            samples.append(sample_fn(x_init[idx: idx + _batch_size], dict(y=y[idx: idx + _batch_size])))
            idx += _batch_size

    else:
        raise NotImplementedError

    samples = torch.cat(samples, dim=0)
    samples = dataset.unpreprocess(samples)
    save_image(make_grid(samples, 10), config.output_path)



from absl import flags
from absl import app
from ml_collections import config_flags


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output samples.")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    evaluate(config)


if __name__ == "__main__":
    app.run(main)
