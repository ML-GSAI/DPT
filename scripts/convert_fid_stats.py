import numpy as np


def convert(resolution):
    # download ImageNet reference batch from https://github.com/openai/guided-diffusion/tree/main/evaluations
    if resolution == 512:
        obj = np.load(f'VIRTUAL_imagenet{resolution}.npz')
    else:
        obj = np.load(f'VIRTUAL_imagenet{resolution}_labeled.npz')
    np.savez(f'fid_stats_imagenet{resolution}_guided_diffusion.npz', mu=obj['mu'], sigma=obj['sigma'])


convert(resolution=512)
