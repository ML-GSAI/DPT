import moxing as mox
import os
import sys
from pathlib import Path
import argparse
import time

def mprint(*args):
    print('\n---------------------------')
    print(*args)
    print('---------------------------\n')

# local paths
cache_local = Path(os.environ.get('CACHE_DIR', '/cache'))
code_local = cache_local / 'DPT'
datasets_local = cache_local / 'datasets'
assets_local = code_local / 'assets'
log_local = code_local / 'log'
stable_diffusion_local = assets_local / 'stable-diffusion/autoencoder_kl.pth'
features_local = assets_local / 'datasets'
fid_stats_local = assets_local / 'fid_stats'
fid_weights_local = '/home/ma-user/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth'

parser = argparse.ArgumentParser()
parser.add_argument('--train_url', type=str)
parser.add_argument('--resume', type=str)
parser.add_argument('--bucket', type=str)
parser.add_argument('--config', type=str)
parser.add_argument('--code_path', type=str, default='')
parser.add_argument('--features_path', type=str, default='')
parser.add_argument('--generative_path', type=str, default='')
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--stage', type=int, default=1)
#drop below
parser.add_argument('--init_method', type=str)
parser.add_argument('--rank', type=str)
parser.add_argument('--world_size', type=str)

args, script_args = parser.parse_known_args()

def convert(arg):
    assert '=' in arg and arg.startswith('--')
    k, val = arg.split('=')
    k = k[2:].replace('-', '.')
    return f'--{k}={val}'

script_args = list(map(lambda x: convert(x), script_args))
script_args = ' '.join(script_args)
script_args = f'--config={args.config} {script_args}'
mprint(script_args)

def prepare():
    mprint('install environment')
    os.system(f'pip install -U pip')
    os.system(f'pip install einops')
    os.system(f'pip install ml_collections')
    os.system(f'pip install Cyanure')
    os.system(f'pip install datasets')
    os.system(f'pip install accelerate')
    os.system(f'pip install wandb')
    os.system(f'pip install tensorboard')
    os.system(f'pip install absl-py')
    os.system(f'pip install timm==0.3.2')
    
    os.system(f'mkdir -p {datasets_local}')
    os.system(f'mkdir -p {log_local}')
    os.system(f'mkdir -p {code_local}')
    os.system(f'mkdir -p {assets_local}')
    os.system(f'mkdir -p {features_local}')
    os.system(f'mkdir -p {fid_stats_local}')
    os.system('mkdir -p ~/.cache/torch/hub/')

    if args.resume is not None:
        if args.resume[-1] == '/':
            args.resume = args.resume[:-1]
        ckpt = os.path.split(args.resume)[-1]
        mox.file.copy_parallel(f's3://{args.resume}', os.path.join(args.train_url, 'ckpts', ckpt))
    
    mprint(f'copying timm0.3.2/helpers.py')
    mox.file.copy_parallel(f's3://{args.bucket}/zebin/timm0.3.2/helpers.py', '/home/ma-user/anaconda3/envs/PyTorch-1.10.2/lib/python3.7/site-packages/timm/models/layers/helpers.py')

    mprint(f'copying code to {code_local}')
    if args.code_path == '':
        mox.file.copy_parallel(f's3://{args.bucket}/zebin/DPT', str(code_local))
    else:
        mox.file.copy_parallel(f's3://{args.code_path}', str(code_local))
    
    mprint(f'copying fid_stats')
    mox.file.copy_parallel(f's3://{args.bucket}/zebin/DPM/assets/fid_stats', str(fid_stats_local))

    mprint(f'copying fid_weights')
    mox.file.copy_parallel(f's3://{args.bucket}/zebin/fid_weights/pt_inception-2015-12-05-6726825d.pth', str(fid_weights_local))

    mprint(f'copying autoencoder_kl.pth')
    mox.file.copy_parallel(f's3://{args.bucket}/zebin/DPM/assets/stable-diffusion/autoencoder_kl.pth', str(stable_diffusion_local))

    mprint(f'copying ILSVRC.tar')
    st = time.time()
    if args.bucket == 'bucket-generative':
        src = 's3://bucket-generative/fanbao/datasets/ILSVRC.tar'
    elif args.bucket == 'bucket-cneast4':
        src = 's3://bucket-cneast4/Vision_data/imagenet_ilsvrc/ILSVRC.tar'
    elif args.bucket == 'bucket-cnsouth1':
        src = 's3://bucket-backup-dataset/ImageNet/ILSVRC.tar'
    elif args.bucket == 'bucket-hang':
        src = 's3://bucket-hang/Vision_data/imagenet_ilsvrc/ILSVRC.tar'
    
    mox.file.copy_parallel(src, str(datasets_local / 'ILSVRC.tar'))
    ed = time.time()
    mprint(f'copying ILSVRC.tar takes {ed - st}s')

    mprint('extracting ILSVRC.tar')
    path = datasets_local / 'ILSVRC.tar'
    cmd = f'cd {datasets_local} && tar -xf ILSVRC.tar'
    mprint(cmd)
    os.system(cmd)
    mprint(f'extracting ILSVRC.tar takes {time.time() - ed}s')

prepare()

mprint('Runing cmd')
mprint(f'python train.py {script_args}')

main_process_ip = os.environ['VC_WORKER_HOSTS'].split(',')[0]
main_process_port = '6666'
num_machines = os.environ['MA_NUM_HOSTS']
machine_rank = os.environ['VC_TASK_INDEX']
num_processes = int(os.environ['MA_NUM_GPUS']) * int(num_machines)

accelerate_args = f'--main_process_ip {main_process_ip} --main_process_port {main_process_port} --machine_rank {machine_rank} --num_machines {num_machines} --num_processes {num_processes} --mixed_precision fp16 --multi_gpu'
mprint(f'main_process_ip={main_process_ip}, main_process_port={main_process_port}, num_machines={num_machines}, machine_rank={machine_rank}, num_processes={num_processes}')

def path_remake(path):
    return path.replace('(', '\(').replace(')', '\)')

if args.stage == 1:
    cmd = f"cd {code_local} && accelerate launch {accelerate_args} train_classifier.py {script_args} --config.output_path={log_local}/stage1_log --func=train_classifier_stage1"
    cmd = path_remake(cmd)
    mprint(cmd)
    os.system(cmd)

    cmd = f"cd {code_local} && accelerate launch {accelerate_args} train_classifier.py {script_args} --func=get_all_features"
    cmd = path_remake(cmd)
    mprint(cmd)
    os.system(cmd)

    cmd = f"cd {code_local} && accelerate launch {accelerate_args} train_classifier.py {script_args} --func=get_cluster"
    cmd = path_remake(cmd)
    mprint(cmd)
    os.system(cmd)

    if machine_rank == '0':
        cmd = f"cd {code_local} && python extract_imagenet_features_semi.py {script_args}"
        cmd = path_remake(cmd)
        mprint(cmd)
        os.system(cmd)

        mox.file.copy_parallel(code_local, os.path.join(args.save_path, 'DPT'))

if args.stage == 2:
    if args.features_path != '':
        mprint(f'copying features')
        st = time.time()

        mox.file.copy_parallel(f's3://{args.features_path}', str(code_local / 'pretrained/datasets/deit_base_p4-imagenet_subsets1-2imgs_class/imagenet256_features.tar.gz'))
        ed = time.time()
        mprint(f'copying features takes {ed - st}s')

        mprint('extracting features.tar')
        path_tmp = str(code_local / 'pretrained/datasets/deit_base_p4-imagenet_subsets1-2imgs_class')

        cmd = f'cd {path_tmp} && tar -zxvf imagenet256_features.tar.gz'
        mprint(cmd)
        os.system(cmd)
        mprint(f'extracting features.tar takes {time.time() - ed}s')

    if args.generative_path != '':
        mprint(f'copying generative model')
        st = time.time()
        mox.file.copy_parallel(f's3://{args.generative_path}', str(code_local / 'assets/DPM/deit_base_p4-imagenet_subsets1-2imgs_class/ckpts/300000.ckpt/nnet_ema.pth'))
        ed = time.time()
        mprint(f'copying generative model cost {ed - st}s')

    cmd = f"cd {code_local} && accelerate launch {accelerate_args} train_ldm.py {script_args}"
    cmd = path_remake(cmd)
    mprint(cmd)
    os.system(cmd)

    cmd = f"cd {code_local} && accelerate launch {accelerate_args} sample_ldm_all.py {script_args} --output_path={log_local}/sample_log"
    cmd = path_remake(cmd)
    mprint(cmd)
    os.system(cmd)

    cmd = f"cd {code_local} && accelerate launch {accelerate_args} train_classifier.py {script_args} --func=get_aug_features"
    cmd = path_remake(cmd)
    mprint(cmd)
    os.system(cmd)

    cmd = f"cd {code_local} && accelerate launch {accelerate_args} train_classifier.py {script_args} --func=train_classifier_stage3 --config.output_path={log_local}/stage3_log"
    cmd = path_remake(cmd)
    mprint(cmd)
    os.system(cmd)

    if machine_rank == '0':
        mox.file.copy_parallel(code_local, os.path.join(args.save_path, 'DPT'))
        



    
