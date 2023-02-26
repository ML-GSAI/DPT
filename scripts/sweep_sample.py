import time
import random
import os
import argparse
import subprocess
from pathlib import Path
import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--port', type=int)
    parser.add_argument('--ckpts', type=str)

    args, unknown = parser.parse_known_args()
    args.ckpt_root = os.path.join(args.workdir, 'ckpts')
    if args.port is None:
        args.port = random.randint(10000, 30000)

    for x in unknown:
        assert '=' in x

    return args, ' '.join(unknown)


def valid_str(unknown):
    items = unknown.split(' ')
    res = []
    for item in items:
        assert item.startswith('--')
        res.append(item[2:].replace('/', '_'))
    return '_'.join(res)


def main():
    args, unknown = parse_args()
    ckpts = [f'{int(ckpt)}.ckpt' for ckpt in args.ckpts.split(',')]
    n_devices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = os.path.join(args.workdir, f'{now}_sweep_sample.log')

    for ckpt in ckpts:
        print(f'try sample {os.path.join(args.ckpt_root, ckpt)}')

        while not os.path.exists(os.path.join(args.ckpt_root, ckpt)):
            time.sleep(5 + 5 * random.random())
        time.sleep(5 + 5 * random.random())

        if os.path.exists(os.path.join(args.ckpt_root, '.state', ckpt, valid_str(unknown)[:100])):
            print(f'{ckpt} already evaluated, skip')
            continue

        os.makedirs(os.path.join(args.ckpt_root, '.state', ckpt), exist_ok=True)  # mark as running
        Path(os.path.join(args.ckpt_root, '.state', ckpt, valid_str(unknown)[:100])).touch()

        dct = dict()
        nnet_path = os.path.join(args.ckpt_root, ckpt, 'nnet_ema.pth')
        dct['nnet_path'] = nnet_path
        dct['output_path'] = output_path

        dct_str = ' '.join([f'--{key}={val}' for key, val in dct.items()])

        accelerate_args = f'--multi_gpu --main_process_port {args.port} --num_processes {n_devices} --mixed_precision fp16'
        cmd = f'accelerate launch {accelerate_args} sample.py {dct_str} {unknown}'
        cmd = list(filter(lambda x: x != '', cmd.split(' ')))
        print(cmd)
        subprocess.Popen(cmd).wait()


if __name__ == "__main__":
    main()
