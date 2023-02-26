import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.lambd = 0.0025
    config.penalty = 'l2'
    config.mask = 0.0
    config.preload = True
    config.fname = 'vitb4_300ep.pth.tar'
    config.model_name = 'deit_base_p4'
    config.pretrained = 'pretrained/'

    config.normalize = True
    config.root_path = '/cache/datasets/ILSVRC/Data/'
    config.image_folder = 'CLS-LOC/'
    config.image_path = '/cache/datasets/ILSVRC/Data/CLS-LOC'
    config.subset_path = 'imagenet_subsets2/2imgs_class.txt'
    config.blocks = 1

    config.seed = 1234
    config.pred = 'noise_pred'
    config.ema_rate = 0.9999
    config.z_shape = (4, 32, 32)
    config.resolution = 256

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth'
    )

    config.dpm_path = 'assets/DPM'

    # augmentation
    config.augmentation_K = 128 # add 128 generated images per class
    config.using_true_label = True
    config.output_path = ''

    config.train = d(
        n_steps=300000,
        batch_size=1024,
        mode='cond',
        log_interval=10,
        eval_interval=5000,
        save_interval=50000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
        amsgrad=False
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit',
        img_size=32,
        patch_size=2,
        in_chans=4,
        embed_dim=1024,
        depth=20,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        pos_drop_rate=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        mlp_time_embed=False,
        num_classes=1001,
        final_layer='conv',
        skip='concat',
        token_conv='none',
        use_checkpoint=True
    )

    config.dataset = d(
        name='imagenet256_features',
        path='',
        cfg=True,
        p_uncond=0.15
    )

    config.sample = d(
        sample_steps=50,
        n_samples=50000,
        mini_batch_size=20,  # the decoder is large
        algorithm='dpm_solver',
        cfg=True,
        scale=0.4,
        path=''
    )

    return config

config = get_config()
