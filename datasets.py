from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import math
import random
from PIL import Image
import os
import glob
import einops
import torchvision.transforms.functional as F


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]


class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, label = self.dataset[item]
        y = 0
        if type(label) == np.ndarray: # If need to keep the label
            if label[1] == 1: # if label[1] == 1, this is a true label or high confidence prediction, Keep labels
                y = label[0]
            elif label[1] != 0: # for exp6
                if random.random() < self.p_uncond * (1-label[1]):
                    y = self.empty_token
                else:
                    y = label[0]
            elif random.random() < self.p_uncond: # set label none with probability p_uncond
                y = self.empty_token
            else: # keep the label if not set to none
                y = label[0]

        else: # if label is not a numpy array, then we don't need to keep labels
            if random.random() < self.p_uncond:
                y = self.empty_token
            else:
                y = label

        return x, np.int64(y)


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


# CIFAR10

class CIFAR10(DatasetFactory):
    r""" CIFAR10 dataset

    Information of the raw dataset:
         train: 50,000
         test:  10,000
         shape: 3 * 32 * 32
    """

    def __init__(self, path, random_flip=False, cfg=False, p_uncond=None, cluster_path=None):
        super().__init__()
        if cluster_path == '':
            cluster_path = None

        transform_train = [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        transform_test = [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        if random_flip:  # only for train
            transform_train.append(transforms.RandomHorizontalFlip())
        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
        self.train = datasets.CIFAR10(path, train=True, transform=transform_train, download=True)
        self.test = datasets.CIFAR10(path, train=False, transform=transform_test, download=True)

        if cluster_path is not None:
            print(f'renew targets from {cluster_path}')
            self.train.targets = np.load(cluster_path)
        assert len(self.train.targets) == 50000
        self.K = max(self.train.targets) + 1
        self.cnt = torch.tensor([len(np.where(np.array(self.train.targets) == k)[0]) for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / 50000 for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt: {self.cnt}')
        print(f'frac: {self.frac}')

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 3, 32, 32

    @property
    def fid_stat(self):
        return 'assets/fid_stats/fid_stats_cifar10_train_pytorch.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


# ImageNet


class FeatureDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        # names = sorted(os.listdir(path))
        # self.files = [os.path.join(path, name) for name in names]

    def __len__(self):
        return 1_281_167 * 2  # consider the random flip

    def __getitem__(self, idx):
        path = os.path.join(self.path, f'{idx}.npy')
        z, label = np.load(path, allow_pickle=True)
        return z, label


class ImageNet256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet512Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 64, 64

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet512_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet(DatasetFactory):
    def __init__(self, path, resolution, random_crop=False, random_flip=True, cluster_path=None, fnames_path=None):
        super().__init__()
        if cluster_path == '':
            cluster_path = None
        if fnames_path == '':
            fnames_path = None

        print(f'Counting ImageNet files from {path}')
        train_files = _list_image_files_recursively(os.path.join(path, 'train'))
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting ImageNet files')

        self.train = ImageDataset(resolution, train_files, labels=train_labels, random_crop=random_crop, random_flip=random_flip)
        self.resolution = resolution
        if len(self.train) != 1_281_167:
            print(f'Missing train samples: {len(self.train)} < 1281167')

        if cluster_path is not None:
            print(f'renew targets from {cluster_path}')
            _cluster_labels = np.load(cluster_path)
            _fnames = torch.load(fnames_path)
            fnames_cluster_labels = dict(zip(_fnames, _cluster_labels))
            self.train.labels = [fnames_cluster_labels[os.path.split(fname)[-1]] for fname in self.train.image_paths]

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet{self.resolution}_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]

class ImageNet_semi(ImageNet):
    def __init__(self, path, resolution, random_crop=False, random_flip=True, cluster_path=None, fnames_path=None, is_true_labels_path=None):
        super().__init__(path, resolution, random_crop, random_flip, cluster_path, fnames_path)
        assert is_true_labels_path is not None
        print(f'concat label with is_true_label from {is_true_labels_path}')
        _fnames = torch.load(fnames_path)
        _is_true_labels = torch.load(is_true_labels_path)
        fnames_is_true_labels = dict(zip(_fnames, _is_true_labels))
        isTruelabels = [fnames_is_true_labels[os.path.split(fname)[-1]] for fname in self.train.image_paths]
        self.train.labels = [(label, isTruelabel) for label, isTruelabel in zip(self.train.labels, isTruelabels)]


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.listdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        labels,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths
        self.labels = labels
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        label = np.array(self.labels[idx], dtype=np.float64)
        return np.transpose(arr, [2, 0, 1]), label


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


# CelebA


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class CelebA(DatasetFactory):
    r""" train: 162,770
         val:   19,867
         test:  19,962
         shape: 3 * width * width
    """

    def __init__(self, path, resolution=64, cluster_path=None):
        super().__init__()
        if cluster_path == '':
            cluster_path = None
        self.resolution = resolution

        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64

        transform = transforms.Compose([Crop(x1, x2, y1, y2), transforms.Resize(self.resolution),
                                        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)])
        self.train = datasets.CelebA(root=path, split="train", target_type=[], transform=transform, download=True)
        self.train = UnlabeledDataset(self.train)

        if cluster_path is not None:
            print(f'get targets from {cluster_path}')
            self.labels = np.load(cluster_path)
            self.train = LabeledDataset(self.train, self.labels)
            self.K = max(self.labels) + 1
            self.cnt = torch.tensor([len(np.where(np.array(self.labels) == k)[0]) for k in range(self.K)]).float()
            self.frac = [self.cnt[k] / 50000 for k in range(self.K)]
            print(f'{self.K} classes')
            print(f'cnt: {self.cnt}')
            print(f'frac: {self.frac}')
        else:
            self.labels = None

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return 'assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz'

    @property
    def has_label(self):
        return self.labels is not None

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


# LSUN Bedroom


class LSUNBedroom(DatasetFactory):
    def __init__(self, path, resolution=64):
        super().__init__()
        self.resolution = resolution
        transform = transforms.Compose([transforms.Resize(resolution), transforms.CenterCrop(resolution),
                                        transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        self.train = UnlabeledDataset(datasets.LSUN(root=path, classes=["bedroom_train"], transform=transform)) \
            if os.path.exists(os.path.join(path, 'bedroom_train_lmdb')) else None

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_lsun_bedroom{self.resolution}_train_50000.npz'

    @property
    def has_label(self):
        return False


class ImageDataset2(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        names = sorted(os.listdir(path))
        self.local_images = [os.path.join(path, name) for name in names]
        self.transform = transform

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        X = Image.open(self.local_images[idx])
        if self.transform is not None:
            X = self.transform(X)
        return X


class LSUNBedroom64(DatasetFactory):
    def __init__(self, path, cluster_path=None):
        super().__init__()
        if cluster_path == '':
            cluster_path = None

        train_path = os.path.join(path, 'lsun_bedroom64_train')
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        self.train = ImageDataset2(path=train_path, transform=transform) if os.path.exists(train_path) else None

        if cluster_path is not None:
            print(f'get targets from {cluster_path}')
            self.labels = np.load(cluster_path)
            self.train = LabeledDataset(self.train, self.labels)
            self.K = max(self.labels) + 1
            self.cnt = torch.tensor([len(np.where(np.array(self.labels) == k)[0]) for k in range(self.K)]).float()
            self.frac = [self.cnt[k] / 50000 for k in range(self.K)]
            print(f'{self.K} classes')
            print(f'cnt: {self.cnt}')
            print(f'frac: {self.frac}')
        else:
            self.labels = None

    @property
    def data_shape(self):
        return 3, 64, 64

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_lsun_bedroom64_train_50000.npz'

    @property
    def has_label(self):
        return self.labels is not None

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


# MS COCO


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


class MSCOCODatabase(Dataset):
    def __init__(self, root, annFile, size=None):
        from pycocotools.coco import COCO
        self.root = root
        self.height = self.width = size

        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        image = center_crop(self.width, self.height, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        anns = self._load_target(key)
        target = []
        for ann in anns:
            target.append(ann['caption'])

        return image, target


def int2bit(x, n=8):
    x = einops.rearrange(x, '... -> ... ()')
    x = np.right_shift(x, np.arange(n))
    x = x % 2
    return x


def bit2int(x):
    n = x.shape[-1]
    if isinstance(x, np.ndarray):
        return (x * (2 ** np.arange(n))).sum(axis=-1)
    elif isinstance(x, torch.Tensor):
        return (x * (2 ** torch.arange(n, device=x.device))).sum(dim=-1)
    else:
        raise NotImplementedError


class _BitMSCOCOText(Dataset):
    def __init__(self, annFile):
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.n_bits = self.tokenizer.vocab_size.bit_length()

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        anns = self._load_target(key)
        ann = random.choice(anns)['caption']  # string

        x = self.tokenizer(ann, truncation=True, max_length=77, return_length=True,
                           return_overflowing_tokens=False, padding="max_length", return_tensors="pt")["input_ids"]
        x = x.squeeze(dim=0)  # tokens
        x = x.numpy()
        x = int2bit(x, self.n_bits)  # {0, 1}
        x = 2 * torch.tensor(x, dtype=torch.float32) - 1  # {-1., 1.}
        return x


class BitMSCOCOText(DatasetFactory):
    def __init__(self, path):
        super().__init__()
        self.train = _BitMSCOCOText(os.path.join(path, 'annotations', 'captions_train2014.json'))

    def unpreprocess(self, v):  # to str
        # v: {-1., 1.}
        v = v > 0  # B L N
        v = bit2int(v).cpu().detach()  # B L
        ss = []
        for _v in v:
            _v = list(filter(lambda x: 0 <= x <= self.train.tokenizer.vocab_size - 1, _v))
            s = self.train.tokenizer.decode(_v, skip_special_tokens=True)
            ss.append(s)
        return ss

    @property
    def data_shape(self):
        return 77, 16

    @property
    def has_label(self):
        return False


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


class MSCOCOFeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root):
        self.root = root
        self.num_data, self.n_captions = get_feature_dir_info(root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        z = np.load(os.path.join(self.root, f'{index}.npy'))
        k = random.randint(0, self.n_captions[index] - 1)
        c = np.load(os.path.join(self.root, f'{index}_{k}.npy'))
        return z, c


def get_karpathy_val_split_gts(path):  # the ground truth for calculating captioning metrics, e.g., BLEU
    split_file = os.path.join(path, f'val_ids.npy')
    split_info = np.load(split_file)
    from pycocotools.coco import COCO
    coco_train2014 = COCO(os.path.join(path, 'captions_train2014.json'))
    coco_val2014 = COCO(os.path.join(path, 'captions_val2014.json'))
    gts = {}
    for fname, key in split_info:
        key = int(key)
        if 'train' in fname:
            gts[key] = coco_train2014.loadAnns(coco_train2014.getAnnIds(key))
        else:
            gts[key] = coco_val2014.loadAnns(coco_val2014.getAnnIds(key))
    return gts


class MSCOCOFeatureDatasetKarpathySplit(Dataset):
    def __init__(self, path, split, ret_key=False):
        self.path = path
        self.ret_key =ret_key
        split_file = os.path.join(path, f'{split}_ids.npy')
        self.split_info = np.load(split_file)

        from pycocotools.coco import COCO
        self.coco_train2014 = COCO(os.path.join(path, 'captions_train2014.json'))
        self.coco_val2014 = COCO(os.path.join(path, 'captions_val2014.json'))
        self.coco_train2014_keys = list(sorted(self.coco_train2014.imgs.keys()))
        self.coco_val2014_keys = list(sorted(self.coco_val2014.imgs.keys()))
        self.coco_train2014_keys_indexes = {key: index for index, key in enumerate(self.coco_train2014_keys)}
        self.coco_val2014_keys_indexes = {key: index for index, key in enumerate(self.coco_val2014_keys)}

        self.coco_train2014_num_data, self.coco_train2014_n_captions = get_feature_dir_info(os.path.join(path, 'train'))
        self.coco_val2014_num_data, self.coco_val2014_n_captions = get_feature_dir_info(os.path.join(path, 'val'))


    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, index):
        fname, key = self.split_info[index]
        key = int(key)
        if key in self.coco_train2014_keys_indexes:
            assert key not in self.coco_val2014_keys_indexes
            assert 'train' in fname
            index = self.coco_train2014_keys_indexes[key]
            z = np.load(os.path.join(self.path, 'train', f'{index}.npy'))
            k = random.randint(0, self.coco_train2014_n_captions[index] - 1)
            c = np.load(os.path.join(self.path, 'train', f'{index}_{k}.npy'))
        else:
            assert key not in self.coco_train2014_keys_indexes
            assert 'val' in fname
            index = self.coco_val2014_keys_indexes[key]
            z = np.load(os.path.join(self.path, 'val', f'{index}.npy'))
            k = random.randint(0, self.coco_val2014_n_captions[index] - 1)
            c = np.load(os.path.join(self.path, 'val', f'{index}_{k}.npy'))
        if self.ret_key:
            return z, c, key
        else:
            return z, c


class MSCOCO256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = MSCOCOFeatureDataset(os.path.join(path, 'train'))
        self.test = MSCOCOFeatureDataset(os.path.join(path, 'val'))
        assert len(self.train) == 82783
        assert len(self.test) == 40504
        print('Prepare dataset ok')

        self.empty_context = np.load(os.path.join(path, 'empty_context.npy'))

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.empty_context)

        # text embedding extracted by clip
        # for visulization in t2i
        self.prompts, self.contexts = [], []
        for f in sorted(os.listdir(os.path.join(path, 'run_vis')), key=lambda x: int(x.split('.')[0])):
            prompt, context = np.load(os.path.join(path, 'run_vis', f), allow_pickle=True)
            self.prompts.append(prompt)
            self.contexts.append(context)
        self.contexts = np.array(self.contexts)

        # image embedding extracted by stable diffusion image encoder
        # for visulization in i2t
        self.img_contexts = []
        for f in sorted(os.listdir(os.path.join(path, 'run_vis_i2t')), key=lambda x: int(x.split('.')[0])):
            if f.endswith('.npy'):
                img_context = np.load(os.path.join(path, 'run_vis_i2t', f))
                self.img_contexts.append(img_context)
        self.img_contexts = np.array(self.img_contexts)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_mscoco256_val.npz'


class MSCOCO256FeaturesKarpathy(DatasetFactory):  # only for i2t
    def __init__(self, path):
        super().__init__()
        print('Prepare dataset...')
        self.train = MSCOCOFeatureDatasetKarpathySplit(path, 'train')
        self.test = MSCOCOFeatureDatasetKarpathySplit(path, 'val', ret_key=True)  # for validation
        assert len(self.train) == 113287
        print('Prepare dataset ok')

        self.val_gts = get_karpathy_val_split_gts(path)

        # image embedding extracted by stable diffusion image encoder
        # for visulization in i2t
        self.img_contexts = []
        for f in sorted(os.listdir(os.path.join(path, 'run_vis_i2t')), key=lambda x: int(x.split('.')[0])):
            if f.endswith('.npy'):
                img_context = np.load(os.path.join(path, 'run_vis_i2t', f))
                self.img_contexts.append(img_context)
        self.img_contexts = np.array(self.img_contexts)

    @property
    def data_shape(self):
        return 4, 32, 32


def get_dataset(name, **kwargs):
    if name == 'cifar10':
        return CIFAR10(**kwargs)
    elif name == 'imagenet':
        return ImageNet(**kwargs)
    elif name == 'imagenet256_features':
        return ImageNet256Features(**kwargs)
    elif name == 'imagenet512_features':
        return ImageNet512Features(**kwargs)
    elif name == 'celeba':
        return CelebA(**kwargs)
    elif name == 'lsun_bedroom':
        return LSUNBedroom(**kwargs)
    elif name == 'lsun_bedroom64':
        return LSUNBedroom64(**kwargs)
    elif name == 'mscoco256_features':
        return MSCOCO256Features(**kwargs)
    elif name == 'mscoco256_features_karpathy':
        return MSCOCO256FeaturesKarpathy(**kwargs)
    elif name == 'bit_mscoco_text':
        return BitMSCOCOText(**kwargs)
    else:
        raise NotImplementedError(name)
