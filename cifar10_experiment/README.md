## Diffusion Models and Semi-Supervised Learners Benefit Mutually with Few Labels<br><sub>Official PyTorch implementation of the NeurIPS 2023 paper</sub>

**Diffusion Models and Semi-Supervised Learners Benefit Mutually with Few Labels**<br>
You, Zebin and Zhong, Yong and Bao, Fan and Sun, Jiacheng and Li, Chongxuan and Zhu, Jun
<br>https://arxiv.org/abs/2302.10586<br>

Abstract: *In an effort to further advance semi-supervised generative and classification tasks, we propose a simple yet effective training strategy called dual pseudo training (DPT), built upon strong semi-supervised learners and diffusion models. DPT operates in three stages: training a classifier on partially labeled data to predict pseudo-labels; training a conditional generative model using these pseudo-labels to generate pseudo images; and retraining the classifier with a mix of real and pseudo images. Empirically, DPT consistently achieves SOTA performance of semi-supervised generation and classification across various settings. In particular, with one or two labels per class, DPT achieves a Fréchet Inception Distance (FID) score of 3.08 or 2.52 on ImageNet 256x256. Besides, DPT outperforms competitive semi-supervised baselines substantially on ImageNet classification tasks, achieving top-1 accuracies of 59.0 (+2.8), 69.5 (+3.0), and 74.4 (+2.0) with one, two, or five labels per class, respectively. Notably, our results demonstrate that diffusion can generate realistic images with only a few labels (e.g., <0.1%) and generative augmentation remains viable for semi-supervised classification.*

## Requirements
```.bash
conda create -n dpt_cifar python==3.9
conda activate dpt_cifar
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install tensorboard
pip install accelerate 
pip install --upgrade scikit-learn
pip install pandas
pip install click
pip install Pillow==9.3.0
```

## Get Started
### Stage 1: Training the Classifier and Generating Pseudo-Labels
To begin, navigate to the TorchSSL directory and run the following commands to train a classifier, generate pseudo-labels, and prepare the dataset required for EDM.
```.bash
# Stage 1: Training the Classifier and Generating Pseudo-Labels
# After training, a high-quality classifier will be saved in the directory: saved_models/freematch_cifar10_40_1/, We can then use this classifier to obtain pseudo-labels
python freematch.py --c config/freematch_cifar10_40_1.yaml

# Generating Pseudo-Labels
python get_labels.py --load_path ./saved_models/freematch_cifar10_40_1/model_best.pth --save_path ./saved_models/freematch_cifar10_40_1/ >> saved_models/freematch_cifar10_40_1/get_labels.log

# Using pseudo-labels, we create a new dataset, which will be used to train our generative model
# Creating a Pseudo-Dataset
python pseudo_dataset.py --data_path ./data/cifar-10-batches-py --save_path ./saved_models/freematch_cifar10_40_1
```
The purpose of the above steps is to generate a new dataset, similar to CIFAR-10 but with pseudo-labels instead of true labels.

```.bash
# Preparing Pseudo Dataset for Training EDM
python dataset_tool.py --source=./saved_models/freematch_cifar10_40_1/cifar-10-python.tar.gz --dest=./saved_models/freematch_cifar10_40_1/freematch-40-seed1-cifar10-32x32.zip
```
This step converts the dataset into the format required by EDM.

### Stage 2: Training the Generative Model and Generating Pseudo-Images
Following the dataset preparation, navigate to the EDM directory and execute the following commands for the second stage:
```.bash
# Stage 2: Training the Generative Model
torchrun --standalone --nproc_per_node=4 train.py --outdir=training-runs \
    --data=../TorchSSL/saved_models/freematch_cifar10_40_1/freematch-40-seed1-cifar10-32x32.zip --cond=1 --arch=ddpmpp --batch-gpu=32
```
In the above command:
- The **`--data`** flag specifies the dataset created earlier
- The **`--cond`** indicates conditional generation
- The **`--arch=ddpmpp`** signifies the use of the ddpmpp model
- The **`--batch-gpu=32`** sets the batch size per GPU to 32
This command is used for training the generative model in the second stage of the process.

Assuming the EDM training has been completed, and the latest generative model is saved as **'./training-runs/00000-freematch-40-seed1-cifar10-32x32-cond-ddpmpp-edm-gpus4-batch512-fp32/network-snapshot-latest.pkl'**, you can use the following command to generate pseudo-images:
```.bash
bash generate.sh 1000 ../TorchSSL/data-aug/nums_1000 ./training-runs/00000-freematch-40-seed1-cifar10-32x32-cond-ddpmpp-edm-gpus4-batch512-fp32/network-snapshot-latest.pkl 4
```
In the above command:
- The first argument, **`1000`**, represents generating 1001 samples for each class, ranging from 0 to 1000.
- The second argument, **`../TorchSSL/data-aug/nums_1000`**, represents the path where the generated samples will be saved.
- The third argument, **`./training-runs/00000-freematch-40-seed1-cifar10-32x32-cond-ddpmpp-edm-gpus4-batch512-fp32/network-snapshot-latest.pkl`**, is the path to the generative model you want to use for generation.
- The fourth argument, **`4`**, specifies the number of GPUs to use for the generation process.

### Stage 3: Training the Classifier with Pseudo-Images
Finally, we can train the classifier with the pseudo-images generated in the previous step. To do so, navigate to the TorchSSL directory and execute the following commands:
```.bash
# stage3
python freematch.py --c config/freematch_cifar10_40_1_aug1000.yaml --aug_path=./data-aug/nums_1000
```

## Question
If you have any questions, please feel free to contact us via email: zebin@ruc.edu.cn

## Citation

```
@inproceedings{you2023diffusion,
  author    = {You, Zebin and Zhong, Yong and Bao, Fan and Sun, Jiacheng and Li, Chongxuan and Zhu, Jun},
  title     = {Diffusion models and semi-supervised learners benefit mutually with few labels},
  booktitle = {Proc. NeurIPS},
  year      = {2023}
}
```

## Acknowledgments

We would like to express our gratitude to the remarkable projects EDM (https://github.com/NVlabs/edm) and TorchSSL (https://github.com/TorchSSL/TorchSSL). Our work is built upon their contributions.
