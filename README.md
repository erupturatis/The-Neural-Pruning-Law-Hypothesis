# Experiments

## Overview
This code allows you to run predefined experiments for training and pruning various neural network models (LeNet300, ResNet50, VGG19) on datasets such as MNIST, CIFAR-10, CIFAR-100, and ImageNet. The focus is on exploring different model sparsities. You also have codes to train networks from scratch for Cifar10/100. 

For running Neural pruning law hypothesis experiments, run manually the imagenet_NPLH.py and NPLH_experiments.py files. For imagenet in particular, you will need to setup the proper environment variables

## Prerequisites
- Python 3.x
- Required dependencies: Ensure you have all necessary packages installed. You might have a `requirements.txt` file for this.
- Pytorch dependencies are not included in `requirements.txt`, make sure to install them from the [official website](https://pytorch.org/get-started/locally/)

## Usage + examples

The script is controlled via command-line arguments to select and run specific experiments.

### Basic Command

```bash
python entry.py --experiment <experiment_name> [--sparsity <percentage>]
```

Running bottleneck experiment

```bash
python entry.py --experiment lenet300_bottleneck
```

Running convergencec MNIST experiment
```bash
python entry.py --experiment lenet300_convergence
```

Run resnet50 cifar100 at 98% sparsity
```bash
python entry.py --experiment resnet50_cifar100 --sparsity 98
```

Run Vgg19 cifar10 at 95% sparsity
```bash
python entry.py --experiment vgg19_cifar10 --sparsity 95
```

For the imagenet experiments, we created predefined files to run for each sparsity, as they are more sensible to regrowth and pruning. The commands are the following

For 96.5% sparsity
```bash
python run_experiments.py --experiment resnet50_imagenet_96.5
```

For 95% sparsity
```bash
python run_experiments.py --experiment resnet50_imagenet_95
```

For 90% sparsity
```bash
python run_experiments.py --experiment resnet50_imagenet_90
```


