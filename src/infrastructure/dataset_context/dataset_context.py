from enum import Enum
from typing import Tuple
import kornia.augmentation as K
from abc import ABC, abstractmethod
from src.infrastructure.configs_general import train_validation_split
from src.infrastructure.constants import DATA_PATH, IMAGENET_PATH
from src.infrastructure.dataset_context.data_preprocessing import cifar10_preprocess, mnist_preprocess, \
    cifar100_preprocess
import torch
from datasets import load_dataset, DownloadMode, DownloadConfig
import os
from src.infrastructure.others import get_device
import torch.nn as nn
from dataclasses import dataclass
from datasets import load_dataset
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from enum import Enum
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.infrastructure.dataset_context.data_preprocessing import cifar10_preprocess, mnist_preprocess
from src.infrastructure.others import get_device
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

############################################################################
# CIFAR10, CIFAR100, MNIST
############################################################################

class DatasetSmallType(Enum):
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    MNIST = 'mnist'

@dataclass
class DatasetContextConfigs:
    batch_size: int
    augmentations: nn.Sequential
    augmentations_test: nn.Sequential

mean_cifar100, std_cifar100 = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
mean_cifar10, std_cifar10 = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
mean_mnist, std_mnist = (0.1307,), (0.3081,)

_AUGMENTATIONS_CIFAR_10 = nn.Sequential(
    K.RandomCrop((32, 32), padding=4),
    K.RandomHorizontalFlip(p=0.5),
    K.Normalize(mean_cifar10, std_cifar10),
).to(get_device())

_AUGMENTATIONS_CIFAR_10_TEST = nn.Sequential(
    K.Normalize(mean_cifar10, std_cifar10),
).to(get_device())

_AUGMENTATIONS_CIFAR_100 = nn.Sequential(
    K.RandomCrop((32, 32), padding=4),
    K.RandomHorizontalFlip(p=0.5),
    K.Normalize(mean_cifar100, std_cifar100),
).to(get_device())

_AUGMENTATION_CIFAR_100_TEST = nn.Sequential(
    K.Normalize(mean_cifar100, std_cifar100),
)

_AUGMENTATIONS_MNIST = nn.Sequential(
    K.Normalize(mean_mnist, std_mnist),
)

_AUGMENTATIONS_MNIST_TEST = nn.Sequential(
    K.Normalize(mean_mnist, std_mnist),
)

BATCH_SIZE_CIFAR_10 = 128
BATCH_SIZE_CIFAR_100 = 128
BATCH_SIZE_MNIST = 128

def dataset_context_configs_cifar100() -> DatasetContextConfigs:
    return DatasetContextConfigs(batch_size=BATCH_SIZE_CIFAR_100, augmentations=_AUGMENTATIONS_CIFAR_100, augmentations_test=_AUGMENTATION_CIFAR_100_TEST)

def dataset_context_configs_cifar10() -> DatasetContextConfigs:
    return DatasetContextConfigs(batch_size=BATCH_SIZE_CIFAR_10, augmentations=_AUGMENTATIONS_CIFAR_10, augmentations_test=_AUGMENTATIONS_CIFAR_10_TEST)

def dataset_context_configs_mnist() -> DatasetContextConfigs:
    return DatasetContextConfigs(batch_size=BATCH_SIZE_MNIST, augmentations=_AUGMENTATIONS_MNIST, augmentations_test=_AUGMENTATIONS_MNIST_TEST)

class DatasetContextAbstract(ABC):
    # TRAINING
    @abstractmethod
    def get_total_batches_training(self) -> int:
        pass

    @abstractmethod
    def get_batch_training_index(self) -> int:
        pass

    @abstractmethod
    def any_data_training_available(self) -> bool:
        pass

    @abstractmethod
    def get_data_training_length(self) -> int:
        pass

    # TESTING
    @abstractmethod
    def get_total_batches_testing(self) -> int:
        pass

    @abstractmethod
    def get_batch_testing_index(self) -> int:
        pass

    @abstractmethod
    def any_data_testing_available(self) -> bool:
        pass

    @abstractmethod
    def get_data_testing_length(self) -> int:
        pass

    @abstractmethod
    def get_batch_size(self) -> int:
        pass

    @abstractmethod
    def get_training_data_and_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def get_testing_data_and_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class DatasetSmallContext(DatasetContextAbstract):
    def __init__(self, dataset: DatasetSmallType, configs: DatasetContextConfigs):
        """
        Loads the entire dataset on the GPU, since dataset is small
        """

        self.training_data_indices_iterator = None
        self.testing_data_indices_iterator = None
        self.dataset = dataset
        self.configs = configs

        self.batch_training_index = 0
        self.batch_test_index = 0

        if dataset == DatasetSmallType.CIFAR10:
            self.train_data, self.train_labels, self.test_data, self.test_labels = cifar10_preprocess()

            total_train = self.train_data.size(0)
            train_size = int(total_train * train_validation_split)
            val_size = total_train - train_size
            self.train_data, self.val_data = torch.split(self.train_data, [train_size, val_size])
            self.train_labels, self.val_labels = torch.split(self.train_labels, [train_size, val_size])
        if dataset == DatasetSmallType.MNIST:
            self.train_data, self.train_labels, self.test_data, self.test_labels = mnist_preprocess()

            total_train = self.train_data.size(0)
            train_size = int(total_train * train_validation_split)
            val_size = total_train - train_size
            self.train_data, self.val_data = torch.split(self.train_data, [train_size, val_size])
            self.train_labels, self.val_labels = torch.split(self.train_labels, [train_size, val_size])
        if dataset == DatasetSmallType.CIFAR100:
            self.train_data, self.train_labels, self.test_data, self.test_labels = cifar100_preprocess()

            total_train = self.train_data.size(0)
            train_size = int(total_train * train_validation_split)
            val_size = total_train - train_size
            self.train_data, self.val_data = torch.split(self.train_data, [train_size, val_size])
            self.train_labels, self.val_labels = torch.split(self.train_labels, [train_size, val_size])

        self.total_training_batches = len(self.train_labels) // self.configs.batch_size
        self.total_test_batches = len(self.test_labels) // self.configs.batch_size

        if len(self.train_labels) % self.configs.batch_size != 0:
            self.total_training_batches += 1
        if len(self.test_labels) % self.configs.batch_size != 0:
            self.total_test_batches += 1

    def init_data_split(self):
        batch_size = self.configs.batch_size
        device = get_device()

        total_training_data_len = len(self.train_data)
        indices = torch.randperm(total_training_data_len, device=device)
        batch_indices = torch.split(indices, batch_size)
        self.training_data_indices_iterator = iter(enumerate(batch_indices))
        self.batch_training_index = 0

        total_test_data_len = len(self.test_data)
        indices = torch.randperm(total_test_data_len, device=device)
        batch_indices = torch.split(indices, batch_size)
        self.testing_data_indices_iterator = iter(enumerate(batch_indices))
        self.batch_test_index = 0

    # TRAINING
    def get_total_batches_training(self) -> int:
        return self.total_training_batches

    def get_batch_training_index(self) -> int:
        return self.batch_training_index

    def any_data_training_available(self) -> bool:
        return self.batch_training_index < self.total_training_batches

    def get_data_training_length(self) -> int:
        return len(self.train_labels)

    def get_training_data_and_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.batch_training_index += 1
        batch_idx, batch = next(self.training_data_indices_iterator)

        data = self.train_data[batch].to(get_device(), non_blocking=True)
        target = self.train_labels[batch].to(get_device(), non_blocking=True)
        data = self.configs.augmentations(data)

        return data, target

    # TESTING
    def get_total_batches_testing(self) -> int:
        return self.total_test_batches

    def get_batch_testing_index(self) -> int:
        return self.batch_test_index

    def any_data_testing_available(self) -> bool:
        return self.batch_test_index < self.total_test_batches

    def get_data_testing_length(self) -> int:
        return len(self.test_data)

    def get_batch_size(self) -> int:
        return self.configs.batch_size

    def get_testing_data_and_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.batch_test_index += 1
        batch_idx, batch = next(self.testing_data_indices_iterator)
        data = self.test_data[batch].to(get_device(), non_blocking=True)
        target = self.test_labels[batch].to(get_device(), non_blocking=True)
        data = self.configs.augmentations_test(data)

        return data, target


############################################################################
# ImageNet
############################################################################

_resnet50_imagenet_train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

_resnet50_imagenet_val_transforms = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


import random
class _ImageNetHFDataset(Dataset):
    def __init__(self, hf_data, split='train', max_retries=5):
        self.hf_data = hf_data
        self.split = split
        self.transform = _resnet50_imagenet_train_transforms if split == 'train' else _resnet50_imagenet_val_transforms
        self.max_retries = max_retries
        self.corrupted_count = 0  

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, idx):
        retries = 0
        while retries < self.max_retries:
            try:
                example = self.hf_data[idx]
                image = example['image'].convert("RGB")
                label = example['label']
                if self.transform:
                    image = self.transform(image)
                return image, label
            except:
                print(f'Corrupted file detected at index {idx}: {e}')
                self.corrupted_count += 1
                retries += 1
                # Select a new random index to attempt loading
                idx = random.randint(0, len(self.hf_data) - 1)
        # After max retries, raise an exception or handle accordingly
        raise RuntimeError(f"Max retries exceeded for index {idx}.")

@dataclass
class DatasetImageNetContextConfigs:
    batch_size: int

class DatasetImageNetContext(DatasetContextAbstract):
    def __init__(self, configs: DatasetImageNetContextConfigs, cache_dir: str = None):
        """
        Does NOT load everything into GPU. Instead, uses DataLoader with parallel workers.
        We apply transformations in the ImageNetHFDataset class on the CPU, before moving data on the GPU, via the dataloader
        """

        self.configs = configs
        self.batch_training_index = 0
        self.batch_test_index = 0
        self.dataset =  load_dataset(
                        "ILSVRC/imagenet-1k",
                        cache_dir=os.environ["HF_DATASETS_CACHE"],
                        download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,  # reuse any shard that exists
                        revision="07900defe1ccf3404ea7e5e876a64ca41192f6c07406044771544ef1505831e8",
                        download_config=DownloadConfig(local_files_only=False, resume_download=True),
                    )

        self.train_dataset = _ImageNetHFDataset(self.dataset['train'], split='train')
        self.val_dataset = _ImageNetHFDataset(self.dataset['validation'], split='val')

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.configs.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        self.test_loader = DataLoader(
            self.val_dataset,
            batch_size=self.configs.batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True
        )

        self.total_training_batches = len(self.train_loader)
        self.total_test_batches = len(self.test_loader)

        self.train_loader_iter = None
        self.test_loader_iter = None

    def init_data_split(self):
        self.batch_training_index = 0
        self.batch_test_index = 0
        self.train_loader_iter = iter(self.train_loader)
        self.test_loader_iter = iter(self.test_loader)

    def get_total_batches_training(self) -> int:
        return self.total_training_batches

    def get_batch_training_index(self) -> int:
        return self.batch_training_index

    def any_data_training_available(self) -> bool:
        return self.batch_training_index < self.total_training_batches

    def get_data_training_length(self) -> int:
        return len(self.train_dataset)

    def get_training_data_and_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns next training batch."""
        self.batch_training_index += 1
        data, target = next(self.train_loader_iter)

        data = data.to(get_device(), non_blocking=True)
        target = target.to(get_device(), non_blocking=True)

        return data, target

    # --------------------- TESTING ---------------------
    def get_total_batches_testing(self) -> int:
        return self.total_test_batches

    def get_batch_testing_index(self) -> int:
        return self.batch_test_index

    def any_data_testing_available(self) -> bool:
        return self.batch_test_index < self.total_test_batches

    def get_data_testing_length(self) -> int:
        return len(self.val_dataset)

    def get_testing_data_and_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns next testing batch."""
        self.batch_test_index += 1
        data, target = next(self.test_loader_iter)
        data = data.to(get_device(), non_blocking=True)
        target = target.to(get_device(), non_blocking=True)
        return data, target

    def get_batch_size(self) -> int:
        return self.configs.batch_size