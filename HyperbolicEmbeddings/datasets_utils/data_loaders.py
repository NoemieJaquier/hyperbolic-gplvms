from pathlib import Path
from typing import Tuple, Any, Optional, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.distributions
from torch.distributions import Normal
from torch import Tensor
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

# The code of this file was originally implemented in https://github.com/joeybose/HyperbolicNF/tree/master/data.


class ToDefaultTensor(transforms.Lambda):

    def __init__(self) -> None:
        super().__init__(lambda x: x.to(torch.get_default_dtype()))


class EuclideanUniform(torch.distributions.Uniform):

    def log_prob(self, value: Tensor) -> Tensor:
        return super().log_prob(value).sum(dim=-1)
    

class ImageDynamicBinarization:

    def __init__(self, train: bool, invert: bool = False) -> None:
        self.uniform = EuclideanUniform(0, 1)
        self.train = train
        self.invert = invert

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((-1,))  # Reshape per element, not batched yet.
        if self.invert:
            x = 1 - x
        if self.train:
            x = x > self.uniform.sample(x.shape)  # dynamic binarization
        else:
            x = x > 0.5  # fixed binarization for eval
        x = x.to(torch.get_default_dtype())
        return x
    

class MnistDataset:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self._in_dim = 784
        self._img_dims = (-1, 1, 28, 28)
        self.data_folder = ROOT_DIR / 'data/'

    @property
    def img_dims(self) -> Optional[Tuple[int, ...]]:
        return self._img_dims

    @property
    def in_dim(self) -> int:
        return self._in_dim

    def _get_dataset(self, train: bool, download: bool, transform: Any) -> torch.utils.data.Dataset:
        return datasets.MNIST(self.data_folder, train=train, download=download, transform=transform)

    def _load_mnist(self, train: bool, download: bool) -> DataLoader:
        # transformation = transforms.Compose(
            # [transforms.ToTensor(),
             # ToDefaultTensor(),
             # transforms.Lambda(ImageDynamicBinarization(train=train))])
        transformation = transforms.Compose(
            [transforms.ToTensor(),
            #  ToDefaultTensor(),
             ImageDynamicBinarization(train=train)])
        return DataLoader(dataset=self._get_dataset(train, download, transform=transformation),
                          batch_size=self.batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=train)
    
    def get_dynamic_binary_mnist_subset(self, train: bool, data_per_class: int = None, classes: List = None)-> torch.utils.data.Dataset:
        transformation = transforms.Compose(
            [transforms.ToTensor(),
             ToDefaultTensor(),
             ImageDynamicBinarization(train=train)])
        dataset = self._get_dataset(train=train, download=True, transform=transformation)

        # Initialize a dictionary to store 100 samples from each class
        if classes is None:
            classes = list(range(10))
        samples_per_class = {c: [] for c in classes}

        # Loop through the dataset and extract 100 samples for each class
        for image, label in dataset:
            if label in samples_per_class:
                if len(samples_per_class[label]) < data_per_class:
                    samples_per_class[label].append(image)
            if all(len(samples_per_class[c]) == data_per_class for c in classes):
                break

        # Convert the lists to torch tensors
        for label in samples_per_class:
            samples_per_class[label] = torch.stack(samples_per_class[label])

        # Display the shape of the extracted data
        # for label in range(10):
        #     print(f"Class {label}: {samples_per_class[label].shape}")

        # Stack all class tensors into a single tensor
        all_samples = torch.cat([samples_per_class[label] for label in classes], dim=0)

        all_labels = torch.tensor(np.hstack([[label] * data_per_class for label in classes]))

        # Plot the 25 first samples of 3
        # plt.figure(figsize=(10, 10))
        # for i in range(25):
        #     plt.subplot(5, 5, i + 1)
        #     plt.imshow(all_samples[300 + i].reshape(28,28), cmap='gray')
        #     plt.axis('off')
        # plt.show()

        return all_samples, all_labels
    
    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = self._load_mnist(train=True, download=False)
        test_loader = self._load_mnist(train=False, download=False)
        return train_loader, test_loader

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x_mb_, x_mb, reduction="none")
    
    def metrics(self, x_mb_: torch.Tensor, mode: str = "train") -> Dict[str, float]:
        return {}
    

def mnist_color_function(label):
    if label == 0:
        color = "darkgreen"  
    elif label == 1:
        color = (0.0, 0.1, 0.0)  
    elif label == 2:
        color = "aquamarine"
    elif label == 3:
        color = "gray"
    elif label == 4:
        color = (1.0, 0.4, 0.0)  
    elif label == 5:
        color = (1.0, 0.7, 0.0)  
    elif label == 6:
        color = (0.5, 0.7, 1.0)  
    elif label == 7:
        color = "mediumblue"
    elif label == 8:
        color = "gold" 
    elif label == 9:
        color = "crimson"

    return color
