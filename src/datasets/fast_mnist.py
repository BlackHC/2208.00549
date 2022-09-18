"""
To speed up experiments, we are going to use Joost's FastMNIST
(https://tinyurl.com/pytorch-fast-mnist), which preloads the dataset onto
the target device.
"""
__all__ = ['FastMNIST', 'FastFashionMNIST']


from typing import Optional, Callable


from torchvision.datasets import MNIST, FashionMNIST


# From https://tinyurl.com/pytorch-fast-mnist
class FastMNIST(MNIST):
    def __init__(self, root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 *args, device, **kwargs):
        super().__init__(root, train, transform, target_transform, download, *args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class FastFashionMNIST(FashionMNIST):
    def __init__(self, root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            *args, device, **kwargs):
        super().__init__(root, train, transform, target_transform, download, *args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual FashionMNIST mean and std
        self.data = self.data.sub_(0.2861).div_(0.3530)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target
