__all__ = ['CINIC10']

import os
from pathlib import Path
from typing import Callable, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive


# based on torchvision.datasets.mnist.py (https://github.com/pytorch/vision/blob/37eb37a836fbc2c26197dfaf76d2a3f4f39f15df/torchvision/datasets/mnist.py)
class CINIC10(ImageFolder):
    """
    Ambiguous-MNIST Dataset
    Please cite:
        @article{mukhoti2021deterministic,
          title={Deterministic Neural Networks with Appropriate Inductive Biases Capture Epistemic and Aleatoric Uncertainty},
          author={Mukhoti, Jishnu and Kirsch, Andreas and van Amersfoort, Joost and Torr, Philip HS and Gal, Yarin},
          journal={arXiv preprint arXiv:2102.11582},
          year={2021}
        }
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        normalize (bool, optional): Normalize the samples.
        device: Device to use (pass `num_workers=0, pin_memory=False` to the DataLoader for max throughput)
    """

    url = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    url_md5 = "6ee4d0c996905fe93221de577967a372"

    splits = ("train", "test", "valid")
    touch_file = "all_extracted"

    def __init__(
        self,
        root: str,
        *,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        imagenet_only: bool = False
    ):
        assert split in self.splits

        if download:
            self.download(root)

        if not self._check_exists(root):
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        is_valid_file = self.is_imagenet_sample if imagenet_only else None

        super().__init__(
            os.path.join(self.data_folder(root), split),
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    @staticmethod
    def is_imagenet_sample(path: str):
        return "cifar10" not in os.path.basename(path)

    @classmethod
    def data_folder(clz, root) -> str:
        return os.path.join(root, clz.__name__)

    @classmethod
    def split_folder(clz, root, split) -> str:
        return os.path.join(clz.data_folder(root), split)

    @classmethod
    def touch_path(clz, root) -> str:
        return os.path.join(clz.data_folder(root), clz.touch_file)

    @classmethod
    def _check_exists(clz, root) -> bool:
        return all(os.path.exists(clz.split_folder(root, split)) for split in clz.splits) and os.path.exists(
            clz.touch_path(root)
        )

    @classmethod
    def download(clz, root: str) -> None:
        if clz._check_exists(root):
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(clz.url, clz.data_folder(root), md5=clz.url_md5)

        Path(clz.touch_path(root)).touch()