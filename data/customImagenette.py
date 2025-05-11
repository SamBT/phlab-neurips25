from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

from torchvision.datasets import VisionDataset#, Imagenette
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive
from torchvision.datasets.folder import find_classes, make_dataset

import torch

### Stuff for custom imagenette dataset using saved tensors
### hopefully faster than loading PIL images on the fly and converting

class TensorImagenette(VisionDataset):
    """
    rudimentary rewrite of torchvision 0.19 Imagenette to support loading pre-converted tensors instead of jpegs
    hopefully speeds things up!
    """

    _ARCHIVES = {
        "full": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz", "fe2fc210e6bb7c5664d602c3cd71e612"),
        "320px": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz", "3df6f0d01a2c9592104656642f5e78a3"),
        "160px": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", "e793b78cc4c9e9a4ccc0c1155377a412"),
    }
    _WNID_TO_CLASS = {
        "n01440764": ("tench", "Tinca tinca"),
        "n02102040": ("English springer", "English springer spaniel"),
        "n02979186": ("cassette player",),
        "n03000684": ("chain saw", "chainsaw"),
        "n03028079": ("church", "church building"),
        "n03394916": ("French horn", "horn"),
        "n03417042": ("garbage truck", "dustcart"),
        "n03425413": ("gas pump", "gasoline pump", "petrol pump", "island dispenser"),
        "n03445777": ("golf ball",),
        "n03888257": ("parachute", "chute"),
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        size: str = "full",
        download=False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        preload: bool = False
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ["train", "val"])
        self._size = verify_str_arg(size, "size", ["full", "320px", "160px"])

        self._url, self._md5 = self._ARCHIVES[self._size]
        self._size_root = Path(self.root) / Path(self._url).stem
        self._image_root = str(self._size_root / self._split)

        if download:
            self._download()
        elif not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it.")

        self.wnids, self.wnid_to_idx = find_classes(self._image_root)
        self.classes = [self._WNID_TO_CLASS[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            class_name: idx for wnid, idx in self.wnid_to_idx.items() for class_name in self._WNID_TO_CLASS[wnid]
        }
        self._samples = make_dataset(self._image_root, self.wnid_to_idx, extensions=".pt")
        self.preload = preload
        self.preloaded_data = []
        self.preloaded_images = []
        if self.preload:
            for x in self._samples:
                path,_ = x
                image = torch.load(path)
                self.preloaded_data.append(image)

    def _check_exists(self) -> bool:
        return self._size_root.exists()

    def _download(self):
        if self._check_exists():
            raise RuntimeError(
                f"The directory {self._size_root} already exists. "
                f"If you want to re-download or re-extract the images, delete the directory."
            )

        download_and_extract_archive(self._url, self.root, md5=self._md5)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        path, label = self._samples[idx]
        if self.preload:
            image = self.preloaded_data[idx]
        else:
            image = torch.load(path)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self._samples)
