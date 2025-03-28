from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset

import io
from datasets.io import load_patches


class PatchDataset(Dataset):
    def __init__(self, patch_file: Path, augs: Callable, s3_client=None, patch_size: int = 256):
        super(PatchDataset, self).__init__()
        if s3_client is not None:
            file = io.BytesIO()
            self.s3_client.download_fileobj(self.bucket_name, patch_file, file)
            file.seek(0)
        else:
            file = patch_file

        self.patches = load_patches(file, str(patch_size))
        self.augs = augs

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img = self.patches[idx]
        # apply augmentations
        img = self.augs(img)
        return img
