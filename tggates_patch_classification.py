import argparse
from multiprocessing.sharedctypes import class_cache
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from ctran import ctranspath
from datasets.io import load_patches

LESION_CLASSES = [
    "Necrosis (area)",
    "Necrosis (single cell)",
    "Mitosis"
]

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
test_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
        default="./ctranspath.pth"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory to write results and logs",
        required=True,
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Path to the dataset directory",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for feature extraction",
        default=512
    )
    return parser


def load_patches_and_labels(dataset_dir: Path, split: str) -> tuple[list[Image], np.array]:
    """
    Load patches and labels from the dataset.

    Args:
        dataset_dir: path to the dataset
        split: split to load

    Returns:
        tuple of patches and labels
    """
    patch_list = []
    label_list = []
    dataset_dir = Path(dataset_dir) / split
    for patch_file in dataset_dir.glob("*.h5"):
        with h5py.File(patch_file, "r") as f:
            if "labels" not in f:
                continue
            labels = f["labels"][:]
        if len(labels) == 0:
            continue
        patches = load_patches(patch_file)
        patch_list.extend(patches)
        label_list.append(labels)

    # stack labels into a single array
    labels = np.concatenate(label_list)
    return patch_list, labels


class PatchClassificationDataset(Dataset):
    def __init__(self, patches: list[Image]):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        patch = self.patches[idx]
        return test_transform(patch)


def compute_embeddings(model: nn.Module, patches: list[Image], batch_size: int) -> np.array:
    patch_dataset = PatchClassificationDataset(patches)
    dataloader = DataLoader(patch_dataset, batch_size=batch_size, pin_memory=True)
    embeddings_list = []
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            batch = batch.cuda(non_blocking=True)
            embeddings = model(batch).float()
            embeddings_list.append(embeddings.cpu().numpy())

    embeddings = np.concatenate(embeddings_list)
    return embeddings


def compute_auc(
        train_embeddings: np.array,
        train_labels_binary: list[int],
        test_embeddings: np.array,
        test_labels_binary: list[int]
) -> float:
    """
    Train Logistic Regression model on the train embeddings and compute AUC on the test embeddings.

    Args:
        train_embeddings: embeddings for the train set
        train_labels_binary: binary labels for the train set
        test_embeddings: embeddings for the test set
        test_labels_binary: binary labels for the test set

    Returns:
        the AUC score
    """
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)
    clf.fit(train_embeddings, train_labels_binary)
    test_probs = clf.predict_proba(test_embeddings)[:, 1]
    # compute roc auc
    roc_auc = roc_auc_score(test_labels_binary, test_probs)
    return roc_auc


def main(args):
    # load the model
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(args.pretrained_weights)
    model.load_state_dict(td['model'], strict=True)
    model.cuda()
    model.eval()
    # load training and test set
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)

    # load patches and labels
    train_patches, train_labels = load_patches_and_labels(dataset_dir, "train")
    print(f"Loaded {len(train_patches)} training patches")
    test_patches, test_labels = load_patches_and_labels(dataset_dir, "test")
    print(f"Loaded {len(test_patches)} test patches")

    # compute embeddings
    train_embeddings = compute_embeddings(model, train_patches, args.batch_size)
    test_embeddings = compute_embeddings(model, test_patches, args.batch_size)

    # train logistic regression model
    auc_scores = {}
    for lesion_class in LESION_CLASSES:
        lesion_index = LESION_CLASSES.index(lesion_class)
        # convert labels to binary
        train_labels_binary = [1 if l[lesion_index] > 0 else 0 for l in train_labels]
        test_labels_binary = [1 if l[lesion_index] > 0 else 0 for l in test_labels]
        print(
            f"Train: {train_labels_binary.count(1)} positive examples, {train_labels_binary.count(0)} negative examples"
        )
        print(
            f"Test: {test_labels_binary.count(1)} positive examples, {test_labels_binary.count(0)} negative examples"
        )

        auc = compute_auc(train_embeddings, train_labels_binary, test_embeddings, test_labels_binary)
        print(f"Class {lesion_class} AUC: {auc}")
        auc_scores[lesion_class] = auc

    avg_auc = np.mean(auc_scores.values())
    print(f"Average AUC: {avg_auc}")


if __name__ == '__main__':
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    main(args)
