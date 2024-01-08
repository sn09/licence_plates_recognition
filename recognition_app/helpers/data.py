"""Module with dataset implementations."""
import logging
import os

import cv2
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class RecognitionDataset(torch.utils.data.Dataset):
    """Dataset for training image-to-text mapping using CTC-Loss."""

    def __init__(
        self,
        filenames: list[os.PathLike],
        alphabet: str,
        transforms: list[nn.Module] | None = None,
        is_train: bool = True,
    ):
        """Image dataset instance.

        Args:
            - filenames: images filenames
            - alphabet: allowed alphabet
            - transforms: transforms to apply to initial images
            - is_train: training dataset flag
        """
        super().__init__()
        self.alphabet = alphabet
        self.filenames = filenames
        self.transforms = transforms
        self.labels = self.get_labels(self.filenames) if is_train else None

    def _check_label(self, label: str) -> bool:
        """Check if label is in correct format.

        Args:
            - label: label to check

        Returns:
            True if label is in correct format
        """
        n_digits = sum(map(str.isdigit, label))
        n_letters = sum(map(str.isalpha, label))
        return n_digits + n_letters == len(label) and n_letters == 3

    def validate_labels(
        self, labels: list[str], raise_wrong_format: bool = False
    ) -> list[str]:
        """Validate if licence plate format is allowed.

        Args:
            - labels: initial licence plates numbers
            - raise_wrong_type: raise error for incorrect format or not

        Returns:
            List of labels in correct format
        """
        correct_labels = []
        for idx, label in enumerate(labels):
            if self._check_label(label):
                continue

            self.filenames.pop(idx)
            if raise_wrong_format:
                raise ValueError("wrong licence plate - %s", label)
            LOGGER.info("wrong licence plate - %s", label)
        return correct_labels

    def get_labels(self, filenames: list[str]) -> list[str]:
        """Get final labels from image filenames.

        Image filename should be in format *_<label>.png

        Args:
            - filenames: filenames of input images

        Returns:
            Formatted labels
        """
        labels = list(map(lambda x: x.stem.split("_")[-1], filenames))
        return self.validate_labels(labels)

    def __len__(self) -> int:
        """Returns length of final dataset."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        """Returns dataset's element by index.

        Args:
            - idx: index of element

        Returns:
            Dict with keys "image", "seq", "seq_len" & "text"
            Image is a numpy array, float32, [0, 1].
            Seq is list of integers.
            Seq_len is an integer.
            Text is a string.
        """
        image = cv2.imread(str(self.filenames[idx])).astype(np.float32) / 255.0

        if self.labels:
            label = self.labels[idx]
            seq = self.text_to_seq(label)
            seq_len = len(seq)
        else:
            label, seq, seq_len = "", tuple(), 0

        output = dict(image=image, seq=seq, seq_len=seq_len, label=label)
        if self.transforms:
            output = self.transforms(output)
        return output

    def text_to_seq(self, text: str) -> list[int]:
        """Encode text to sequence of integers.

        Args:
            - text: input text

        Returns:
            List of integers where each number is index of corresponding characted in alphabet + 1.
        """

        seq = [self.alphabet.find(c) + 1 for c in text]

        return seq

    @staticmethod
    def collate_fn(batch):
        """Function for torch.utils.data.Dataloader for batch collecting.

        Args:
            - batch: List of dataset __getitem__ return values (dicts).

        Returns:
            Dict with same keys but values are either torch.Tensors of batched images or sequences or so.
        """
        images, seqs, seq_lens, texts = [], [], [], []
        for item in batch:
            images.append(torch.from_numpy(item["image"]).permute(2, 0, 1).float())
            seqs.extend(item["seq"])
            seq_lens.append(item["seq_len"])
            texts.append(item["text"])
        images = torch.stack(images)
        seqs = torch.Tensor(seqs).int()
        seq_lens = torch.Tensor(seq_lens).int()
        batch = {"image": images, "seq": seqs, "seq_len": seq_lens, "text": texts}
        return batch


class RecognitionDataModule(L.LightningDataModule):
    """Dataset implementation using Lightning."""

    def __init__(
        self,
        train_path: os.PathLike,
        test_path: os.PathLike,
        alphabet: str,
        batch_size: int,
        num_workers: int,
        train_frac: float = 0.8,
        transforms: list[nn.Module] | None = None,
    ):
        """Image dataset instance.

        Args:
            - train_path: path to folder with train images
            - test_path: path to folder with test images
            - alphabet: allowed alphabet
            - batch_size: batch size for dataloaders
            - num_workers: number of workers for dataloaders
            - train_frac: fraction of data to train
            - transforms: transforms to apply to initial images
        """
        super().__init__()
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms

        train_images = [*train_path.glob("**/*.png")]
        self.train_images, self.val_images = train_test_split(
            train_images, train_size=train_frac, shuffle=True
        )
        self.test_images = [*test_path.glob("**/*.png")]

    def setup(self, stage: str):
        """Setup dataset for stages (fit, predict)."""
        if stage == "fit":
            self.train_ds = RecognitionDataset(
                self.train_images, alphabet=self.alphabet, transforms=self.transforms
            )
            self.validation_ds = RecognitionDataset(
                self.val_images, alphabet=self.alphabet, transforms=self.transforms
            )
        elif stage == "predict":
            self.test_ds = RecognitionDataset(
                self.test_images,
                alphabet=self.alphabet,
                transforms=self.transforms,
                is_train=False,
            )
        else:
            raise NotImplementedError(f"stage {stage} is not implemented")

    def train_dataloader(self):
        """Get train dataloader."""
        return torch.utils.data.DataLoader(
            self.train_ds,
            collate_fn=RecognitionDataset.collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return torch.utils.data.DataLoader(
            self.validation_ds,
            collate_fn=RecognitionDataset.collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        """Get predict dataloader."""
        return torch.utils.data.DataLoader(
            self.test_ds,
            collate_fn=RecognitionDataset.collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
