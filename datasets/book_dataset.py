import logging
import os

import numpy as np
import requests
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from datasets.utils import get_authors_books
from utils.file_utils import create_if_not_exist
from utils.text_utils import process_raw_text

logger = logging.getLogger(__name__)


class BookDataset(Dataset):
    def __init__(
        self,
        tokeniser: Tokenizer,
        download: bool = False,
        progress: bool = False,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        # if tokeniser is None:
        #     raise ValueError("A tokeniser must be provided.")
        self.tokeniser = tokeniser

        self.tokens = None
        self.mask = None
        self.max_seq_len = max_seq_len

        full_path = os.path.realpath(__file__)
        path = os.path.split(os.path.split(full_path)[0])[0]
        data_path = os.path.join(path, "data", "book")
        self.text_path = os.path.join(data_path, "raw_txt")
        self.processed_path = os.path.join(data_path, "tokenised")
        self.tokens_path = os.path.join(self.processed_path, "tokens.npy")
        self.masks_path = os.path.join(self.processed_path, "mask.npy")

        if download:
            api = "https://wolnelektury.pl/api/"

            author_list = requests.get(api + "authors/")
            author_slugs = []
            for author in author_list.json():
                slug = author["slug"]
                author_slugs.append(slug)

            get_authors_books(authors=author_slugs, data_dir=path, progress=progress)
        else:
            if (
                not os.path.exists(self.text_path)
                or len(os.listdir(self.text_path)) == 0
            ):
                raise FileNotFoundError(
                    "No dataset found, please run with download=True at least once."
                )

        if (
            not os.path.exists(self.processed_path)
            or len(os.listdir(self.processed_path)) == 0
        ):
            # Tokenise and load dataset
            self.__prepare_dataset()
        else:
            # Load tokenised dataset
            self.tokens = np.load(self.tokens_path)
            self.mask = np.load(self.masks_path)

        # Divide the dataset based on sequence lengths
        self.dataset_length = int(
            (len(self.tokens) - (len(self.tokens) % self.max_seq_len))
            / self.max_seq_len
        )

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if idx >= self.dataset_length:
            raise IndexError("Dataset index out of range!")
        tokens = self.tokens[idx * self.max_seq_len : (idx + 1) * self.max_seq_len]
        labels = self.tokens[
            idx * self.max_seq_len + 1 : (idx + 1) * self.max_seq_len + 1
        ]
        masks = self.mask[idx * self.max_seq_len : (idx + 1) * self.max_seq_len]
        return tokens, labels, masks

    def __prepare_dataset(self):
        full_text = process_raw_text(self.text_path)

        tokenised = self.tokeniser.encode(full_text)
        create_if_not_exist(self.processed_path)
        tokens = np.array(tokenised.ids, dtype=np.uint16)
        mask = np.array(tokenised.attention_mask, dtype=np.uint16)

        # Save the processed tokens
        np.save(self.tokens_path, tokens)
        np.save(self.masks_path, mask)

        self.tokens = tokens
        self.mask = mask
