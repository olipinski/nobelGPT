import logging
import os
import re

import numpy as np
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from datasets.utils import get_authors_books
from utils.file_utils import create_if_not_exist

logger = logging.getLogger(__name__)


class NobelDataset(Dataset):
    def __init__(
        self,
        tokeniser: Tokenizer,
        download=False,
        seq_len: int = 2048,
    ):
        super().__init__()
        self.authors = ("henryk-sienkiewicz", "wladyslaw-stanislaw-reymont")
        self.tokeniser = tokeniser

        self.tokens = None
        self.mask = None
        self.seq_len = seq_len

        full_path = os.path.realpath(__file__)
        path = os.path.split(os.path.split(full_path)[0])[0]
        data_path = os.path.join(path, "./data")
        self.text_path = os.path.join(path, "data", "raw_txt")
        self.processed_path = os.path.join(path, "data", "tokenised")
        self.tokens_path = os.path.join(self.processed_path, "tokens.npy")
        self.masks_path = os.path.join(self.processed_path, "mask.npy")

        if download:
            get_authors_books(authors=self.authors, data_dir=data_path)
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
            (len(self.tokens) - (len(self.tokens) % self.seq_len)) / self.seq_len
        )

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if idx >= self.dataset_length:
            raise IndexError("Dataset index out of range!")
        tokens = self.tokens[idx * self.seq_len : (idx + 1) * self.seq_len]
        labels = self.tokens[idx * self.seq_len + 1 : (idx + 1) * self.seq_len + 1]
        masks = self.mask[idx * self.seq_len : (idx + 1) * self.seq_len]
        return tokens, labels, masks

    def __prepare_dataset(self):
        text_files = [
            f
            for f in os.listdir(self.text_path)
            if os.path.isfile(os.path.join(self.text_path, f))
        ]

        full_text = ""

        for file in text_files:
            f = open(os.path.join(self.text_path, file))
            # remove the top and bottom parts
            text = f.read()

            up_to_word = "\n\n\n"
            rx_to_first = r"^.*?{}".format(re.escape(up_to_word))

            res = re.sub(rx_to_first, "", text, flags=re.DOTALL).strip()
            res = re.sub(r"-----.*", "", res, flags=re.DOTALL).strip()

            full_text += res

            f.close()

        tokenised = self.tokeniser.encode(full_text)
        create_if_not_exist(self.processed_path)
        tokens = np.array(tokenised.ids, dtype=np.uint16)
        mask = np.array(tokenised.attention_mask, dtype=np.uint16)

        # Save the processed tokens
        np.save(self.tokens_path, tokens)
        np.save(self.masks_path, mask)

        self.tokens = tokens
        self.mask = mask
