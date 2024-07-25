import logging
import os
import re
from os.path import isfile

import nltk
from torch.utils.data import Dataset

from datasets.utils import get_authors_books
from tokeniser import NobelGPTTokeniser

logger = logging.getLogger(__name__)


class NobelDataset(Dataset):
    def __init__(
        self, tokeniser: NobelGPTTokeniser, sentence_split=True, download=False
    ):
        super().__init__()
        self.authors = ("henryk-sienkiewicz", "wladyslaw-stanislaw-reymont")
        full_path = os.path.realpath(__file__)
        path = os.path.split(os.path.split(full_path)[0])[0]
        self.tokeniser = tokeniser.get_tokeniser()
        self.data_path = os.path.join(path, "./data")
        self.sentence_split = sentence_split
        if download:
            get_authors_books(authors=self.authors, data_dir=self.data_path)
        self.full_text = self.__prepare_dataset()

    def __len__(self):
        return len(self.full_text)

    def __getitem__(self, idx):
        encoded = self.tokeniser.encode(self.full_text[idx])
        text = encoded.ids
        mask = encoded.attention_mask
        return text[:-1], [0], [text[-1]], mask

    def __getitems__(self, indices):
        encoded = self.tokeniser.encode_batch([self.full_text[idx] for idx in indices])
        texts = [t.ids[:-1] for t in encoded]
        masks = [t.attention_mask[:-1] for t in encoded]
        labels = [t.ids[-1] for t in encoded]
        return texts, [0 for _ in indices], labels, masks

    def __prepare_dataset(self):
        text_path = os.path.join(self.data_path, "raw_txt")
        text_files = [
            f for f in os.listdir(text_path) if isfile(os.path.join(text_path, f))
        ]

        full_text = ""

        for file in text_files:
            f = open(os.path.join(text_path, file))
            # remove the top and bottom parts
            text = f.read()

            up_to_word = "\n\n\n"
            rx_to_first = r"^.*?{}".format(re.escape(up_to_word))

            res = re.sub(rx_to_first, "", text, flags=re.DOTALL).strip()
            res = re.sub(r"-----.*", "", res, flags=re.DOTALL).strip()

            full_text += res

            f.close()

        if self.sentence_split:
            full_text = nltk.tokenize.sent_tokenize(full_text, language="polish")

        return full_text
