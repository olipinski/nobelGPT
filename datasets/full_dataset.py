import os

import requests
from torch.utils.data import Dataset

from datasets.utils import get_authors_books


class FullDataset(Dataset):
    def __init__(self):
        super().__init__()

        api = "https://wolnelektury.pl/api/"

        author_list = requests.get(api + "authors/")
        author_slugs = []
        for author in author_list.json():
            slug = author["slug"]
            author_slugs.append(slug)

        full_path = os.path.realpath(__file__)
        path = os.path.split(os.path.split(full_path)[0])[0]
        path = os.path.join(path, "../data")

        get_authors_books(authors=author_slugs, data_dir=path)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
