import logging
import os
from pathlib import Path
from typing import List, Tuple, Union

import requests

from utils.file_utils import create_if_not_exist

logger = logging.getLogger(__name__)


def get_authors_books(authors: Union[List, Tuple], data_dir: Union[Path, str]) -> None:
    """
    Get all books for a list of authors.

    Currently, ignores all epubs and pdfs.

    Parameters
    ----------
    authors: List
        List of authors to get books for. Must use the slug format of wolnelektury API.
    data_dir: Path | str
        Directory to use to save the books.
    """
    api = "https://wolnelektury.pl/api/"
    book_hrefs = []

    # Make sure data_dir exists
    create_if_not_exist(data_dir)
    create_if_not_exist(os.path.join(data_dir, "raw_txt"))
    # create_if_not_exist(os.path.join(data_dir, "raw_pdf"))
    # create_if_not_exist(os.path.join(data_dir, "raw_epub"))

    for author in authors:
        response = requests.get(api + "authors/" + author + "/books")
        response = response.json()
        for book in response:
            book_hrefs.append(book["href"])

    for book in book_hrefs:
        book_title = book.split("/")[-2]
        response = requests.get(book)
        response = response.json()
        if response["txt"]:
            fname = os.path.join(data_dir, "raw_txt", f"{book_title}.txt")
            if os.path.isfile(fname):
                logger.debug(f"Book {book_title} already downloaded.")
                continue
            logger.debug(f"Downloading book {book_title}.")
            book_text = requests.get(response["txt"])
            with open(fname, "wb") as fd:
                for chunk in book_text.iter_content(chunk_size=128):
                    fd.write(chunk)
        elif response["epub"]:
            continue
            # fname = os.path.join(data_dir, "raw_epub", "{book_title}.epub")
            # if os.path.isfile(fname):
            #     continue
            # book_text = requests.get(response["epub"])
            # with open(os.path.join(fname), "wb") as fd:
            #     for chunk in book_text.iter_content(chunk_size=128):
            #         fd.write(chunk)
        elif response["pdf"]:
            continue
            # fname = os.path.join(data_dir, "raw_pdf", "{book_title}.pdf")
            # if os.path.isfile(fname):
            #     continue
            # book_text = requests.get(response["pdf"])
            # with open(os.path.join(fname), "wb") as fd:
            #     for chunk in book_text.iter_content(chunk_size=128):
            #         fd.write(chunk)
        else:
            print(f"Cannot get {book_title}. No parsable formats available")
