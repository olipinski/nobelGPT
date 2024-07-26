import os
import re
from os.path import isfile
from typing import Union


def process_raw_text(text_path: Union[os.PathLike, str]) -> str:
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

    return full_text
