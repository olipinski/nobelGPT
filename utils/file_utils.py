import logging
import os
import time
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def create_if_not_exist(new_dir: Union[Path, str]) -> None:
    if not os.path.exists(new_dir):
        logger.debug(f"Creating directory {new_dir}.")
        os.makedirs(new_dir)
        # Sleep is used to make sure the directory is created
        time.sleep(1)
