import os

from tokenizers import Tokenizer, normalizers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from utils.text_utils import process_raw_text


def train_tokeniser(
    data_path,
    text_path,
    vocab_size=32768,
    unk_token="[UNK]",
    special_tokens=["[UNK]", "[PAD]", "[MASK]", "[RAND]"],
    progress=False,
) -> Tokenizer:
    """
    Train a Byte-Pair Encoding (BPE) tokeniser on the provided text data.

    Parameters
    ----------
    data_path : str
        Path to save the trained tokeniser file (tokenizer-ngpt.json).
    text_path : str
        Path to the raw text file used for training the tokeniser.
    vocab_size : int, optional
        The size of the tokeniser vocabulary (default is 32768).
    unk_token : str, optional
        The token to represent unknown words (default is "[UNK]").
    special_tokens : list of str, optional
        List of special tokens to include in the tokeniser, such as
        unknown, padding, mask, and random tokens (default is ["[UNK]", "[PAD]", "[MASK]", "[RAND]"]).
    progress : bool, optional
        Whether to display training progress (default is False).

    Returns
    -------
    Tokenizer
        A trained Tokenizer instance.
    """
    full_text = process_raw_text(text_path)

    tokeniser = Tokenizer(BPE(unk_token=unk_token))
    trainer = BpeTrainer(
        show_progress=progress,
        vocab_size=vocab_size,
        special_tokens=special_tokens + ["[CLS]", "[SEP]"],
    )

    tokeniser.normalizer = normalizers.Sequence(
        [normalizers.Strip(), normalizers.NFKC()]
    )
    tokeniser.pre_tokenizer = ByteLevel()

    tokeniser.train_from_iterator(full_text, trainer=trainer, length=len(full_text))

    tokeniser.save(os.path.join(data_path, "tokeniser-ngpt.json"))

    tokeniser.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokeniser.token_to_id("[CLS]")),
            ("[SEP]", tokeniser.token_to_id("[SEP]")),
        ],
    )
    tokeniser.decoder = ByteLevelDecoder()

    return tokeniser
