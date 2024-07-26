import os
import re
from os.path import isfile

from tokenizers import Tokenizer, normalizers
from tokenizers.decoders import BPEDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from utils.text_utils import process_raw_text


def train_tokeniser(
    data_path,
    text_path,
    vocab_size=24000,
    unk_token="[UNK]",
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[RAND]"],
) -> Tokenizer:
    full_text = process_raw_text(text_path)

    tokeniser = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=24000,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[RAND]"],
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
    tokeniser.decoder = BPEDecoder()

    return tokeniser
