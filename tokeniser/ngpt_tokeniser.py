import os
import re
from os.path import isfile

import nltk
from tokenizers import Tokenizer, normalizers
from tokenizers.decoders import BPEDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


class NobelGPTTokeniser:
    def __init__(
        self,
        sentence_tokenised: bool,
        pretrained: str = None,
        data_path: str = None,
    ):
        self.sentence_tokenised = sentence_tokenised
        if pretrained is None:
            self.__train(data_path)
        else:
            self.tokeniser = Tokenizer.from_file(
                os.path.join(pretrained, "tokeniser-ngpt.json")
            )

    def get_tokeniser(self):
        return self.tokeniser

    def __train(self, data_path):
        sentence_tokenised = True

        text_path = os.path.join(data_path, "raw_txt")

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

        if sentence_tokenised:
            full_text = nltk.tokenize.sent_tokenize(full_text, language="polish")

        tokeniser = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            vocab_size=8000,
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

        self.tokeniser = tokeniser
