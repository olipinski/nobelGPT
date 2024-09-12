# nobelGPT

nobelgPT provides a way to train a llama/GPT-2/3 based model on Polish books.
The name comes from the fact that the initial dataset used was only books by
Polish nobel laurates.

The development of the nobelGPT model was greatly helped, and inpired by, Andrej
Karpathy's videos on the GPT and GPT-2 architectures. Check out his channel
[here](https://www.youtube.com/andrejkarpathy)! It is an amazing resource to learn more about LLMs.

## Issues

Below are the few issues that are unresolved yet. They do not break the model,
but make it slightly more annoying to train.

- Sync dist is broken for self.log when used with torch.compile
- Generation callback will not work with the FSPD strategy

## Downloading the datasets

*Please be mindful of the Wolne Lektury bandwidth!*

The datasets will be downloaded, if the parameter `download` is set to `True`.

If you opt to download the whole book dataset, consider donating to Wolne
Lektury [here](https://wolnelektury.pl/pomagam/).

## Future Improvements

Below are a few ideas for improvements either in terms of training speed or
usefulness of the model.

- Pre-tokenised dataset could speed up the training.
- End-of-text token should probably be used for the different texts in the
  datasets
- In generate.py, the model should decide when to stop text generation, instead
  of a pre-set number of tokens.
- Automatically detect when tokeniser was changed. Currently, when changing
  tokeniser the processed folder must be deleted manually.

## Additional Software

The software below could be useful if processing was expanded to epub and/or
pdf.

https://github.com/aerkalov/ebooklib

https://github.com/py-pdf/pypdf

## Licenses

All texts in the datasets directory are licensed under the CC-BY
[here](https://creativecommons.org/licenses/by/4.0) or the Free Art License
[here](https://artlibre.org/licence/lal/en/).

All texts are courtesy of Wolne Lektury Foundation (https://wolnelektury.pl)

![Wolne Lektury](assets/WolneLektury.svg)
