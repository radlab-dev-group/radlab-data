# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-11-22

### Changed

- Optional heavy dependencies. `transformers`, `torch`, `datasets`, `numpy`,
  `scikit-learn`, `scipy`, `matplotlib`, `spacy`, `fasttext-langdetect`,
  `deepl`, `pandas` and `openpyxl` are no longer install requirements of the
  whole package; they moved to the `ml`, `spacy`, `lang` and `tables` extras
  (and `all`). The core installs `tqdm`, `requests`, `beautifulsoup4`,
  `python-docx` and `pypdf` only.
- `radlab_data.text.loaders.loader_i.get_tokenizer()` imports `transformers`
  inside the function and `LoaderI` loads its tokenizer lazily, on the first
  use of token chunking. Importing `Document`, `DirectoryFileReader`, the
  loaders or `TextUtils` no longer requires the model stack, so a host which
  only reads documents (or only answers queries) does not have to install it.
- Declared `python_requires=">=3.10"`.

### Added

- `tests/test_reader.py::test_core_modules_import_without_the_model_stack`,
  which imports the document-loading core in a subprocess where
  `torch`, `transformers` and `spacy` raise `ImportError`.

### Fixed

- `tests/test_loaders.py::test_get_tokenizer_is_cached` patches
  `transformers.AutoTokenizer` instead of the module-level name which no
  longer exists in `loader_i`.

## [0.1.0]

- First packaged release: document loaders (PDF, DOCX, TXT, MD, HTML,
  input text), text processors and cleaners, directory reader, dataset
  classes for classification / sequence labelling / seq2seq, preprocessing
  pipeline and training utilities.
