# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The released versions below are the git tags; everything that happened after
the last tag is the unreleased or current version. The `0.1.0` and `0.1.1`
numbers that appeared in `setup.py` along the way were never tagged and are
folded into `0.0.4`.

## [0.0.4] - 2026-11-22

### Changed

- LangChain is no longer a dependency. Loaders read documents directly:
  `pypdf` for PDF, BeautifulSoup for HTML, `python-docx` for DOCX, and plain
  file reading with encoding autodetection for TXT and Markdown.
- Every loader returns `DocumentPage` objects (`page_content` plus a
  `metadata` dictionary), also for DOCX and input text, so no
  document-loading framework is needed to use the library.
- Heavy dependencies are optional. `transformers`, `torch`, `datasets`,
  `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `spacy`,
  `fasttext-langdetect`, `deepl`, `pandas` and `openpyxl` moved to the `ml`,
  `spacy`, `lang` and `tables` extras (and `all`). The core installs `tqdm`,
  `requests`, `beautifulsoup4`, `python-docx` and `pypdf` only.
- Heavy imports are lazy: `transformers` inside
  `radlab_data.text.loaders.loader_i.get_tokenizer()` (first use of token
  chunking), `spacy` inside `radlab_data.text.processors.splitter.get_nlp()`,
  and `ftlangdetect` / `deepl` inside the `TextUtils` methods which use them.
  Importing `Document`, `DirectoryFileReader`, the loaders or `TextUtils` no
  longer pulls in the model stack.
- Packaging moved from `setup.py` to `pyproject.toml` (PEP 621), which also
  fixes built distributions missing the top-level `radlab_data` package. The
  distribution is named `radlab-data` (the import name stays `radlab_data`)
  and its version is read from a `.version` file.
- `MANIFEST.in` controls the sdist contents: README, CHANGELOG, LICENSE,
  `requirements.txt`, `pyproject.toml` and `.version` are included, `tests`
  and build artefacts are excluded.
- `requirements.txt` is split into core, extras and a dev/packaging section,
  and `Levenshtein` is gone.
- HF training arguments use `eval_strategy` instead of the deprecated
  `evaluation_strategy`.
- Declared `python_requires=">=3.10"`.

### Added

- A `DocumentPage` model and text file reading helpers.
- `html` in the supported file extensions of the loader config.
- A test suite for the loaders, readers, datasets, preprocessing pipeline and
  utilities, including a test that imports the document core in a subprocess
  where `torch`, `transformers` and `spacy` raise `ImportError`.
- `pytest` as a development dependency, and packaging tests that keep
  `requirements.txt`, `.version` and `MANIFEST.in` consistent with
  `pyproject.toml`.
- Classifiers, a `Homepage` URL and a README rendered on PyPI.

### Fixed

- Loaders crashing when no `options` are given.
- Loader state mutation and directory reading issues.
- Document pairs and preprocessing dataset loading.
- Dataset loading and pipeline element edge cases.
- Errors in `radlab_data.utils`; the spaCy model now loads lazily.
- The label histogram with an empty list of labels.
- `tests/test_loaders.py::test_get_tokenizer_is_cached` patches
  `transformers.AutoTokenizer` instead of the module-level name which no
  longer exists in `loader_i`.

## [0.0.3] - 2026-10-14

### Added

- Markdown support: a `MarkdownLoader` (with an optional `strip_markdown`
  option) and README updates.

## [0.0.2] - 2026-10-14

The state captured by the first tag.

### Added

- Text loading framework: the `LoaderI` base class, the PDF, TXT, DOCX and
  input-text loaders, and the loader configuration.
- Text processing utilities: cleaner, splitter and mappers, plus regex
  definitions.
- HTML loading and cleaning.
- Dataset classes for classification, sequence labelling and sequence to
  sequence tasks, the preprocessing pipeline and the training utilities.
- Apache 2.0 license.
