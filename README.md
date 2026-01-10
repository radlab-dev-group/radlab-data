# radlab‑data

A versatile Python library for loading, preprocessing, and handling a variety of textual data sources (PDF, DOCX, TXT,
and raw strings) and preparing them for downstream machine‑learning tasks such as classification, sequence labeling, and
generative modeling.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
    - [Dataset Types](#dataset-types)
    - [Text Loaders](#text-loaders)
    - [Processing Pipeline](#processing-pipeline)
- [Command‑Line Utilities](#command-line-utilities)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

`radlab-data` provides a unified interface for:

1. **Reading documents** from common file formats (PDF, DOCX, TXT) as well as plain‑text strings.
2. **Cleaning and normalising** raw text using a chain of configurable processors.
3. **Splitting** text into sentences, paragraphs, or token‑level chunks.
4. **Mapping** raw annotations (e.g., NER spans) onto tokenised representations.
5. **Packaging** data into HuggingFace `datasets.Dataset` objects, ready for training with `transformers`.

The library is built with extensibility in mind – you can plug in your own tokenizers, preprocessing steps, or custom
dataset classes.

---

## Features

- **Multi‑format document loading** – PDF, DOCX, TXT, and custom “input_text”.
- **Automatic language detection** (via `ftlangdetect`).
- **Flexible text cleaning** – phrase removal, abbreviation normalisation, regex‑based sanitisation, etc.
- **Sentence‑level splitting** using spaCy’s Polish model (`pl_core_news_lg`).
- **Special token handling** – conditional insertion of tokens based on metadata rules.
- **Dataset abstractions** for:
    - Text classification
    - Sequence labeling (NER, IOB tagging)
    - Sequence‑to‑sequence tasks (e.g., translation, summarisation)
    - Semantic similarity and dialogue alternatives
- **Caching** of tokenised datasets for fast reloads.
- **Multiprocessing support** for loading large corpora.
- **Command‑line tools** for dataset conversion, preprocessing, and training argument handling.

---

## Installation

```shell script
# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# Install radlab-data and its dependencies
pip install -e .   # editable install if you cloned the repo
```

> **Note**: The library requires Python 3.10+ and the following external packages (automatically installed via
`requirements.txt`):

- `torch`, `transformers`, `datasets`
- `spacy` (Polish large model – `python -m spacy download pl_core_news_lg`)
- `langchain`, `langchain‑community`, `ftlangdetect`, etc.

---

## Quick Start

```python
from radlab_data.text.document import Document
from radlab_data.datasets.dataset_loader import DatasetLoader
from transformers import AutoTokenizer

# 1️⃣ Load a document (PDF, DOCX, TXT, or raw text)
doc = Document(
    file_path="data/example.pdf",
    prepare_proper_pages=True,
    clear_texts=True,
    use_text_denoiser=False,
)
pages = doc.load()  # pages is a list of `langchain` Document objects

# 2️⃣ Initialise a tokenizer
tokenizer = AutoTokenizer.from_pretrained("radlab/polish-fast-tokenizer")

# 3️⃣ Load a dataset (JSON or JSONL) and automatically cache tokenisation
loader = DatasetLoader(
    dataset_path="data/my_dataset",
    tokenizer=tokenizer,
    split_train_valid_test="0.8:0.1",  # 80 % train, 10 % validation, 10 % test
    pre_shuffle=True,
)
dataset = loader.load()

# 4️⃣ Access train / validation / test splits
train = dataset.train_dataset
valid = dataset.valid_dataset
test = dataset.test_dataset
```

The resulting `train`, `valid`, and `test` objects are HuggingFace `datasets.Dataset` instances, ready to be passed to a
`transformers` trainer.

---

## Core Concepts

### Dataset Types

| Class                         | Typical Use‑Case                                | Key Characteristics                                               |
|-------------------------------|-------------------------------------------------|-------------------------------------------------------------------|
| `TextClassificationDataset`   | Single‑label classification                     | Stores `text` and a scalar `label`.                               |
| `SequenceLabellingDataset`    | NER / token‑level tagging                       | Generates IOB tags aligned to tokenised inputs.                   |
| `SequenceToSequenceDataset`   | Translation, summarisation, etc.                | Handles source‑target pairs (`text` ↔ `label`).                   |
| `SemanticSimilarityDataset`   | Pairwise similarity / ranking                   | Provides `sentence1`, `sentence2`, and a similarity `score`.      |
| `DPODataset`                  | Preference‑based learning (chosen vs. rejected) | Keeps raw prompts and two possible completions.                   |
| `DialogueAlternativesDataset` | Multi‑choice dialogue generation                | Stores a shared context and several possible endings with scores. |

All dataset classes inherit from `DatasetBaseClass`, which implements caching, optional shuffling, and split handling.

### Text Loaders

| Loader            | Supported Extension | Behaviour                                                |
|-------------------|---------------------|----------------------------------------------------------|
| `PDFLoader`       | `.pdf`              | Uses `langchain_community.document_loaders.PyPDFLoader`. |
| `TXTLoader`       | `.txt`              | Simple line‑by‑line read via `TextLoader`.               |
| `DOCXLoader`      | `.docx`             | Parses paragraphs and tables via `python-docx`.          |
| `InputTextLoader` | `.input_text`       | Wraps a raw string supplied via options.                 |

Loaders share a common base (`LoaderI`) that handles:

- Tokeniser initialisation,
- Optional language detection,
- Text denoising,
- Chunking into token‑size windows.

### Processing Pipeline

`radlab-data` ships with a lightweight pipeline framework:

```python
from radlab_data.preprocessing.pipeline import Pipeline
from radlab_data.preprocessing.pipeline_modules import SplitLabels, RemoveDuplicates

pipeline = Pipeline(modules=[
    SplitLabels(),
    RemoveDuplicates(),
    # add your custom PipelineElementBase subclasses here
])

processed_dataset = pipeline(original_dataset)
```

Each pipeline element inherits from `PipelineElementBase` and implements a `_process_data` method that receives a
dataset instance and returns the transformed instance. The framework automatically validates that the element supports
the dataset type.

---

## Command‑Line Utilities

The package includes a set of entry‑points for common tasks (exposed via `setup.cfg`/`pyproject.toml`):

| Command                  | Description                                                               |
|--------------------------|---------------------------------------------------------------------------|
| `radlab-data-preprocess` | Apply a predefined preprocessing pipeline to a raw dataset folder.        |
| `radlab-data-convert`    | Convert JSONL to cached HuggingFace format with optional tokeniser.       |
| `radlab-data-train-args` | Generate a JSON file with default `TrainingArguments` for `transformers`. |

Run `--help` on any command to see the full list of options.

---

## Configuration

All default settings live under `radlab_data/text/loaders/config.py`:

```python
TOKENIZER_PATH = "radlab/polish-fast-tokenizer"
AVAILABLE_FILE_EXTENSIONS = ["txt", "pdf", "docx", "input_text"]
```

You can override these values at runtime by passing the appropriate arguments to `Document` or the loader classes.

---

## License

`radlab-data` is released under the **Apache 2.0 License**. See the `LICENSE` file for full details.