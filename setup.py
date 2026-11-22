from setuptools import setup

# The package is split into a light core and optional extras.
#
# Core: document loading (txt/md/pdf/docx/html), text cleaning, directory
# readers and small utilities. Importing these modules never pulls in
# torch, transformers or spacy - the tokenizer is only loaded when token
# chunking is actually requested.
#
# Extras:
#   ml      - token chunking (transformers), HF datasets, training/metrics
#             (torch, scikit-learn, scipy, numpy, matplotlib)
#   spacy   - sentence splitting with the Polish spaCy model
#   lang    - language detection (fasttext) and machine translation (deepl)
#   tables  - pandas/openpyxl support used by dataset preprocessing
install_requires = [
    "tqdm",
    "requests",
    "beautifulsoup4",
    "python-docx",
    "pypdf",
]

extras_require = {
    "ml": [
        "transformers",
        "torch",
        "datasets",
        "numpy",
        "scikit-learn",
        "scipy",
        "matplotlib",
    ],
    "spacy": [
        "spacy",
    ],
    "lang": [
        "fasttext-langdetect",
        "deepl",
    ],
    "tables": [
        "pandas",
        "openpyxl",
    ],
    "all": [
        "transformers",
        "torch",
        "datasets",
        "numpy",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "spacy",
        "fasttext-langdetect",
        "deepl",
        "pandas",
        "openpyxl",
    ],
}

setup(
    name="radlab_data",
    version="0.1.1",
    description="Text processing and modelling",
    author="RadLab",
    author_email="hello@radlab.dev",
    packages=[
        "radlab_data.preprocessing",
        "radlab_data.datasets",
        "radlab_data.text",
        "radlab_data.text.loaders",
        "radlab_data.text.processors",
        "radlab_data.utils",
    ],
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.10",
)
