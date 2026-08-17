from setuptools import setup

setup(
    name="radlab_data",
    version="0.2",
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
    install_requires=["spacy", "datasets", "tqdm"],
)
