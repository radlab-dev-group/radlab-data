"""Keep requirements.txt in sync with pyproject.toml.

requirements.txt is the full development environment (all extras plus
development tools); pyproject.toml declares what pip actually installs.
This test fails if the two drift apart.
"""

from pathlib import Path

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # Python 3.10
    import tomli as _toml

REQUIREMENTS_PATH = Path(__file__).parent.parent / "requirements.txt"
PYPROJECT_PATH = Path(__file__).parent.parent / "pyproject.toml"
VERSION_FILE = Path(__file__).parent.parent / ".version"

# requirements.txt is also the development environment; these packages are
# tooling rather than something the distribution declares.
DEV_PACKAGES = {"pytest", "setuptools", "build", "tomli"}


def version_file_content() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def _requirements_txt_packages() -> set[str]:
    packages = set()
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        name = line.split(";", 1)[0].split("==", 1)[0].split(">=", 1)[0].split("~=", 1)[0]
        packages.add(name.split("[", 1)[0].strip().lower())
    return packages


def _pyproject_packages() -> set[str]:
    with open(PYPROJECT_PATH, "rb") as fh:
        project = _toml.load(fh)["project"]
    packages = {dep.split("<")[0].split(">")[0].strip().lower() for dep in project["dependencies"]}
    extras = project.get("optional-dependencies", {})
    for group in ("core", "ml", "spacy", "lang", "tables"):
        if group in extras:
            packages.update(dep.split("<")[0].split(">")[0].strip().lower() for dep in extras[group])
    return packages


def test_requirements_txt_covers_every_pyproject_package():
    missing = _pyproject_packages() - _requirements_txt_packages()
    assert not missing, f"requirements.txt is missing packages declared in pyproject.toml: {sorted(missing)}"


def test_pyproject_covers_every_non_dev_package_from_requirements_txt():
    extra = _requirements_txt_packages() - _pyproject_packages() - DEV_PACKAGES
    assert not extra, f"requirements.txt lists packages not declared in pyproject.toml: {sorted(extra)}"


def test_version_file_holds_a_valid_release_version():
    from packaging.version import Version

    assert Version(version_file_content())


def test_version_is_loaded_from_the_version_file():
    with open(PYPROJECT_PATH, "rb") as fh:
        dynamic = _toml.load(fh)["tool"]["setuptools"]["dynamic"]

    assert dynamic["version"] == {"file": VERSION_FILE.name}


def test_version_file_is_shipped_in_the_sdist():
    manifest = (PYPROJECT_PATH.parent / "MANIFEST.in").read_text(encoding="utf-8")
    included = {
        line.split(maxsplit=1)[1]
        for line in manifest.splitlines()
        if line.startswith("include ")
    }

    assert VERSION_FILE.name in included
