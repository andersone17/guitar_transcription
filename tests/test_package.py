"""Smoke tests: the package and its Stage 1 subpackages import from the installed distribution."""

import importlib

import pytest

SUBPACKAGES = ["domain", "guitar", "audio", "notation"]


def test_package_imports() -> None:
    import guitar_transcription

    assert guitar_transcription.__name__ == "guitar_transcription"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(f"guitar_transcription.{name}")

    assert module.__doc__, f"guitar_transcription.{name} should document its responsibility"
