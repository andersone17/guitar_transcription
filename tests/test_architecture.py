"""Architecture guards: each package imports only the stdlib and the packages it may depend on."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import guitar_transcription

PACKAGE_DIR = Path(guitar_transcription.__file__).parent

# package -> project packages it may import (itself included). No third-party imports allowed.
ALLOWED = {
    "domain": {"domain"},
    "guitar": {"domain", "guitar"},
    "audio": {"domain", "guitar", "audio"},
    "rhythm": {"domain", "rhythm"},
    "notation": {"domain", "guitar", "rhythm", "notation"},
}

# The only files allowed to import a given third-party package (backend adapters).
THIRD_PARTY_OWNERS = {
    "basic_pitch": {"audio/backends/basic_pitch.py"},
    "music21": {"notation/musicxml.py"},
}


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add("." if node.level else node.module or "")
    return modules


def is_allowed(module: str, allowed_packages: set[str]) -> bool:
    if module == "." or module.split(".")[0] in sys.stdlib_module_names:
        return True
    parts = module.split(".")
    return parts[0] == "guitar_transcription" and len(parts) > 1 and parts[1] in allowed_packages


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_package_imports_only_allowed_modules(package: str) -> None:
    sources = sorted((PACKAGE_DIR / package).rglob("*.py"))
    assert sources, f"no sources found for {package}"

    violations = {
        f"{path.name}: {module}"
        for path in sources
        for module in imported_modules(path)
        if not is_allowed(module, ALLOWED[package])
        and not is_owned_third_party(module, path.relative_to(PACKAGE_DIR).as_posix())
    }

    assert not violations


def is_owned_third_party(module: str, relative_path: str) -> bool:
    return relative_path in THIRD_PARTY_OWNERS.get(module.split(".")[0], set())


@pytest.mark.parametrize("third_party", sorted(THIRD_PARTY_OWNERS))
def test_backend_packages_are_imported_only_by_their_adapter(third_party: str) -> None:
    importers = {
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in PACKAGE_DIR.rglob("*.py")
        for module in imported_modules(path) | dynamically_imported(path)
        if module.split(".")[0] == third_party
    }

    assert importers <= THIRD_PARTY_OWNERS[third_party]


def dynamically_imported(path: Path) -> set[str]:
    """String literals passed to ``importlib.import_module`` (used for lazy backend imports)."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            modules.add(node.args[0].value)
    return modules


def test_importing_audio_and_adapter_does_not_import_backend() -> None:
    # Subprocess: the backend may already be loaded in this interpreter by other tests.
    code = (
        "import sys, guitar_transcription.audio, guitar_transcription.audio.backends.basic_pitch; "
        "heavy = ('basic_pitch', 'tensorflow', 'numpy', 'librosa'); "
        "loaded = [m for m in heavy if m in sys.modules]; "
        "assert not loaded, loaded"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("module", "allowed"),
    [
        ("math", True),
        ("guitar_transcription.domain.pitch", True),
        ("guitar_transcription.guitar", False),
        ("guitar_transcription", False),
        ("music21", False),
        ("basic_pitch", False),
        ("numpy", False),
    ],
)
def test_guard_classifies_imports(module: str, allowed: bool) -> None:
    assert is_allowed(module, {"domain"}) is allowed


def test_cli_imports_music21_only_when_writing_notation() -> None:
    # music21 takes ~1 s to import; plain transcription and --help shouldn't pay for it.
    code = (
        "import sys, guitar_transcription.cli, guitar_transcription.notation; "
        "assert 'music21' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
