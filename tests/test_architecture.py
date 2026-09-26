"""Architecture guards: each package imports only the stdlib and the packages it may depend on."""

import ast
import sys
from pathlib import Path

import pytest

import guitar_transcription

PACKAGE_DIR = Path(guitar_transcription.__file__).parent

# package -> project packages it may import (itself included). No third-party imports allowed.
ALLOWED = {
    "domain": {"domain"},
    "guitar": {"domain", "guitar"},
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
    }

    assert not violations


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
