"""Architecture guard: ``domain`` imports only the standard library and itself."""

import ast
import sys
from pathlib import Path

import guitar_transcription.domain

DOMAIN_DIR = Path(guitar_transcription.domain.__file__).parent


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add("." if node.level else node.module or "")
    return modules


def is_allowed(module: str) -> bool:
    return (
        module == "."
        or module.split(".")[0] in sys.stdlib_module_names
        or module == "guitar_transcription.domain"
        or module.startswith("guitar_transcription.domain.")
    )


def test_domain_imports_only_stdlib_and_itself() -> None:
    sources = sorted(DOMAIN_DIR.rglob("*.py"))
    assert sources, "no domain sources found"

    violations = {
        f"{path.name}: {module}"
        for path in sources
        for module in imported_modules(path)
        if not is_allowed(module)
    }

    assert not violations
