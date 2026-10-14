# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check paper sources and attachments before rendering."""

from pathlib import Path

import regex as re

PAPER = Path(__file__).resolve().parent
COMMENT = re.compile(r"(?<!\\)%.*")
REFERENCES = re.compile(
    r"\\(input|include|includegraphics)(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}"
)
TABLE = re.compile(r"\\begin\{(?:tabularx?\*?|longtable)\}")
NAN = re.compile(r"(?<![A-Za-z])nan(?![A-Za-z])", re.IGNORECASE)


def search_dirs(root):
    directories = [root]
    for folder in ("figures", "styles"):
        path = root / folder
        directories.append(path)
        directories.extend(p for p in path.rglob("*") if p.is_dir())
    return directories


def checks(root=PAPER):
    """Return errors for source references, attachments, and table contents."""
    errors, referenced = [], set()
    sources = [
        p
        for p in root.rglob("*")
        if p.suffix in (".tex", ".sty") and "build" not in p.relative_to(root).parts
    ]
    for src in sources:
        text = COMMENT.sub("", src.read_text())
        relative = src.relative_to(root)
        if "tables" not in relative.parts and TABLE.search(text):
            errors.append(f"{relative}: move raw tables into tables/")
        for line in text.splitlines():
            if "&" in line and NAN.search(line):
                errors.append(f"{relative}: NaN in table row")
        for kind, name in REFERENCES.findall(text):
            suffixes = (
                ("", ".pdf", ".png", ".jpg", ".jpeg")
                if kind == "includegraphics"
                else ("", ".tex")
            )
            candidates = [
                base / (name + suffix)
                for base in search_dirs(root)
                for suffix in suffixes
            ]
            matches = [p for p in candidates if p.is_file()]
            if not matches:
                errors.append(f"{relative}: missing {kind} reference {name}")
            referenced.update(p.resolve() for p in matches)
        for name in re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", text):
            for package in name.split(","):
                for base in search_dirs(root):
                    path = base / (package + ".sty")
                    if path.exists():
                        referenced.add(path.resolve())
    attachments = [
        p
        for folder in ("figures", "styles")
        for p in (root / folder).rglob("*")
        if p.is_file()
    ]
    names = [p.name for p in attachments]
    for path in attachments:
        if names.count(path.name) > 1:
            errors.append(f"{path.relative_to(root)}: duplicate attachment filename")
        if path.resolve() not in referenced:
            errors.append(f"{path.relative_to(root)}: unused attachment")
    if not (root / "article.tex").is_file():
        errors.append("Missing article.tex")
    return errors


if __name__ == "__main__":
    errors = checks()
    print("\n".join(errors) if errors else "Paper checks passed")
    raise SystemExit(bool(errors))
