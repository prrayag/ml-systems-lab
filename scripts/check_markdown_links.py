from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def markdown_files() -> list[Path]:
    return sorted([ROOT / "README.md", *ROOT.glob("docs/*.md"), ROOT / "results" / "summary.md"])


def local_target(raw_target: str) -> str | None:
    target = raw_target.split("#", 1)[0].strip()
    if not target or target.startswith(("http://", "https://", "mailto:")):
        return None
    return target


def main() -> None:
    missing = []
    for markdown_file in markdown_files():
        text = markdown_file.read_text()
        for match in LINK_RE.finditer(text):
            target = local_target(match.group(1))
            if target is None:
                continue
            path = (markdown_file.parent / target).resolve()
            if not path.exists():
                missing.append(f"{markdown_file.relative_to(ROOT)} -> {target}")

    if missing:
        raise SystemExit("missing markdown targets:\n" + "\n".join(missing))

    print("markdown links look complete")


if __name__ == "__main__":
    main()
