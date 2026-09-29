"""Keep editable installs usable with Python 3.10 -I in a Korean Windows path.

uv's interpreter probe ignores PYTHONUTF8. Plain UTF-8 path lines in .pth files
then fail under the CP949 locale. ASCII Python import lines avoid either locale.
Only this repository's virtual environments are eligible for the conversion.
"""

import argparse
import json
from pathlib import Path


def fix(environment: Path):
    root = Path(__file__).resolve().parents[1]
    environment = environment.resolve()
    if root not in environment.parents:
        raise ValueError("Only a virtual environment inside this workspace may be changed")
    for path in (environment / "Lib/site-packages").glob("*.pth"):
        original = path.read_text(encoding="utf-8-sig")
        lines = []
        changed = False
        for line in original.splitlines():
            if line and not line.startswith(("#", "import ", "import\t")) and not line.isascii():
                target = Path(line)
                if not target.is_absolute():
                    target = path.parent / target
                lines.append("import sys; sys.path.append(" + json.dumps(str(target), ensure_ascii=True) + ")")
                changed = True
            else:
                lines.append(line)
        if changed:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"Normalized editable path: {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", type=Path)
    fix(parser.parse_args().environment)
