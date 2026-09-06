#!/usr/bin/env python3
"""Rewrite TeX-style maths delimiters into the dollar style Quarto actually parses.

Quarto's markdown (pandoc `tex_math_dollars`) does **not** read ``\\( ... \\)`` or
``\\[ ... \\]`` — chat exports are full of them, and they silently render as plain
text with the backslashes eaten. This script converts, outside fenced code blocks::

    \\( x^2 \\)      ->  $x^2$
    \\[ a = b \\]    ->  $$\\na = b\\n$$

Usage::

    python3 scripts/normalize_math.py talking/**/*.qmd      # in place
    python3 scripts/normalize_math.py --check file.qmd      # report only
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.S)
INLINE = re.compile(r"\\\((.+?)\\\)", re.S)


def convert(text: str) -> tuple[str, int]:
    """Return (new_text, n_changes), leaving ``` fenced blocks untouched."""
    chunks = re.split(r"(?m)^(```[^\n]*$)", text)
    inside = False
    total = 0
    for i, chunk in enumerate(chunks):
        if chunk.startswith("```"):
            inside = not inside
            continue
        if inside:
            continue
        new, n1 = DISPLAY.subn(lambda m: "$$\n" + m.group(1).strip() + "\n$$", chunk)
        new, n2 = INLINE.subn(lambda m: "$" + m.group(1).strip() + "$", new)
        chunks[i] = new
        total += n1 + n2
    return "".join(chunks), total


def main(argv: list[str]) -> int:
    check = "--check" in argv
    paths = [Path(a) for a in argv if not a.startswith("--")]
    if not paths:
        print(__doc__)
        return 2
    changed = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        new, n = convert(text)
        if n:
            changed += n
            print(f"{path}: {n} delimiter pair(s)")
            if not check:
                path.write_text(new, encoding="utf-8")
    print("nothing to do" if changed == 0 else f"{changed} pair(s) converted")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
