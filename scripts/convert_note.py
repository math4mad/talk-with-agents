#!/usr/bin/env python3
"""Convert a raw chat-log ``.md`` export from ``Talkmd/`` into a Quarto page body.

The raw exports are pasted out of an AI chat panel and share a few quirks:

* an ``<h1>`` wrapped in ``<span style=...>`` carrying the (very long) prompt,
* turn separators made of ``---`` / ``***`` rules,
* question lines prefixed with the human's handle (``Math4Mad:``),
* tab-separated tables pasted as plain text,
* pseudo-code written as bare lines, some of them starting with ``#`` (which
  would otherwise render as <h1> headings inside the page),
* zero-width spaces (U+200B) pasted inside display math.

Normalised layout used across the site::

    ## 1. <question, abbreviated -> becomes the TOC entry>

    ::: {.callout-note}
    **🗣️ Ask**
    > <verbatim question>
    :::

    <answer body: pipe tables, fenced python, headings demoted one level>

Usage::

    python3 scripts/convert_note.py "Talkmd/Jacobi-poly-nomial.md" out.md
    python3 scripts/convert_note.py Talkmd/x.md out.md --drop=9      # skip turns
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

TALKMD_DIR = Path(__file__).resolve().parent.parent / "Talkmd"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_math import convert as normalize_math  # noqa: E402

# The human's handle in a chat export. Its spelling drifts between tools and
# sessions (``Math4Mad:``, ``m4mad:`` …), so it is matched case-insensitively.
ASK_HANDLES = ("Math4Mad", "m4mad")
ASK_ALT = "|".join(ASK_HANDLES)
ANSWER_HANDLES = ("Ima.Copilot", "Copilot", "ChatGPT", "Claude", "Gemini")
stray_images: list[str] = []   # <img> tags lifted out of a question line
CJK = re.compile(r"[\u3400-\u9fff\u3040-\u30ff]")

# --------------------------------------------------------------------------- #
# text normalisation
# --------------------------------------------------------------------------- #


def clean(text: str) -> str:
    """Strip chat-export artefacts that break pandoc / Quarto."""
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"<span[^>]*>|</span>", "", text)
    text = re.sub(r"\[\d+\]\(@ref\)|\(@ref\)", "", text)   # pandoc/Hugo ref artefacts
    text = re.sub(r"(?m)^\s*(?:Ima\.Copilot|Copilot|ChatGPT|Claude|Gemini)[^\n]{0,12}:\s*$", "", text)
    text = text.replace("&#x95ee;", "问").replace("&#x95EE;", "问")
    # escaped inline link inside a heading: \[text\]\(url\) -> text (url)
    text = re.sub(r"\\\[(.*?)\\\]\\\(.*?\\\)", r"\1", text, flags=re.S)
    text = re.sub(r"(?<!\\)\\([()\[\]#])", r"\1", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    return text


def heading_from(question: str, limit: int = 58) -> str:
    """One-line, TOC-friendly heading taken from the raw question.

    Heuristics, in order: drop pasted filenames / URLs, drop a long leading
    ASCII title, keep the last sentence (Chinese prompts tend to ask at the
    end), then truncate.
    """
    q = re.sub(r"^>+", "", question.strip()).strip()
    q = re.sub(r"[#|`*]", "", q)
    q = re.sub(r"\]\([^)]*\)", "", q)                       # (url) half of a link
    q = re.sub(r"\[([^\]]*)\]", r"\1", q)                    # [text] -> text
    q = re.sub(r"\S*://\S+", "", q)                           # bare URLs
    q = re.sub(r"\S*\.(?:pdf|md|qmd|ipynb|py|png|jpe?g|csv|mat)\b", "", q, flags=re.I)
    q = re.sub(r"\s+", " ", q).strip(" ,、，-—/")
    if CJK.search(q) and CJK.search(q).start() >= 20:         # pasted English title
        q = q[CJK.search(q).start():]
    parts = [p.strip() for p in re.split(r"(?<=[。！？!.])", q) if p.strip()]
    cjk_parts = [p for p in parts if len(CJK.findall(p)) >= 3]
    if cjk_parts:
        q = cjk_parts[-1]
    q = re.sub(r"\s+", " ", q).strip(" ,、，.。-—/")
    return q if len(q) <= limit else q[: limit - 1].rstrip() + "…"


def demote(body: str, by: int = 1) -> str:
    """Push ``##``+ headings deeper so the turn headings own level 2."""
    def shift(m: re.Match) -> str:
        return "#" * min(6, len(m.group(1)) + by) + " "

    return re.sub(r"(?m)^(#{2,6}) ", shift, body)


def tab_tables(body: str) -> str:
    """Convert pasted tab-separated tables into pipe tables."""
    lines = body.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if "\t" in lines[i]:
            j, block = i, []
            while j < len(lines) and "\t" in lines[j].rstrip():
                block.append([c.strip() for c in lines[j].rstrip().split("\t")])
                j += 1
            wide = max(len(r) for r in block)
            if len(block) >= 2 and wide >= 2:
                rows = [r + [""] * (wide - len(r)) for r in block]
                out.append("| " + " | ".join(rows[0]) + " |")
                out.append("|" + "|".join([" --- "] * wide) + "|")
                out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
                i = j
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# pseudo-code detection
# --------------------------------------------------------------------------- #

CODE_KW = re.compile(r"^\s*(import|from|def|class|for|if|elif|else|while|return|print|with|try|except)\b")
STRONG = re.compile(
    r"^(import |from |def |class |for |if |elif |else|while |return |print\(|with |try|except)"
    r"|[A-Za-z_]\w*\.\w+\("
    r"|^[A-Za-z_][\w\.\[\]'\", ]*=[^=]"
    r"|^\s{4,}\S"
)
CALL = re.compile(r"[A-Za-z_]\w*\s*\([^)\n]*\)\s*$|[A-Za-z_]\w*\.\w+\(")
ASSIGN = re.compile(r"^\s*[A-Za-z_\]\[][A-Za-z0-9_\.\[\]'\", ]*\s*=\s*\S")
BULLETS = re.compile(r"^\s*(?:[-*+•]\s|>\s|\d+[.)、]\s|\|)")
ASCIIISH = re.compile(r"""[A-Za-z0-9_ ,()\[\]{}.:='"@+\-*/<>%#!~&|?°→≈μλΔΣ∏√≤≥±…·;_-]+""")
CODE_START = re.compile(r"^\s*(#|import |from |def |class |for |if |elif |else|while |return |print\(|with |try|except|@|\w+[\w\.\[\]]*\s*=[^=]|\s{4,}\w)")


def code_view(line: str) -> str:
    """Drop trailing ``# …`` comments and string bodies (CJK in them is not prose)."""
    s = re.sub(r"\s#.*$", "", line)
    s = re.sub(r"""f?'[^']*'|f?"[^"]*\"""", "", s)
    return s


def is_code(line: str) -> bool | None:
    """``True`` = code, ``False`` = prose, ``None`` = neutral (follows its neighbour)."""
    s = line.strip()
    if not s:
        return None
    if BULLETS.match(line) or s.startswith(("$$", ":::", "\\[")):
        return False
    if re.match(r"^#(?!#)[^\s#]", s) or re.match(r"^#\s+\S", s):   # a single-# python comment
        return True
    view = code_view(s)
    cjk = len(CJK.findall(view))
    signal = bool(STRONG.match(view)) or bool(CALL.search(view)) or bool(ASSIGN.match(view))
    if cjk == 0:
        return True if (signal or ASCIIISH.fullmatch(s)) else None
    if signal and cjk <= 10:                          # label + expression on one line
        return True
    return False


def fence_code(body: str) -> str:
    """Wrap obvious python runs in fenced code blocks.

    A run starts on a code line and keeps going through blank lines and through
    triple-quoted docstrings, whose prose would otherwise break the block.
    """
    lines = body.split("\n")
    n = len(lines)
    out: list[str] = []
    i = 0
    while i < n:
        if is_code(lines[i]):
            run, j = [lines[i]], i + 1
            in_doc = lines[i].count('"""') % 2 == 1
            blanks = 0
            while j < n:
                line = lines[j]
                if in_doc:                                     # swallow the docstring
                    run.append(line)
                    if line.count('"""') % 2 == 1:
                        in_doc = False
                    j += 1
                    continue
                flag = is_code(line)
                if flag:
                    run.append(line)
                    blanks = 0
                    if line.count('"""') % 2 == 1:
                        in_doc = True
                    j += 1
                    continue
                if flag is None and not line.strip() and blanks < 2:
                    ahead = [k for k in range(j, min(n, j + 3)) if is_code(lines[k])]
                    if ahead:
                        run.append(line)
                        blanks += 1
                        j += 1
                        continue
                break
            content = [l for l in run if l.strip()]
            if len(content) >= 2 and any(CODE_START.match(l) or CALL.search(l) for l in content):
                out.append("```python")
                out.extend(run)
                out.append("```")
                i = j
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def stray_headings(body: str) -> str:
    """A lone ``# comment`` outside a fence renders as <h1>: bold it instead."""
    out: list[str] = []
    inside = False
    for line in body.split("\n"):
        if line.lstrip().startswith("```"):
            inside = not inside
        if not inside and re.match(r"^#(?!#)\s*\S", line):
            out.append(f"**{line.lstrip('#').strip()}**")
        else:
            out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# turn parsing
# --------------------------------------------------------------------------- #


def split_question(text: str) -> tuple[str | None, str]:
    """Return ``(question, remaining_body)`` for one chat turn."""
    global stray_images
    stray_images = []
    first = text.split("\n", 1)
    head = first[0].strip()

    # "m4mad: question on the same line"  (colon is mandatory here)
    m = re.match(rf"^\s*(?:{ASK_ALT})\s*:\s*(.+)$", head, re.I)
    if m:
        return m.group(1).strip().lstrip(">").strip(), (first[1] if len(first) > 1 else "")

    # handle alone on its line, question on the following (possibly quoted) lines
    if re.match(rf"^\s*(?:{ASK_ALT})\s*:?\s*$", head, re.I):
        rest_lines = (first[1] if len(first) > 1 else "").split("\n")
        q_lines: list[str] = []
        consumed = 0
        for line in rest_lines:
            st = line.strip()
            if not st:
                consumed += 1
                if q_lines:
                    break
                continue
            if st.startswith(ANSWER_HANDLES):
                break
            # images pasted with a question belong to the page, not to the Ask box
            img_lines = re.findall(r'(?i)<img[^>]*>', line)
            if img_lines:
                stray_images.extend(img_lines)
                line = re.sub(r'(?i)<img[^>]*>', '', line)
                st = line.strip()
                if not st:
                    consumed += 1
                    continue
            q_lines.append(re.sub(r"^>+\s*", "", st))
            consumed += 1
        return (" ".join(q_lines) or None), "\n".join(rest_lines[consumed:]).strip()
    return None, text


def images(text: str, prefix: str = "", copy_to: Path | None = None) -> str:
    """``<img src="./img/1.jpg">`` in an export -> a captioned Quarto figure.

    ``prefix`` is the page-relative path to the project root (e.g. ``../../``);
    ``copy_to`` is the image store (``img/``) the referenced files are copied into.
    """
    def repl(m: re.Match) -> str:
        src = m.group(1).strip()
        name = Path(src).name
        if copy_to is not None:
            src_file = (TALKMD_DIR / src.lstrip("./"))
            if src_file.exists():
                copy_to.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, copy_to / name)
        n = re.sub(r"\D", "", name) or "?"
        return ("\n\n" + f"![截图 {n} · chat artefact {n}"
                f"（`Talkmd/img/{name}`，原样保留，未重绘）]"
                f"({prefix}img/{name}){{#fig-img-{n} width=58%}}" + "\n\n")

    IMG = re.compile(r'<img[^>]*?src=["\']([^"\']+)["\'][^>]*?/?>', re.I)
    text = re.sub(r'(?mi)^\s*' + IMG.pattern + r'\s*$', repl, text)
    return re.sub(IMG.pattern, repl, text)


def convert(source: Path, drop: tuple[int, ...] = (), img_prefix: str = "",
            img_store: Path | None = None) -> str:
    raw = clean(source.read_text(encoding="utf-8"))
    if img_store is None and not img_prefix:
        raw = re.sub(r"(?m)^\s*<img[^>]*>\s*$", "", raw)      # drop chat screenshots
    turns = [t.strip().strip("-").strip()
             for t in re.split(r"\n\s*(?:-{3,}|\*{3,})\s*\n", raw)]
    turns = [t for t in turns if t]
    multi = sum(1 for t in turns if split_question(t)[0]) > 1

    blocks: list[str] = []
    n = 0
    for turn in turns:
        question, body = split_question(turn)
        if question:
            n += 1
            if n in drop:
                continue
            blocks.append(
                f"## {n}. {heading_from(question)}\n\n"
                "::: {.callout-note}\n"
                "**🗣️ Ask**\n\n"
                f"> {re.sub(chr(10), ' ', question)}\n"
                ":::"
            )
        body = images(body, prefix=img_prefix, copy_to=img_store)
        body = tab_tables(body)
        body = fence_code(body)
        body = stray_headings(body)
        if stray_images:
            body = "\n\n".join(stray_images) + "\n\n" + body
        body = images(body, prefix=img_prefix, copy_to=img_store)
        if multi and question:
            body = demote(body)
        if body.strip():
            blocks.append(body.strip())

    return re.sub(r"\n{4,}", "\n\n\n", "\n\n".join(blocks)) + "\n"


if __name__ == "__main__":
    drop: tuple[int, ...] = ()
    args = []
    for a in sys.argv[1:]:
        if a.startswith("--drop="):
            drop = tuple(int(x) for x in a.split("=", 1)[1].split(",") if x.strip())
        else:
            args.append(a)
    out = convert(Path(args[0]), drop=drop)
    if len(args) > 1:
        Path(args[1]).write_text(out, encoding="utf-8")
        print(f"wrote {args[1]} ({len(out)} chars)")
    else:
        sys.stdout.write(out)
