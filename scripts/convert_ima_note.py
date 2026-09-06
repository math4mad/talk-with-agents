#!/usr/bin/env python3
"""Convert an ``ima``-style chat export (one long stream, no turn rules).

The export in ``Talkmd/泛函空间…?.md`` is not markdown at all: 1.1 MB sits on a
handful of lines, each answer ends with the app's *suggested follow-up chips*,
then a raw retrieval dump (numbered snippets, URLs, sometimes whole arXiv
sections), then the user's next question — all glued together without breaks.

Layout of one turn::

    <answer body, real newlines kept>
    <last line: answer tail + suggested chips + retrieval dump + next question>

This script keeps the answer body, drops the dump, and reports the tail of every
turn as ``<!-- TAIL … -->`` so the real question can be written by hand.

Style / LaTeX fixes applied to the kept text:

* zero-width spaces and U+200B inside maths removed;
* Japanese full stops/commas (｡ ､) from scraped snippets -> 。 、;
* ``\\[`` ``\\]`` ``\\%`` escapes from the chat renderer unescaped;
* ``[9](@ref)`` cross-reference artefacts dropped;
* citation indices glued to sentence ends (``连续。5``, ``框架。51``) removed;
* ``\\begin{…}`` blocks wrapped as display maths, ``$$`` put on their own lines;
* paragraph breaks re-inserted before ``N.`` / ``N、`` / ``一、`` / ``定义：`` … ;
* inline maths padded with spaces so MathJax does not swallow CJK punctuation.

Usage::

    python3 scripts/convert_ima_note.py Talkmd/泛函空间….md out.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_math import convert as normalize_math  # noqa: E402

MARKER = re.compile(r"ima引用\s*\d+\s*篇资料作为参考")

# strong signals that a chunk of text is a retrieval dump rather than prose
DUMP_SIGNALS = re.compile(
    r"https?://|\.(?:com|cn|org|net|pdf|io|html)\b|arXiv|ar5iv|paperreading|"
    r"URL\s|PDF\s*Copy|SciSpace|Semantic Scholar|百度百科|知乎|CSDN|github\.com",
    re.I,
)
NUM_TITLE = re.compile(r"(?<![\d.])\d{1,2}\s*[:.]\s+[A-Za-z\u4e00-\u9fff]")

# headings that the chat used as inline labels — start a new paragraph
BREAK_BEFORE = re.compile(
    r"(?:(?<=^)|(?<=[。；：！？：\\n]))(?=[ \\t]*(?:"
    r"\d{1,2}\s*[.、．]\s"           # 1.  /  1、
    r"|[一二三四五六七八九十]+\s*[、.．]\s"   # 一、
    r"|定义[：:]"
    r"|动机[：:]"
    r"|地位[：:]"
    r"|结论[：:]"
    r"|总结[：:]"
    r"|一句话[总小]"
    r"|注意[：:]"
    r"|示例[：:]"
    r"|前提[：:]"
    r"|原理[：:]"
    r"|应用[：:]"
    r"|数学意义[：:]"
    r"|物理意义[：:]"
    r"|核心[要思]"
    r"))"
)


def clean_prose(text: str) -> str:
    """Style + LaTeX normalisation for the kept answer prose."""
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u00ad", "")
    text = text.replace("｡", "。").replace("､", "、").replace("［", "[").replace("］", "]")
    text = text.replace("&#x95ee;", "问").replace("&#x95EE;", "问")
    text = re.sub(r"<span[^>]*>|</span>", "", text)
    text = re.sub(r"\[\d+\]\(@ref\)", "", text)              # pandoc-style ref artefacts
    text = re.sub(r"(?<!\\)\\([()\[\]%_#])", r"\1", text)     # chat-escaped punctuation
    # citation indices stuck to the end of a sentence:  …连续。5  /  …框架。51
    # trailing citation indices: …连续。5 / …它。51 / …的宝藏112。 / …交互1234，
    text = re.sub(r"(?<=[。；：！？])\s*\d{1,4}(?=\s|$|[。；，、])", "", text, flags=re.M)
    text = re.sub(r"(?<=[\u4e00-\u9fff”）])[0-9]{1,4}(?=[。；！？]|$)", "", text, flags=re.M)

    # display maths on their own lines
    text = re.sub(r"(?<!\n)\s*(\$\$)", r"\n\1", text)
    text = re.sub(r"(\$\$)\s*(?!\n)", r"\1\n", text)
    # \begin{...} environments that lost their $$ wrapper
    text = re.sub(r"(?<!\$\$)\s*(\\begin\{(?:matrix|pmatrix|bmatrix|aligned|cases|equation)\})",
                  r"\n$$\n\1", text)
    text = re.sub(r"(\\end\{(?:matrix|pmatrix|bmatrix|aligned|cases|equation)\})(?!\$\$)",
                  r"\1\n$$", text)

    # paragraph breaks before inline labels
    text = BREAK_BEFORE.sub("\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # citation indices that sit before the punctuation: …显著压缩 35。 / …矩阵 Ŵ45。
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+\d{1,3}(?=[。；，、])", "", text)
    # (never touch 1000x500 / 7\times 7 style products: the digit must not follow x × * / -)
    text = re.sub(r"(?<![x×*/-])(?<=[A-Za-z$)）\u00c0-\u024f])\d{1,3}(?=[。；，、])", "", text)
    text = re.sub(r"\s+\d{1,3}(?=[。；](?:\s|$))", "", text, flags=re.M)

    # markdown niceties: bold labels, bullet lists
    text = re.sub(r"(?m)^([^\n#|$]{2,18}?)：\s*$", r"**\1**", text)
    return text.strip()


def is_dump_line(line: str) -> bool:
    """A retrieval-dump line: huge, linky, or mostly latin reference text."""
    s = line.strip()
    if len(s) > 3000:
        return True
    if not s:
        return False
    cjk = len(re.findall(r"[\u3400-\u9fff]", s))
    if DUMP_SIGNALS.search(s) and cjk / max(len(s), 1) < 0.35:
        return True
    if NUM_TITLE.match(s) and cjk / max(len(s), 1) < 0.25:
        return True
    return False


CHIP_END = re.compile(
    r"[\u3400-\u9fffA-Za-z0-9 ]{2,24}?"
    r"(?:有哪些|是什么|如何[^\s，。；]{0,14}|怎样|有何[^\s，。；]{0,10}|的表现如何|的区别|的作用)"
)


def split_chips(tail: str) -> tuple[str, str]:
    """Split an answer's last line into (real prose, suggested chips + next question).

    The app appends exactly three templated follow-up chips with no punctuation;
    the first position from which two chip-shaped clauses follow in short order
    is where the answer ends.
    """
    hits = list(CHIP_END.finditer(tail))
    for k, h in enumerate(hits):
        nxt = [g for g in hits if g.start() > h.start() and g.start() - h.start() < 90]
        if len(nxt) >= 1 and h.start() > 12:
            return tail[: h.start()].strip(), tail[h.start():].strip()
    return tail.strip(), ""


def reindent_python(block: list[str]) -> list[str]:
    """The export flattens python indentation to one space; rebuild it from colons."""
    out, level = [], 0
    for line in block:
        s = line.strip()
        if not s:
            out.append("")
            continue
        if re.match(r"^(else|elif|except|finally)\b", s) and level:
            level -= 1
        out.append("    " * level + s)
        if s.endswith(":") and not s.startswith(("#", '\"""')):
            level += 1
        elif re.match(r"^(return|pass|break|continue|raise)\b", s) and level:
            level -= 1
    return out


def fence_code(text: str) -> str:
    """``pythonimport torch`` in the export means a collapsed ```python fence."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = re.match(r"^(python|json|bash|text)(?=\S|$)", lines[i].strip())
        if m:
            lang = m.group(1)
            first = lines[i].strip()[len(lang):]
            j, body = i + 1, [first] if first else []
            while j < len(lines):
                nxt = lines[j].strip()
                if not nxt or nxt.startswith(("**", "#")) or re.match(r"^\d+[.、]", nxt) \
                        or len(nxt) > 150:
                    break
                body.append(nxt)
                j += 1
            if lang == "python":
                body = reindent_python(body)
            out.append(f"```{lang}")
            out.extend(body)
            out.append("```")
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def sub_headings(text: str) -> str:
    """A standalone ``N. 短标签`` line is a sub-heading, not a list item."""
    def repl(m: re.Match) -> str:
        num, label = m.group(1), m.group(2).strip()
        return f"\n### {num}. {label}\n"

    return re.sub(r"(?m)^(\d{1,2})[.、．]\s*([^\n。！？$]{2,30})\s*$", repl, text)


def split_prose(chunk: str) -> tuple[str, str]:
    """Return (answer prose, everything from the first dump line onwards)."""
    lines = chunk.split("\n")
    keep: list[str] = []
    for line in lines:
        if is_dump_line(line):
            break
        keep.append(line)
    return "\n".join(keep), "\n".join(lines[len(keep):])


def tail_question(rest: str, limit: int = 1200) -> str:
    """Best-effort: chips + next user question, i.e. what survives after the dump."""
    seg = rest[-limit:]
    for pattern in (DUMP_SIGNALS, NUM_TITLE):
        hits = list(pattern.finditer(seg))
        if hits:
            seg = seg[hits[-1].end():]
    return seg.strip()


def convert(source: Path) -> str:
    raw = source.read_text(encoding="utf-8")
    parts = MARKER.split(raw)
    blocks: list[str] = []
    n = 0
    for chunk in parts[1:]:
        n += 1
        prose, rest = split_prose(chunk)
        # the final line of the prose holds: answer tail + suggested chips + next question
        body_lines = prose.split("\n")
        tail = body_lines[-1] if body_lines else ""
        body = "\n".join(body_lines[:-1])
        text = clean_prose(body)
        answer_tail, chips = split_chips(tail)
        tail_clean = clean_prose(answer_tail)
        if tail_clean and tail_clean not in text:
            text = f"{text}\n\n{tail_clean}" if text else tail_clean
        text = fence_code(text)
        text = sub_headings(text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        question = chips or tail_question(rest)
        blocks.append(
            f"## {n}. TODO — 补全提问\n\n"
            f"<!-- CHIPS+Q: {question[:700]} -->\n\n"
            f"{text}"
        )
    out = "\n\n".join(blocks)
    return re.sub(r"\n{4,}", "\n\n\n", out) + "\n"


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = convert(src)
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if dest:
        dest.write_text(out, encoding="utf-8")
        print(f"wrote {dest} ({len(out)} chars)")
    else:
        sys.stdout.write(out)
