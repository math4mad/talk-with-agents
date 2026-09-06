#!/usr/bin/env python3
"""Intake for the ``Talkmd/`` naming convention declared in ``AGENTS.MD``.

    Talkmd/talk-<date>-<time>-<order>.md   -> a dialogue to publish under talking/
    Talkmd/repo-<date>-<time>-<order>.md   -> a repository note under coding/repos/

Running ``python3 scripts/intake.py`` (the "update" step) reports which sources are
new or have changed since they were last converted, and writes a Quarto draft for
each one into ``.draft/``. Add ``--publish`` to also drop the draft into its
category folder as a page skeleton ready for the manual pass.

State lives in ``output/intake.json`` (source -> {page, sha256, converted}), so a
re-run only touches files that actually changed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TALKMD = ROOT / "Talkmd"
DRAFTS = ROOT / ".draft"
STATE = ROOT / "output" / "intake.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_ima_note import MARKER as IMA_MARKER  # noqa: E402
from convert_ima_note import convert as convert_ima  # noqa: E402
from convert_note import convert as convert_note     # noqa: E402

SOURCE_RE = re.compile(r"^(talk|repo)-(\d{8})-(\d{4})(?:-(\d+))?\.md$")

HEAD = '''---
title: "{title}"
subtitle: "{subtitle}"
description: "{description}"
author:
  - name: Math4Mad
    url: "https://github.com/math4mad"
date: "{date}"
image: "{prefix}assets/figures/site-cover.png"
bibliography: {prefix}references.bib
format:
  html:
    toc-depth: 3
---

::: {{.callout-note appearance="minimal"}}
## 关于这一篇 · About this note

原始记录：`Talkmd/{source}`（由 `scripts/intake.py` 转换）。
待补：一句话说明这一篇从哪来、删掉了什么、改了什么。
:::

'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def detect_kind(text: str) -> str:
    """Which converter owns this export?"""
    if IMA_MARKER.search(text):
        return "ima"
    if re.search(r"(?m)^\s*(?:Math4Mad|m4mad)\s*:?", text, re.I):
        return "note"
    return "unknown"


def pretty_date(stamp: str) -> str:
    try:
        return dt.datetime.strptime(stamp, "%Y%m%d").date().isoformat()
    except ValueError:
        return dt.date.today().isoformat()


def slug_from_title(body: str) -> str:
    """First question in the export -> a filename slug (ASCII, hyphenated)."""
    m = re.search(r"(?m)^\s*Math4Mad\s*:\s*(.+)$", body) or re.search(r"(?m)^#{1,6}\s+(.+)$", body)
    words = re.findall(r"[A-Za-z][A-Za-z0-9]+", m.group(1) if m else "")
    slug = "-".join(w.lower() for w in words[:6])
    return slug or "untitled"


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--publish", action="store_true",
                    help="also write the page skeleton into its category folder")
    args = ap.parse_args(argv)

    if not TALKMD.is_dir():
        print(f"no {TALKMD} directory")
        return 1

    state = load_state()
    todo, unchanged = [], []
    for path in sorted(TALKMD.glob("*.md")):
        m = SOURCE_RE.match(path.name)
        if not m:
            continue                                   # not part of the convention
        kind, stamp, _time, _order = m.groups()
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            print(f"  - {path.name}: empty placeholder — nothing to convert yet")
            continue
        digest = sha256(path)
        record = state.get(path.name)
        if record and record.get("sha256") == digest:
            unchanged.append(path.name)
            continue
        todo.append((path, kind, stamp, text, digest))

    if not todo:
        print(f"intake: nothing new ({len(unchanged)} source(s) already converted)")
        return 0

    DRAFTS.mkdir(exist_ok=True)
    for path, kind, stamp, text, digest in todo:
        fmt = detect_kind(text)
        if fmt == "unknown":
            print(f"  ! {path.name}: not a chat export (no Ask/answer markers) — "
                  f"treat it as a reference source and cite it from a page")
            state[path.name] = {"sha256": digest, "format": "unknown", "kind": kind,
                                "page": None,
                                "converted": dt.datetime.now().isoformat(timespec="seconds")}
            continue
        body = convert_ima(path) if fmt == "ima" else convert_note(path, drop=())
        stem = slug_from_title(text)
        draft = DRAFTS / f"intake-{path.name}.md"
        draft.write_text(body, encoding="utf-8")

        folder = "coding/repos" if kind == "repo" else "talking/mathematics"
        prefix = "../../"
        page = ROOT / folder / f"{stem}.qmd"
        head = HEAD.format(title="TODO 标题", subtitle="TODO 副标题", description="TODO 摘要",
                           date=pretty_date(stamp), prefix=prefix, source=path.name)
        print(f"  + {path.name}  [{kind} / {fmt}]  {len(text):,} chars -> {len(body):,} chars body")
        print(f"      draft: {draft.relative_to(ROOT)}")
        if args.publish:
            page.write_text(head + body, encoding="utf-8")
            print(f"      page : {page.relative_to(ROOT)}  (front matter + questions still TODO)")
        else:
            print(f"      page : would be {folder}/{stem}.qmd  (re-run with --publish)")
        state[path.name] = {"sha256": digest, "format": fmt, "kind": kind,
                            "page": str(page.relative_to(ROOT)),
                            "converted": dt.datetime.now().isoformat(timespec="seconds")}

    save_state(state)
    print(f"intake: {len(todo)} new/changed, {len(unchanged)} unchanged")
    print("next  : finish front matter + Ask callouts + tables, then `quarto render`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
