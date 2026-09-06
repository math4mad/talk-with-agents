#!/usr/bin/env python3
"""Turn a ``Talkmd/repo-<date>-<time>-<n>.md`` drop into the site's repository list.

The drop is not a dialogue — it is a numbered list of repositories, usually with
one line of intent each and often a placeholder URL (``...?tab=repositories``)
where the real link was not pasted. This script

1. parses those entries (label, URL, the author's own note),
2. resolves every entry against the GitHub API (cached in ``output/repos.json``),
   matching placeholder links by repository name,
3. writes ``coding/repos/repo-list.qmd``: one table of the curated list plus a
   short "active elsewhere" tail, each row marked with whether this site already
   carries a note about that repo.

Usage::

    python3 scripts/repo_list.py                 # regenerate from the newest drop
    python3 scripts/repo_list.py --offline       # use the cached repos.json only
    python3 scripts/repo_list.py --drop Talkmd/repo-20260906-1510.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TALKMD = ROOT / "Talkmd"
PAGE = ROOT / "coding" / "repos" / "repo-list.qmd"
CACHE = ROOT / "output" / "repos.json"
OWNER = "math4mad"

# pages on this site that already document a repository
DOCUMENTED = {
    "SARCOS-ML-with-Agents": "coding/repos/sarcos-ml-with-agents.qmd",
    "talk-with-agents": "index.qmd",
}

ENTRY = re.compile(r"^\s*(\d{1,2})\s*[.,、]?\s+(?:\[([^\]]+)\]\(([^)]+)\)|(\S+))\s*(.*)$")
URL_IN_TEXT = re.compile(r"https?://\S+")


# --------------------------------------------------------------------------- #
# GitHub metadata
# --------------------------------------------------------------------------- #
def fetch_repos() -> list[dict]:
    """Repo metadata from the GitHub API, cached so renders stay offline-safe."""
    try:
        out = subprocess.run(
            ["gh", "api", f"users/{OWNER}/repos?per_page=100&sort=updated",
             "--jq", ".[] | {name, description, language, pushed_at, html_url, homepage, fork}"],
            capture_output=True, text=True, timeout=90, check=True).stdout
        repos = [json.loads(line) for line in out.splitlines() if line.strip()]
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps({"fetched": dt.datetime.now().isoformat(timespec="seconds"),
                                     "repos": repos}, indent=2), encoding="utf-8")
        return repos
    except Exception as exc:                                    # offline / no token
        if CACHE.exists():
            print(f"  ! GitHub API unavailable ({exc}); using cached repos.json")
            return json.loads(CACHE.read_text(encoding="utf-8"))["repos"]
        raise


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# --------------------------------------------------------------------------- #
# the drop
# --------------------------------------------------------------------------- #
def latest_drop() -> Path | None:
    drops = sorted(TALKMD.glob("repo-*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return next((p for p in drops if p.read_text(encoding="utf-8").strip()), None)


def parse_drop(path: Path) -> list[dict]:
    """Numbered entries with their continuation lines folded in."""
    entries: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = ENTRY.match(raw)
        if m:
            label = (m.group(2) or Path(m.group(3) or m.group(4)).name).strip()
            url = (m.group(3) or m.group(4) or "").strip()
            if not url.startswith("http"):
                found = URL_IN_TEXT.search(m.group(5) or "")
                url = found.group(0).rstrip(".,") if found else ""
            entries.append({"label": label, "url": url, "note": (m.group(5) or "").strip()})
        elif raw.strip() and entries:
            entries[-1]["note"] = (entries[-1]["note"] + " " + raw.strip()).strip()
    return entries


def resolve(entries: list[dict], repos: list[dict]) -> list[dict]:
    """Attach real repo metadata; placeholder URLs are matched by name."""
    by_name = {norm(r["name"]): r for r in repos}
    rows = []
    for e in entries:
        url = e["url"]
        name = ""
        m = re.search(r"github\.com/[^/]+/([^/?#]+)", url or "")
        if m:
            name = m.group(1)
        meta = by_name.get(norm(name or e["label"])) or by_name.get(norm(e["label"]))
        if meta:
            name = meta["name"]
            url = meta["html_url"]
        elif url and url.rstrip("/").endswith("math4mad"):        # the "?tab=repositories" stub
            url = ""
        rows.append({**e, "name": name or e["label"], "url": url, "meta": meta})
    return rows


# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #
def clean_note(text: str) -> str:
    """The author's own notes keep their meaning, lose the typos that break markdown."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .,、g")
    fixes = {"pytyon": "python", "usin": "using", "symoblics": "symbolic", "forumla": "formula",
             "enxtension": "extension", "pluto": "Pluto",
             "derivate": "derivations", "bakset": "basket", "julie": "Julia", "refactor": "refactor"}
    for bad, good in fixes.items():
        text = re.sub(rf"\b{bad}\b", good, text, flags=re.I)
    text = re.sub(r"\bjulia language\b", "Julia", text, flags=re.I)
    return text


def build_page(rows: list[dict], repos: list[dict], drop: Path) -> str:
    listed = {norm(r["name"]) for r in rows if r["meta"]}
    today = dt.date.today().isoformat()

    head = f'''---
title: "仓库清单 · Repository list"
subtitle: "由 Agent 参与、或以 Agent 为主题的 Python / Julia 仓库"
description: "站点收录的仓库总表：意图、语言、最近推送、是否已有本页笔记。清单来自 Talkmd/{drop.name}，元数据由 GitHub API 解析（缓存在 output/repos.json）。"
author:
  - name: Math4Mad
    url: "https://github.com/{OWNER}"
date: "{today}"
sidebar: coding
format:
  html:
    toc-depth: 2
categories:
  - 与 Agent 共码
  - Agent 主导的仓库
---

:::{{.callout-note appearance="minimal"}}
## 这份清单怎么来的 · Provenance

源文件是 `Talkmd/{drop.name}`：一份编号的仓库列表（意图各一行，部分链接当时没贴全）。
`python3 scripts/repo_list.py` 把它和 GitHub API 的元数据对账：占位链接按仓库名解析回真实地址，
语言与"最近推送"取仓库实际值，**意图一栏保留原话**（只修掉会破坏渲染的拼写）。
所以表里"最近推送"是事实，"意图"是作者的说法，两者不必一致。
:::

## 清单 · The list

:::{{.repo-list}}

| 仓库 | 意图（原话整理） | 语言 | 最近推送 | 本站 |
| --- | --- | --- | --- | --- |
'''

    def cell(r: dict) -> str:
        meta = r["meta"] or {}
        unlisted = not meta and re.search(r"github\.com/[^/]+/[^/?#]+", r["url"] or "")
        if unlisted:                     # exists for the owner, invisible to readers
            return (f"| [{r['name']}]({r['url']}) | {clean_note(r['note']) or '—'} | "
                    f"私有 · private | — | — |")
        name = r["name"]
        link = f"[{name}]({r['url']})" if r["url"] else f"`{name}`（链接待补）"
        note = clean_note(r["note"]) or (meta.get("description") or "—")
        if meta.get("description") and clean_note(r["note"]):
            note = f"{note} ｜ GitHub 描述：{meta['description'].strip()}"
        lang = meta.get("language") or "—"
        pushed = (meta.get("pushed_at") or "")[:10] or "—"
        page = DOCUMENTED.get(name)
        here = f"[笔记](../../{page})" if page else "—"
        return f"| {link} | {note} | {lang} | {pushed} | {here} |"

    body = "\n".join(cell(r) for r in rows) + "\n\n:::"

    skip = listed | {norm(n) for n in DOCUMENTED}
    extras = [r for r in repos if norm(r["name"]) not in skip
              and (r.get("pushed_at") or "") >= "2026-08" and not r.get("fork")]
    tail = ""
    if extras:
        tail = ("\n\n## 清单之外近期还在动的 · Active but not on the list\n\n"
                "这些不在源清单里（也不在本页已收录的仓库中），但 2026-08 之后有推送，"
                "通常意味着「还没写进笔记」：\n\n:::{.repo-list-extra}\n\n"
                "| 仓库 | 语言 | 最近推送 |\n| --- | --- | --- |\n" +
                "\n".join(f"| [{r['name']}]({r['html_url']}) | {r.get('language') or '—'} | "
                          f"{(r.get('pushed_at') or '')[:10]} |" for r in extras[:12])
                + "\n\n:::\n")

    foot = f'''

## 怎么维护 · Maintenance

```bash
python3 scripts/repo_list.py            # 读最新的 Talkmd/repo-*.md + 刷新 GitHub 元数据
python3 scripts/repo_list.py --offline  # 只用 output/repos.json 缓存（CI 断网时）
```

新增仓库：往 `Talkmd/` 放一份新的 `repo-<日期>-<时间>-<序号>.md`，说一声 update 即可。
给某个仓库写正文：在 `coding/repos/` 新建一页（参考
[SARCOS inverse dynamics with Agents](sarcos-ml-with-agents.qmd) 的写法：只写能被脚本复现的结论），
然后把仓库名加进 `scripts/repo_list.py` 的 `DOCUMENTED`，本站一栏就会自动出现链接。
'''
    return head + body + tail + foot


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drop", default=None, help="explicit repo-*.md drop to read")
    ap.add_argument("--offline", action="store_true", help="use the cached repos.json")
    args = ap.parse_args(argv)

    drop = Path(args.drop) if args.drop else latest_drop()
    if not drop or not drop.exists():
        print("no non-empty Talkmd/repo-*.md drop found")
        return 1
    repos = json.loads(CACHE.read_text(encoding="utf-8"))["repos"] if (args.offline and CACHE.exists()) else fetch_repos()
    rows = resolve(parse_drop(drop), repos)
    unresolved = [r["label"] for r in rows if not r["meta"]]
    PAGE.write_text(build_page(rows, repos, drop), encoding="utf-8")
    print(f"repo list: {len(rows)} entries from {drop.name} -> {PAGE.relative_to(ROOT)}")
    if unresolved:
        print("  unresolved (kept as written, link marked 待补): " + ", ".join(unresolved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
