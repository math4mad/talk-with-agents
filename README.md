# Talk With Agents

Working notes on **talking with** and **coding with** LLM agents — a Quarto website.

* `talking/` — the agent as a *discussion partner*: mathematics, teaching mathematics,
  the biological basis of learning theory, physics taught through data, gossip.
* `coding/` — the agent as a *collaborator*: runnable snippets and agent-built Python repos.

## Layout

```
_quarto.yml            site, navigation, two sidebars, publication figure defaults
styles.css             house style (dialogue callouts, journal tables, hero)
index.qmd              landing page with the note listing
about.qmd              conventions: how a chat export becomes a page
Talkmd/                raw chat exports and pasted source papers (never edited)
                       drop-in names: talk-<date>-<time>-<n>.md, repo-<date>-<time>-<n>.md
talking/               notes, one .qmd per dialogue, grouped by sub-category
coding/                snippets + repo summary pages
scripts/
  figstyle.py          publication style: widths, CM maths, colour-blind-safe palette
  make_figures.py      regenerates assets/figures/*.{png,pdf}
  space_search_demo.py the experiment behind coding/snippets/function-space-search.qmd
  convert_note.py      chat export -> Quarto body (turns, tables, pseudo-code)
  convert_ima_note.py  retrieval-app export (no rules, dumps glued to answers) -> Quarto body
assets/figures/        generated at 300 dpi + vector PDF (English-only labels — enforced)
img/                   screenshots pasted into the dialogues (raw copies stay in Talkmd/img/)
output/                CSV artefacts that the numbers on the pages come from
```

## Build

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # figures need only numpy/scipy/matplotlib
python3 scripts/make_figures.py          # 300 dpi PNG + PDF
python3 scripts/space_search_demo.py     # ~1 min, writes output/space_search_results.csv
quarto render                            # -> _output/
quarto serve                             # preview
```

Publishing: pushing to `main` runs `.github/workflows/quarto-publish.yml`, which regenerates the
figures and the experiment, renders with Quarto, and publishes the result to the `gh-pages` branch.
