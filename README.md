# Knowing When to Think Fast or Slow at Test Time

Source for the research post at **https://tinevimbomusingadi.github.io/kwtfs-blog/**.

One small language model (Qwen3-4B + LoRA, trained with [Tunix](https://github.com/google/tunix) on spot TPUs)
learns, question by question, whether to decide in a single forward pass, answer in a few words, or reason first.
Code: [kwtfs-public](https://github.com/TinevimboMusingadi/kwtfs-public).

## How the page is built

No number in the post is typed by hand.

```
tools/build_data.py   saved experiment outputs  ->  data/*.json (figure data) + data/numbers.json
tools/render.py       src/post.html + numbers.json + tools/refs.json  ->  index.html + medium.md
tools/snapshot.py     headless Chrome  ->  img/fig-*.png (static figures for medium.md) + img/og.png
tests/reward_parity.py  the reward explorer's JS against the Python training reward
tools/linkcheck.py    every external link returns 200
```

`src/post.html` uses two placeholders, and rendering fails if either does not resolve:

- `{{n:path|fmt}}` — a value from `data/numbers.json`
- `{{cite:key}}` — a numbered citation from `tools/refs.json`

The site is static: `index.html`, `style.css`, `js/` (D3 figures) and `data/`. To preview:

```bash
python -m http.server 8765
```

`build_data.py` needs the original experiment outputs (`--src` pointing at the training repo's `runs/`), so the
committed `data/` is what the page uses. `research_notes/bibliography.md` holds the annotated, verified bibliography.

## Data notes

Example traces quote question text only from sources whose licences allow it (GSM8K, AQuA-RAT, CommonsenseQA and
procedurally generated tasks). Questions from the BEV decision mix, OpenBookQA and ARC appear only as IDs and
statistics. Quoted questions keep their source licences and are not covered by the CC BY 4.0 licence below.

## Licence

Text and figures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code (`js/`, `tools/`, `tests/`): MIT.

## Citation

```bibtex
@misc{musingadi2026kwtfs,
  author = {Musingadi, Tinevimbo},
  title  = {Knowing When to Think Fast or Slow at Test Time},
  year   = {2026},
  url    = {https://tinevimbomusingadi.github.io/kwtfs-blog/}
}
```
