# Data Analysis for Diplomatic Neutrality vs. Military Neutrality

The scripts here were used to analyze data for the paper *Diplomatic Neutrality vs. Military Neutrality*.

Country mentions are extracted from a sample of the *GlobalDiplomacyNet* corpus, scored with aspect-based sentiment analysis (ABSA), and aggregated to a country-quarter panel, and then analysed.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       # on VPS/Linux/Mac
pip install -r requirements.txt
```

For a CPU-only machine, install torch first to avoid the large CUDA download:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Layout

```
Data/        *.jsonl (files from the *GlobalDiplomacyNet* corpus, e.g. AUT_mofa.jsonl) and country_aliases.json
Scripts/     the pipeline (see below)
Output/      all results, prefixed by the script number (01_*, 02_*, ...)
Graphs/     figures for the paper
```

All paths, country groups, the invasion quarter and thresholds are set in
`Scripts/config.py`; edit them there.

## Main specification: deduplicated sentences

The analysis uses **deduplicated sentences**: a sentence is counted once per
(direction, source country, target country), however often it recurs across
documents, quarters or files.
Scripts `03`-`07` therefore run with `--dedupe`, and their outputs carry the suffix
`_dedup` (figures keep their plain file names). The non-deduplicated panel is
kept only as a robustness variant in `06`.

## Run order

| Step | Command | What it does |
|---|---|---|
| 1 | `python Scripts/01_extract_mentions.py` | Finds sentences mentioning RUS/UKR (from source countries), the source countries (from RUS/UKR), and each belligerent in the other's files (`dyadic`) |
| 1b | `python Scripts/01b_show_duplicates.py` | Lists all sentences that deduplication collapses (check before relying on it) |
| 2 | `python Scripts/02_score_absa.py` | ABSA sentiment toward the mentioned country. Resumes automatically; `--limit 200` for a timing test |
| 2b | `python Scripts/02b_validation.py --make`, then `--evaluate` | Draws a sample to label manually, then reports model agreement |
| 3 | `python Scripts/03_build_panel.py --dedupe` | Country-quarter panel (tilt, asymmetry, valence, joint reception) |
| 4 | `python Scripts/04_summary_stats.py --dedupe` | Descriptive tables (CSV and LaTeX) and in-text macros |
| 4b | `python Scripts/04b_data_overview.py --dedupe` | Mention counts per country and period; coverage heatmap |
| 5 | `python Scripts/05_models.py --dedupe` | Estimates H1-H4 |
| 6 | `python Scripts/06_robustness.py --dedupe` | Variants, leave-one-out, bootstraps, randomization inference |
| 6b | `python Scripts/06b_belligerent_dyad.py --dedupe` | Benchmark: how Russia and Ukraine speak about each other vs. about the neutrals |
| 7 | `python Scripts/07_figures.py --dedupe` | Figures 1-4 |


Note that steps `01` and `02` are slow (`02` takes hours on CPU;
use tmux/nohup). Everything from `03` onward is fast.

For the `no_dedupe` variant in step 6, also build the full panel once:
`python Scripts/03_build_panel.py`.

## Outputs

- `Output/02_mentions_absa.parquet`: all mentions with sentiment scores (the main data product)
- `Output/03_panel_dedup.parquet`: country-quarter panel used by everything downstream
- `Output/04_numbers_dedup.tex`, `04b_numbers_dedup.tex`, `05_numbers_dedup.tex`,
  `06_numbers_dedup.tex`: LaTeX macros for in-text numbers (`\input` them in the preamble)
- `Output/*_summary*.txt`, `*_results*.txt`: readable reports for each step
- `Graphs/`: `fig1_equidistance`, `fig2_asymmetry_time`, `fig3_reception`,
  `fig4_tilt_by_group`, `figA1_mention_coverage_dedup`
- `Output/06b_dyad_table_dedup.csv`, `06b_dyad_summary_dedup.txt`: benchmark of RUS<->UKR sentiment vs. sentiment toward neutrals
- `Graphs/figA2_dyad_benchmark_dedup`: quarterly benchmark figure

## Notes

- Sentiment: `absa_score = P(positive) - P(negative)` toward the mentioned country, in [-1, 1].
- Tilt = score toward Russia minus score toward Ukraine; asymmetry = |tilt|.
- Check `Output/01_alias_counts.csv` for noisy aliases before the full scoring run.
- The ABSA model (`yangheng/deberta-v3-base-absa-v1.1`, ~1 GB) is downloaded on first use.
- Mentions with direction `dyadic` (RUS<->UKR) are used only by `06b`; all other scripts ignore them.

Anthropic's Claude (Sonnet 5.5) aided with code optimization and debugging, with commenting the scripts (e.g., through docstrings), and with creating this README-file.