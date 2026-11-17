# SupeRR Selector S2: Pressure Index dashboard

## Contents
- `dashboard/index.html`: the interactive dashboard. It is one self-contained file (HTML, CSS, JavaScript and data inline). Double-click to open it in any browser. Fonts load from Google Fonts when you are online; without a connection it falls back to system fonts.
- `dashboard/source/`
  - `template.html`: the page without data.
  - `data.json`: player data used by the dashboard.
  - `build_data.py`: rebuilds `data.json` from the pipeline outputs and the data pack. It also derives country, capped status, batting position and bowling phase, and applies the hand-corrected cap list.
- `pipeline/`
  - `pressure_pipeline.py`: the v2 pipeline (Markov model, SPI v2, player metrics, validation).
  - `PRESSURE_PIPELINE_v2.md`: full explanation of the pressure definition, the Markov model and every change from v1.
  - `results/`: rankings, top-10 lists, figures, the calibration table and `run_report.json` from the full run.

## Rebuild
1. Run the pipeline:

   ```
   python pipeline/pressure_pipeline.py --bbb data/ball_by_ball/ball_by_ball.csv --data-dir data --out out
   ```

2. Edit the paths `D` (data folder) and `R` (pipeline output folder) at the top of `dashboard/source/build_data.py`, and the output path for `data.json`. Then run it.
3. Insert `data.json` into `template.html` at the `/*__DATA__*/` placeholder, for example:

   ```
   python -c "t=open('template.html').read();open('index.html','w').write(t.replace('/*__DATA__*/',open('data.json').read()))"
   ```

## Note
Capped status comes from T20I appearances in the data (2022–26), plus hand corrections for players capped in other formats or before 2022. Check current status before the auction.
