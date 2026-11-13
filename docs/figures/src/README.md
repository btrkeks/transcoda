# README figure sources

The PNG files in `docs/figures` are built from the TikZ sources here:

```sh
docs/figures/src/build.sh
uv run --no-sync python docs/figures/src/manifest.py
```

`build.sh` needs pdflatex with pgfplots, Fira Sans and Source Code Pro,
plus pdftoppm and ImageMagick. `manifest.py` records each image hash in
`docs/figures/figures.json`. The release audit accepts only images listed there.

## Image sources

| Asset | Source | Terms |
| --- | --- | --- |
| `liszt-p26-*` scan crops | Liszt, Hungarian Rhapsody No. 2, Peters edition, printed page 26 (PDF page 7) of [IMSLP file 300584](https://imslp.org/wiki/File:PMLP12453-Liszt_Klavierwerke_Peters_Sauer_Band_1_02_Ungarische_Rhapsodie_No_2_scan.pdf). IMSLP-Real sample `imslp__liszt-hungarian-rhapsody-no2-s-244-2-imslp300584__p0007`, cropped from the 1050 px model input. | Public domain as stated by IMSLP (see `evaluation/imslp-file-licenses.tsv`). |
| `liszt-p26-systems4-5-transcoda.png` | The released constrained-decoding prediction for that page, rendered by `render_prediction.py` and cropped to systems 4 and 5. | Original output of this release. |
| `grandstaff-*` pages | Four GrandStaff test pieces, converted to Humdrum and rendered by the released generator. | MIT, as GrandStaff. |

The GrandStaff pages come from the first 40 GrandStaff rows of the synthetic test split,
converted with `kern_parser serialize-jsonl --preset humdrum-kern --input-dialect transcoda-v2`
and written one file per row. Then:

```sh
uv run --no-sync python scripts/generate.py --input-kern <kern-dir> --source grandstaff \
  --work <work> --output <out> --workers 4 --target-count 6 --seed 7 --continuation-fraction 0
```

The clean panel uses the same command with `--no-augmentation`.
Asset names give the 0-based output row. Pages are resized to 420 px width in grayscale.
The hero crop is system 1, bars 1 to 3. The scan-versus-output figure shows systems 4 and 5.
The bar chart values are the mean OMR-NED scores in `evaluation/`:
`frozen-*-summary.json` for Transcoda and `baselines/*/*/report.samples.jsonl` for SMT++ and Legato.
The qualitative page scores 1.28% OMR-NED with constrained decoding.
It is one of the better IMSLP-Real pages; the mean is 22.59%.
