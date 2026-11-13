# Licenses and attribution

The implementation, training configurations and data-generation code use AGPL-3.0-only.
The full text is in LICENSE. Asset licenses below apply to their stated scopes.

| Asset | License and scope |
| --- | --- |
| `vendor/kern_parser` | PolyForm Noncommercial 1.0.0. Required Notice: Copyright 2026 Daniel Dratschuk. This license applies only to that directory. |
| VeroSim, fetched by `scripts/setup_verosim.sh` | MIT. Retain its upstream notice. |
| `vendor/musicxml2hum` | BSD-2-Clause. Copyright Craig Stuart Sapp and other contributors; see LICENSE.txt. |
| pugixml bundled with musicxml2hum | MIT. Copyright Arseny Kapoulkine; notice retained in its source headers. |
| verovioxide | Apache-2.0. The Cargo build fetches the pinned fork. |
| Verovio | LGPL-3.0. Used through verovioxide, VeroSim and the Python package. Retain its license and notices. |
| Fine-tuned checkpoint | CC BY-NC 4.0, with upstream ConvNeXt V2 weight notices retained. |
| Original tokenizer, annotations and dataset metadata | CC BY 4.0, limited to original contributions. |
| PDMX synthetic test material | CC BY 4.0. Source rights and attribution remain applicable. |
| GrandStaff synthetic test material | MIT. Retain the source license. |
| README figures in `docs/figures` | Our figure compositions, diagrams and charts are CC BY 4.0. Scan crops from IMSLP file 300584 are public domain as stated by IMSLP. Pages rendered from GrandStaff pieces are MIT. Sources: `docs/figures/src/README.md`. |

ConvNeXt V2: the encoder is initialized from `facebook/convnextv2-tiny-22k-224`.
Its Hugging Face model metadata lists Apache-2.0. The ConvNeXt V2 repository
assigns CC BY-NC 4.0 to pretrained weights. The fine-tuned checkpoint therefore
uses CC BY-NC 4.0 and retains both notices. See
https://huggingface.co/facebook/convnextv2-tiny-22k-224 and
https://github.com/facebookresearch/ConvNeXt-V2/blob/main/LICENSE.

KernScores encodings and rendered KernScores pages are not redistributed.
The training snapshot contains 11,391 KernScores rows and is not released.
Users must obtain source material themselves and follow each file's terms.
We acknowledge CCARH for MuseData, the Fryderyk Chopin Institute for Polish Scores,
and Helmut Schaffrath for Essen.

IMSLP scans remain on IMSLP under each file's terms; see the per-file scan-license TSV.
Most references started from an existing encoding of the same piece, such as a
community MuseScore file or a ChopinScores edition. The first author checked and
corrected every reference transcription against the scanned page.

Our original transcriptions, corrections and metadata are CC BY 4.0. The per-page
`provenance` field in the public TSV and JSON records the starting source:

| Provenance | Pages | Starting source |
| --- | ---: | --- |
| `hand_transcription` | 42 | Original hand transcription. |
| `chopinscores` | 23 | Scherzo Op. 31, ChopinScores first-edition encodings. |
| `existing_encoding_unrecorded` | 1 | Op. 10 No. 4; the original encoding source was not recorded. |
| `musescore_community` | 136 | MuseScore.com community downloads, corrected against the scan. |

ChopinScores-derived pages use the Fryderyk Chopin Institute (NIFC)'s
`humdrum-chopin-first-editions` encodings, created by Marcin Konik, under
CC BY 4.0. Source: https://chopinscores.org and
https://doi.org/10.5281/zenodo.14879352. Our corrections adapt these encodings.

MuseScore-derived pages started from community uploads of public-domain works.
We could not fully establish the encoding terms for these uploads. Our CC BY 4.0
license covers our original contributions and does not relicense source encodings.
We will remove any page on request through
https://github.com/btrkeks/transcoda/issues.

License texts: https://creativecommons.org/licenses/by/4.0/,
https://creativecommons.org/licenses/by-nc/4.0/ and
https://polyformproject.org/licenses/noncommercial/1.0.0/.
