# IMSLP-Real

IMSLP-Real contains 202 reference pages from 79 IMSLP documents of 28 pieces.
It provides compact Humdrum targets, document IDs, IMSLP file IDs and one-based
PDF page numbers. No scans are included.

Most references started from an existing encoding of the same piece, such as a
community MuseScore file or a ChopinScores edition. The first author checked and
corrected every reference transcription against the scanned page.
The public pages were selected before seeing any test result. Every piece retains
unreleased annotated pages. The 307 further test pages and 134 validation pages
are private and are absent from this release.

References omit lyrics and fingerings and do not consistently include ties, slurs,
articulations or ornaments. The shared evaluation mask keeps pitch, visible
accidentals, notehead and duration, dots, beams, tuplets, grace notes and grace
slashes, arpeggios, barlines, repeats, clefs, key signatures, time signatures and
ottavas. It removes other symbols from predictions and references alike.

The set favors widely played pieces and contains 122 Chopin pages. Treat its
results as evidence for this collection rather than all music notation.
Document-cluster intervals use 79 IMSLP files as clusters and 50,000 resamples.
Symbol-count thresholds were fixed from the 509-page extended reference set
before model scoring. Only the scalar thresholds are included here.

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

`evaluation/imslp-file-licenses.tsv` records scan terms for every included IMSLP
file. Scan licenses do not establish symbolic encoding rights. Scans remain on
IMSLP. The download helper prints file links, downloads user-supplied direct PDF
URLs after license review, and extracts only approved pages. Scan raster and crop
equivalence is not guaranteed.
