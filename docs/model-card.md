---
language: en
license: cc-by-nc-4.0
tags:
- optical-music-recognition
- image-to-text
- music
---
# Transcoda V2.1

Transcoda converts printed Western staff notation into compact Humdrum notation.
It has 108M parameters: a ConvNeXt V2 Tiny encoder and a Transformer decoder.
The tokenizer and grammar are included in the code repository. Load the exported
checkpoint with `scripts/infer.py`; the default model ID is in `config/artifacts.json`.

The model is intended for music research, transcription drafts and tool development.
Check its output against the score before archival, analytical or performance use.
It can miss notes, rhythms, meters and structural changes. It is not evaluated on
handwriting or non-Western notation. The reference collection favors common piano
works and is dominated by Chopin. It omits ties, slurs, articulations, dynamics, lyrics, voltas and D.C./Fine instructions.

Training uses 343,470 synthetic pages from PDMX, GrandStaff and KernScores.
No real scans enter training. The selected checkpoint is step 8715 from a
10,695-update run, selected using the lowest real-validation CER.
The 134 real validation pages are private, so exact selection cannot be repeated.
One training seed is reported; confidence intervals describe test-page variation.
Optimizer, scheduler and machine state are absent from the export. Tensor equality
and the source checkpoint hash are recorded in `release/model-artifact.json`.

| Mode | Synthetic OMR-NED | IMSLP-Real OMR-NED, document CI | IMSLP-Real renderability |
| --- | ---: | ---: | ---: |
| Beam | 1.56% | 28.50% [23.71, 34.14] | 89.1% |
| Constrained greedy | 1.25% | 22.59% [18.87, 27.21] | 100.0% |

Both modes use fp32, max length 2048, repetition penalty 1.1 and length penalty 1.0.
Beam width is 3, without constraints. The default uses greedy decoding with the
released grammar, compact barlines, data-only empty fields, dead-end lookahead,
spine constraints and the timeline rule. The latter pads short voices with
invisible rests. Images are resized to width 1050 and padded or cropped to height
1485, with no whitespace trimming or layout normalization.

Weights use CC BY-NC 4.0. The encoder is initialized from
`facebook/convnextv2-tiny-22k-224`. Its Hugging Face metadata lists Apache-2.0;
the ConvNeXt V2 repository assigns CC BY-NC 4.0 to pretrained weights.
Retain both upstream notices. See NOTICE.md for links and attribution.
