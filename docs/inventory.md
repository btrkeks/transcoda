# Release inventory

`release/file-inventory.json` lists the intended source and evaluation files with
sizes and SHA-256 hashes. Downloaded weights and synthetic images stay outside Git.
The export contains model, training, generation and evaluation code; tokenizer and
grammar; public scan-reference metadata; synthetic targets; predictions and scores.

Excluded: review application, corpus database, private reference tools, real-page
training flows, experiment infrastructure, machine configs and internal history.
Real scans, the 307 further test pages, 134 validation pages, Polish references,
KernScores encodings and rendered KernScores pages are not release assets.
The parser directory contains the public data-preparation subset, including its
license and attribution notices. Its source revision is recorded in
`release/source-revisions.json`.
