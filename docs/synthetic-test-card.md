---
license:
- cc-by-4.0
- mit
task_categories:
- image-to-text
language: en
---
# Transcoda synthetic test split

2,500 synthetic sheet-music pages with compact Humdrum transcriptions:
1,500 from an unused PDMX hold-out pool and 1,000 from the official GrandStaff
test split. There are no KernScores rows. No test page shares a source ID or
content hash with the training data, under the frozen construction recorded in
the paper. Pages use the generation pipeline without visual augmentation.

PDMX material uses CC BY 4.0. GrandStaff material uses MIT. These terms apply
to their respective rows, identified by `document_id` prefixes `synthetic:pdmx:`
and `synthetic:grandstaff:`. Original metadata use CC BY 4.0. Upstream material
is not relicensed. Sources: https://github.com/pnlong/PDMX and
https://huggingface.co/datasets/PRAIG/GrandStaff. Retain their license and attribution notices.

The directory uses Hugging Face `datasets.save_to_disk`. Load it with:

```python
from datasets import load_from_disk
dataset = load_from_disk("data/synthetic-test")
```

Columns are `image`, `transcription`, `sample_id` and `document_id`. Images are
1050 by 1485 pixels. Embedded image bytes and targets match the evaluated split;
the release narrows metadata columns. `release/synthetic-test-artifact.json`
records file hashes and source counts. The dataset ID is in `config/artifacts.json`.

The split measures recognition of rendered music rather than scan degradation.
Its layout and notation coverage follow the source corpora and renderer. Results
should be checked on real scans before deployment. It is a test set, not training data.
