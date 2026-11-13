#!/usr/bin/env python3
"""Download released assets. IDs are in config/artifacts.json."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transcoda.inference.assets import artifact_config, resolve_checkpoint
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("asset", choices=["model", "synthetic-test"])
p.add_argument("--out", type=Path, default=Path("data/synthetic-test"))
a = p.parse_args()
if a.asset == "model":
    print(resolve_checkpoint())
else:
    from huggingface_hub import snapshot_download
    print(snapshot_download(repo_id=artifact_config()["synthetic_test_id"], repo_type="dataset", local_dir=a.out))
