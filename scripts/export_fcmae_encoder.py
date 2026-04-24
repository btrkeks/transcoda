from __future__ import annotations

import argparse

from src.pretraining.fcmae.lightning_module import FCMAEPretrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the FCMAE encoder as a HF directory")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    module = FCMAEPretrainer.load_from_checkpoint(
        args.checkpoint_path,
        map_location="cpu",
    )
    encoder = module.model.encoder
    if not hasattr(encoder, "save_pretrained"):
        raise TypeError("wrapped encoder does not expose save_pretrained")
    encoder.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
