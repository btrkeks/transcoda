# Third-Party Licenses

## facebookresearch/ConvNeXt-V2

- Upstream: https://github.com/facebookresearch/ConvNeXt-V2
- License: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/LICENSE
- Usage: small FCMAE helper functions were ported and adapted for rectangular dense masked image modeling in `src/pretraining/fcmae/`.

## craigsapp/humlib (musicxml2hum)

- Upstream: https://github.com/craigsapp/humlib
- Fork used: https://github.com/btrkeks/musicxml2hum
- License: BSD-2-Clause
- Usage: the `musicxml2hum` binary from our fork is invoked by `scripts/dataset_generation/kern_conversions/convert_xml_to_kern.sh` to convert MusicXML source files to Humdrum `**kern` during dataset generation. Reproducing the published datasets requires the fork; upstream `musicxml2hum` may produce subtly different output.

