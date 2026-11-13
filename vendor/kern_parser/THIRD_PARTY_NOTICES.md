# Third-party software

`verovioxide`, `verovioxide-sys`, and `verovioxide-data` come from
<https://github.com/btrkeks/verovioxide> at revision
`642a6634f2c8cf333a95aebb18274460a4f5094d`.
The wrapper uses Apache-2.0. Transcoda's workspace repeats this Git pin because
Cargo applies patches from the workspace root.

The wrapper builds Verovio from <https://github.com/rism-digital/verovio>.
Verovio uses LGPL-3.0. Its fonts and bundled resources have their own notices
in the upstream distribution.

Other Rust dependencies and their versions are listed in Cargo.lock.
Their licenses are separate from this crate's license.
