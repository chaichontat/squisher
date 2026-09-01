# squisher monorepo

This repository contains five Python projects:

- [`squisher/`](squisher/README.md): CZI compression and verification.
- [`lightsheet/`](lightsheet/README.md): lightsheet stitching workflow commands.
- [`lightsheet-psf/`](lightsheet-psf/README.md): lightsheet point-spread-function analysis.
- [`deconv/`](deconv/README.md): GPU deconvolution tools.
- [`segment/`](segment/README.md): Cellpose training and distributed segmentation.

Use `squisher pyramid` to add fixed two-level TIFF SubIFD pyramids to existing
OME-TIFF folders or files; see [`squisher/README.md`](squisher/README.md#usage).
Cross-channel lightsheet registration, including the native method-8 DLL flow,
is documented in [`lightsheet/README.md`](lightsheet/README.md#cross-channel-native-registration).

```bash
uv sync --locked --all-groups
```
