# Third-party notices

## video-subtitle-remover derivative-work notice

CleanCanvas Studio is an independently maintained, substantially
modified derivative work of
[YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover),
copyright YaoFANGUK and contributors, licensed under Apache License 2.0.

The original project supplied the initial desktop application structure and
subtitle-removal workflow. This project has been renamed and extended with
selectable repair engines, OCR review, precise polygon masks, `MaskPlan`,
video mask strategies, model management, Real-ESRGAN enhancement, tests and
documentation. It is not an official upstream release and is not affiliated
with or endorsed by the upstream project or its contributors.

The repository-level change record is available in `MODIFICATIONS.md`. Source
and binary distributions must include the root `LICENSE`, `NOTICE`, this file,
and any upstream notices applicable to included material.

## IOPaint-derived implementation guidance

The lightweight LaMa/Anime-LaMa/Manga wrappers and Real-ESRGAN implementation
in this project were written with reference to selected files from IOPaint,
copyright its contributors, licensed under Apache License 2.0. The project
does not bundle IOPaint's Web application or diffusion-model stack.

Reference source: <https://github.com/Sanster/IOPaint> (archived upstream).
The complete Apache-2.0 license is included in this repository as `LICENSE`.
Distributions must retain the applicable Apache-2.0 license and attribution
notices when shipping derived source.

## Real-ESRGAN and BasicSR

Real-ESRGAN model URLs, RRDB-style architecture and weights are from the
Real-ESRGAN project, copyright Xintao Wang and contributors, BSD-3-Clause:
<https://github.com/xinntao/Real-ESRGAN>.

The Real-ESRGAN BSD-3-Clause text is included at
`THIRD_PARTY_LICENSES/Real-ESRGAN-BSD-3-Clause.txt`. The BasicSR-derived RRDB
architecture is Apache-2.0 and is covered by the root Apache-2.0 `LICENSE` and
the attribution in this file. Model weights are not included in this repository:
users download them on demand after choosing a mode, and hashes are verified
before use.

## LaMa / Anime-LaMa / Manga weights

Weights are not distributed with this feature. The application uses the URLs
and MD5 digests published by IOPaint/Sanster's model releases and downloads
them only when the user selects the corresponding mode. Before redistributing
weights in an installer, verify the current upstream model license and include
the upstream notices alongside the installer.

## Dependencies and bundled executables

FFmpeg, OpenCV, PaddleOCR/PaddlePaddle, PyTorch and other dependencies have
licenses independent of this repository. Release maintainers must generate a
version-specific dependency inventory and include notices required by the exact
wheels, executables, models and assets included in a release.
