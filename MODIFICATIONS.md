# Upstream modifications and derivative-work notice

CleanCanvas Studio is a substantially modified and independently
maintained derivative work of
[YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover),
which is distributed under the Apache License 2.0.

The project has been renamed and reorganized. It is not an official upstream
release and is not affiliated with or endorsed by the upstream maintainers.

## Material modifications

- selectable OpenCV, LaMa, Anime-LaMa and Manga repair engines;
- OCR review, precise polygon masks, local contour refinement and `MaskPlan`;
- static, sampled and per-frame OCR video-mask strategies;
- model caching, integrity checks, CPU/CUDA selection and Real-ESRGAN enhancement;
- tests, documentation, multilingual product naming and open-source governance.

Material local changes occur in `gui.py`, `backend/`, `ui/`, `docs/`, tests,
configuration and project documentation. Keep applicable upstream copyright,
license and attribution notices when redistributing source files. This record
does not remove any obligation imposed by the Apache License 2.0.
