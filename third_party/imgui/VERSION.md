# Dear ImGui (vendored)

- Upstream: https://github.com/ocornut/imgui (MIT, see LICENSE.txt)
- Tag: `v1.92.9b-docking` (docking branch), source archive
  `imgui-1.92.9b-docking.zip`, sha256
  `0434445157a575f452ff0f2d1681fdd90ea8939e0c6983e6f8e47c51fba1bccd`
- Vendored 2026-09-30 (owner-approved; the one planned download).
- Copied: core (`imgui*.h/.cpp`, `imconfig.h`, `imstb_*.h`, LICENSE) and
  the `win32` + `vulkan` backends. Nothing else.
- Files are UNMODIFIED. Configure via compile definitions in CMake, not by
  editing `imconfig.h`. To upgrade: replace the files from a new tag and
  update this note.
- Exempt from the repo's ASCII-only and clang-format rules (upstream code).
