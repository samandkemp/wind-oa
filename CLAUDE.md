# wind-oa

## What this is

An interactive 3D GPU wind tunnel: an LBM solver (subsonic) and a compressible
Euler solver (transonic) on Vulkan compute, with a Dear ImGui front-end.
Personal research / learning tool and portfolio piece for a single expert user
(Samuel, Chemistry MSc) -- not a shipped product.

## Design goals (these override convenience)

- **The physics is the product.** Every model is gated against an external
  reference (analytic solution, published data, a documented invariant)
  before it is made fast or pretty. Correctness is testable; "looks right"
  is not.
- **Headless engine.** `engine/` is a library with no window and no GUI; the
  validation gates and tools drive it directly.
- **Interactive.** The UI never blocks on GPU work: compute on its own queue,
  long jobs (voxelise, compile) report progress instead of freezing, shaders
  compiled to SPIR-V at BUILD time (no JIT stalls), per-frame GPU timing.
- **Identity discipline.** Every optional subsystem (spin, turbulence,
  sponge, dye) is bit-identical to its absence when switched off.
- **Explicit memory layout.** Buffer layouts are written down (std430,
  sizes asserted): a misaligned u8 field corrupts flags silently.

## Stack

- C++20, MSVC 2022, CMake >= 3.25 with the Visual Studio 17 2022
  generator. No Ninja, no package manager, no FetchContent.
- Vulkan 1.3 (LunarG SDK 1.4.363), compute shaders in GLSL -> SPIR-V via
  `glslc`, embedded at build time.
- **Dependencies (owner, 2026-09-30): "we don't want any external
  downloads/dependencies, where it can be avoided."** The Vulkan SDK is the
  system toolchain. Memory is plain vkAllocateMemory (no VMA -- a handful of
  large buffers needs no sub-allocator). A Win32 window +
  VK_KHR_win32_surface (no GLFW), Dear ImGui (owner-chosen) VENDORED into
  `third_party/` (a copied, pinned release -- ask before fetching anything),
  plots drawn with ImGui's draw lists (no ImPlot). Hand-rolled: STL I/O,
  voxeliser, marching cubes, PNG writer, test harness (plain executables +
  CTest). Any new dependency: say so explicitly, justify it, let Samuel
  decide.
- Target GPU: AMD Radeon RX 7900 XT (RDNA3, wave32/64, 20 GB).

## Architecture

```
engine/      headless core (static lib windoa_engine); Vulkan::Vulkan only
  include/windoa/context.hpp   Context (ContextOptions: graphics, async
                               compute queue), Buffer, ComputeKernel,
                               groups_for, timeline semaphores, Image3D
                               (R32F 3-D texture, hardware trilinear)
  include/windoa/lbm.hpp       lbm::Solver (+ inlet turbulence, per-step hook)
  include/windoa/euler.hpp     euler::Solver (transonic)
  include/windoa/dye.hpp       Dye (D3Q7 TRT)
  include/windoa/mesh.hpp      Mesh, STL I/O, fit_to_box, transform
  include/windoa/voxeliser.hpp Voxeliser (winding fill, thin shell, signed distance)
  include/windoa/isosurface.hpp marching_cubes (CPU; gated by V16)
  include/windoa/catalogue.hpp the 30 procedural models
  include/windoa/tunnel.hpp    Tunnel: the sandbox (everything the app does
                               that is not UI) + TunnelSettings (tunables)
  include/windoa/convergence.hpp, flow_cache.hpp  settling detector, cache
  include/windoa/flow_stats.hpp FlowStats (GPU time averages), wake survey
  include/windoa/spectrum.hpp   spectrum (uneven samples -> Hann -> FFT)
  shaders/                     lbm_*, vox_*, euler_*, dye_*, turb_* + includes
render/      VolumeRenderer (march, Q cores, dye, surface paints incl. oil-flow
             LIC, slice LIC, mean reversed-flow shells, splats) + Tracers
             (smoke / timelines, streamlines, slice arrows, markers), reading
             snapshot buffers; depends on engine
app/         Win32 + Dear ImGui app: App (panels incl. Analysis), SimWorker (Tunnel on a
             thread + snapshot hand-off), Swapchain, OrbitCamera, plots, PNG
third_party/ imgui (v1.92.9b-docking, vendored unmodified; VERSION.md)
tools/       device_info, compute_selftest, lbm_run, tunnel_run, euler_run,
             bench, lbm_equiv
validation/  gates V1-V26 (one exe each, CTest label "gate", "long" for
             the > 1 min ones); gate.hpp = ledger + numerics, cases.hpp +
             euler_cases.hpp = setups shared by several gates
docs/        public, in three tiers under README.md (tier 1): GUIDE.md (use it),
             MODEL.md (understand it), REFERENCE.md (lookup) = tier 2;
             THEORY.md (the spec, sections N.M), VALIDATION.md (gates) = tier 3.
             Local, gitignored: PLAN.md, PROGRESS.md, TRANSITION.md
```

Dependency arrows point one way: app -> render -> engine; tools/validation
-> engine. **engine never includes window or imgui headers.**

## Conventions

- ASCII-only source and comments (no Greek / Unicode). The one exception is
  the section sign in Markdown docs: docs cite theory sections as §N.M,
  source cites the same ones as `THEORY N.M`. THEORY sections are stable
  references (source and gates point at them), so never renumber them.
- Docs and comments follow the owner's writing style (British English,
  measured and impersonal, units spaced `0.5 %`, no contractions in formal
  docs). Published titles and quotations stay verbatim. Our own identifiers
  and file names are British too (`Voxeliser`, `catalogue`, `regularised`,
  `colour`); names that must match external code keep their spelling
  (Vulkan, ImGui, GLSL built-ins, `std`). Keep the five-doc hierarchy: a
  new fact goes in the one doc whose job it is (README for orientation,
  GUIDE for operating, MODEL for how it works, REFERENCE for flags and
  defaults, THEORY for equations, VALIDATION for gates).
- Comments, docs and commit messages state a design reason as a property
  of the method, never as a comparison with some other code base.
- Format with clang-format (`.clang-format`, format-on-save). Warnings at
  /W4 and treated seriously.
- Gates: one executable per validation case under `validation/`, exits
  non-zero on failure, registered with CTest by gate id. Each checks its
  EXTERNAL reference and prints the measured value. See
  `docs/VALIDATION.md`.
- New physics = new gate, before optimising it. If a gate fails after a
  change, understand why before re-baselining.
- Tunables live in one place (`TunnelSettings` in tunnel.hpp), not
  scattered literals.
- Prefer clear code over clever; profile (GPU timestamp queries) before
  optimising. Repeat a timing before believing a difference.

## Commands

- Configure + build: `cmake --preset dev` then `cmake --build --preset dev`
  (RelWithDebInfo; `--build --preset debug` / `release` for the others).
  Any prompt: the VS generator finds MSVC itself.
- Gates: `ctest --preset dev` (everything, ~13 min), `ctest --preset quick`
  (skips label `long`, ~80 s), `ctest --preset dev -R V17` (one gate).
- GPU report: `build\tools\RelWithDebInfo\device_info`
- App: `build\app\RelWithDebInfo\windoa_app [fast|balanced|fine] [transonic]`
  (finds the repo root itself); scripted: `--show / --field / --warmup /
  --frames / --shot` (app/src/main.cpp). Dev tools: `tunnel_run`,
  `euler_run`, `bench`. clang-format: the VS Code C++ extension's
  (`%USERPROFILE%\.vscode\extensions\ms-vscode.cpptools-*\LLVM\bin`).
- Validation layers: `cmake --preset validation-layers` then
  `cmake --build --preset validation-layers` (tree in `build-vl/`).
- Shaders: `windoa_add_shaders(<target> shaders/x.comp)` (cmake/Shaders.cmake)
  compiles GLSL with glslc at BUILD time and embeds it; `#include
  "x_spv.hpp"` gives `windoa::spv::x`. Never load shader files at runtime.

## Physics rules (hard-won -- do not relearn)

- LBM pressure outlet reads the PREVIOUS step's vel at the last interior
  plane: anything that writes distributions directly must refresh rho/vel.
- Closed domains leak mass: `enforce_mass()` (uniform rescale) is needed for
  the cavity to pass.
- Moving walls: spin cap 0.08 lattice; thin rotor blades 0.025 (0.04
  diverged). Stability tests need a 3-D body -- a periodic slab is 2-D
  turbulence and goes NaN at U 0.11 regardless.
- A new body / big placement change restarts from rest with the inlet ramp
  (an impulsive start reaches |u| 0.31).
- Ahmed Cd plateaus ~1.14 vs 0.285 experimental: Re + blockage, NOT a
  solver defect. (That is V15's setup -- 256x128x128, L 100, regularised,
  Re 7,500; the app's fast preset, L 64 + sponge, reads ~0.72: a different
  configuration.) The sphere error budget: blockage ~3.8 pts, streamwise
  confinement ~1.2, wall placement ~0.6. Boundary-layer resolution is NOT
  the cause -- do not cite it to motivate grid refinement.
- Voxeliser: nonzero WINDING rule (not parity), 2-of-3 axis vote, plus a
  thin-feature supercover shell kept only where the far side of the sheet is
  within 1.5 cells or no solid is near. Four simpler thin rules were REFUTED.
- Euler walls: image-point ghost cells need an EXACT signed distance (a
  flags-only phi puts the wedge 13 % off); curvature-corrected symmetry
  (Dadone & Grossman) for lift; wall pressure interpolated to the TRUE wall.
  A mirror about the GRID axis builds a spurious boundary layer; reflecting
  the ADJACENT cell makes trailing edges porous.
- Euler 2-D sections need SLIP z faces; far field >= ~6 chords away.
- Forces: sum over MODEL cells only (the floor once read Cl -124).
- Flow-cache keys must include everything that changes the flow: solver
  source, voxeliser source, spinner definitions, ground mode.
- Validate a restore against a never-restored CONTROL, not a fixed bar --
  the wake is unsteady.
- Free-slip faces MIRROR the populations about the face (the cell's own
  row). Pulling the reflected population from the row inside creates and
  destroys mass along the walls; the inlet and outlet hide it from the
  total, but the wake survey read 3 - 5 % low until it was fixed. V2
  checks the local conservation, V25 the balance, V26 D the acoustics.
- Perfectly reflecting side walls ring: transverse acoustic modes at
  n c_s / (2 n_y) (333 steps on the fast grid) show in every spectrum.
  The shedding St is the strongest peak below 0.8 of the first mode.

## Engineering rules (add as they are learned)

- A fence wait does NOT make one batch's shader writes visible to the next
  batch: `submit_and_wait` brackets every batch with full barriers.
- Never early-`return` before a `barrier()` in a kernel that reduces in
  shared memory (non-uniform barrier = undefined behaviour): mark the
  thread inactive and fall through (see lbm_step.comp main()).
- Deterministic sums: per-workgroup fixed-order trees + a single-group
  reduce, no float atomics. Host-side sums of partials in double.
- Dispatches over cells split groups over x and y (`groups_for`): the
  fine preset exceeds 65,535 groups in x.
- `macro` (u.xyz, rho) is double-buffered with f, so the outlet reads
  exactly the previous step (a single buffer would race on that read).
- Two queues (async compute + graphics): never `vkDeviceWaitIdle` from
  the UI thread while the worker runs (it needs every queue externally
  synchronised) -- wait on the graphics queue. The worker never touches
  renderer resources; the renderer reads SNAPSHOTS, never live solver
  buffers (they are rewritten every step).
- Every Vulkan object must be destroyed before its Context -- including
  members destroyed after an explicit `ctx_.reset()` in a destructor (the
  screenshot buffer crashed the app on exit).
- Dye + spinning walls: the Ladd moving-wall term makes the flow locally
  compressible, and the conservative dye piled up (C 760): the D3Q7 step
  has a C <= 1.25 limiter (identity for the validated cases).
- Spinners are dropped onto the road WITH the model in ground mode (left at
  the undropped centre, road-car wheel spin captures no cells).
- P4 rule: an optimisation that claims to be an identity must pass
  `lbm_equiv` (switch on vs off in one binary) AND `lbm_equiv --check
  build\p4_orig` (vs results saved from the pre-optimisation kernel). Code
  motion that "cannot change the arithmetic" did: streaming the collide
  outputs changed ~1e-7 bits (FMA contraction) and was reverted.
- Restoring a file with `Copy-Item` keeps its OLD timestamp, so MSBuild
  does not rebuild it (a stale binary once passed a check it should have
  failed). Touch restored files: `(Get-Item f).LastWriteTime = Get-Date`.
- Placement arithmetic: centres in DOUBLE, rounded to float once (`shapes`
  takes double). `200 * 0.3f` = 60.0000038 drops 5 of 2,109 sphere cells
  (the cells at exactly r) and moves V9's Cd by 1 %.
- Marching cubes interpolates every edge from its lower corner, so shared
  vertices are bit-identical and the soup welds into a closed surface.
- Splats lying in the slice plane (arrows, timelines, probes) need a depth
  bias: the slice registers its hit a little in front of the plane.
- FlowStats stores rho - 1, and plane sums add the 1 in double: f32 rho
  rounds away the 1e-4 deviations a momentum balance needs.
- Shell: Bash heredocs collapse doubled backslashes (a `\\n` inside a
  Python string arrives as a newline), so edits touching backslashes go
  through the editor tools or a Python script written with the Write tool.
  Never use PowerShell `Set-Content -Encoding utf8` on sources (it adds a
  BOM). PowerShell's table formatting can swallow later output lines: pipe
  test output through `Out-String` when filtering it.

## Guardrails

- Never let `engine/` depend on the window, the GUI or the renderer.
- Imported STLs (F35, LowPoly_F35, f1) never ship; only procedural models.
- Do not optimise a subsystem before its gates are green.
- Commit only when Samuel asks.

## How to work with me

- Plan before executing non-trivial work; get sign-off.
- Decision questions: options ranked by complexity, honest pros/cons, a
  clear recommendation. Samuel makes the call.
- Teach as you go on non-obvious C++ / Vulkan idioms (one line on why).
- Smallest change that works; flag uncertainty; say when this file is stale.

## Where we are (continuity)

The local, gitignored notes hold the detail: `docs/TRANSITION.md` (the
handover; read it first in a new session) and `docs/PROGRESS.md` (the log).

- 2026-09-30: repo created. Toolchain: CMake in `C:\Program Files\CMake`,
  Vulkan SDK 1.4.363.0 in `C:\VulkanSDK`. 7900 XT: async compute family 1,
  f16 yes, subgroup size 64. P0 gates `P0_device_info`, `P0_compute`.
- 2026-09-30 - 10-01, build sessions: P1 LBM core, P2 voxeliser, P3 app
  (Win32 + vendored Dear ImGui; solver on the async queue via SimWorker
  snapshots), renderer (Q cores, dye, smoke, streamlines), the 30-model
  catalogue, the Tunnel sandbox (spin, ground, guard, settling, flow cache,
  turbulence, dye), P5 Euler + transonic mode.
- 2026-10-01, P4 first pass: LBM 2,850 -> 3,550 MLUPS bit-identically,
  opt-in f16 ~4,800; Euler 2.6x; renderer Q throttle. Safety net =
  tools/lbm_equiv + euler_run equiv (baselines in build\p4_orig,
  build\p4_euler -- local; regenerate from the reference path if the build
  dir is wiped).
- 2026-10-02, V&V session: renderer samples Image3D textures; gates V1-V23
  written and green; docs restructured into README + GUIDE / MODEL /
  REFERENCE + THEORY / VALIDATION; source cites `THEORY N.M`.
- 2026-10-03: identifiers and file names made British; caps emphasis in
  comments lowered; repo pushed to https://github.com/samandkemp/wind-oa
  (origin, main). Owner: the repo reads as a C++ project in its own right,
  so every gate stands on its external reference alone. V16 (marching
  cubes, `marching_cubes`) added: 23 gates.
- 2026-10-03, features (owner chose: flow statistics, surface + slice
  visuals, probes + spectra; no big physics yet): GPU time averages (mean
  speed, turbulence intensity, mean reversed-flow shells), the wake survey
  (drag from a control-volume momentum balance), probes, spectra with
  acoustic modes marked; surface paints (near-wall speed, reversed flow,
  oil-flow LIC), slice LIC, slice arrows, pulsed timelines; Analysis
  panel; tunnel_run --average / --probe / --no-cache; app --zoom. The
  survey exposed the free-slip mass defect (owner: fix it); results
  re-measured, build\p4_orig regenerated. Gates V24 - V26: 26 gates.
  Open: gating f16.
