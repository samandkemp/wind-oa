# Reference

Every flag, tunable, preset, model and control, with its type, default and the theory section that
specifies it. This is a lookup table: keep it open while running a study or changing a setting. For
the prose walkthrough of *how* to use any of it, see [`docs/GUIDE.md`](GUIDE.md); for what a model
does with a value, follow the § reference into [`docs/THEORY.md`](THEORY.md).

**Source of truth.** When this page and the code disagree, the code is right. The tunnel's tunables
are `TunnelSettings` in [`engine/include/windoa/tunnel.hpp`](../engine/include/windoa/tunnel.hpp);
the solvers' options are `lbm::Config` in
[`engine/include/windoa/lbm.hpp`](../engine/include/windoa/lbm.hpp) and `euler::Config` in
[`engine/include/windoa/euler.hpp`](../engine/include/windoa/euler.hpp); the catalogue is
[`engine/src/catalogue.cpp`](../engine/src/catalogue.cpp). All values are in lattice units (cells,
steps) unless stated.

## Contents

- [Grid presets](#grid-presets)
- [Tunnel settings](#tunnel-settings)
- [Lattice Boltzmann solver options](#lattice-boltzmann-solver-options)
- [Compressible solver options](#compressible-solver-options)
- [Catalogue models](#catalogue-models)
- [Command-line tools](#command-line-tools)
- [App controls](#app-controls)
- [Test presets and labels](#test-presets-and-labels)
- [Files written at run time](#files-written-at-run-time)
- [The flow-cache key](#the-flow-cache-key)

## Grid presets

| Preset | Grid | Cells | Notes |
|---|---|---|---|
| `fast` | 256 x 96 x 96 | 2.36 M | The default; about 3,300 MLUPS in the sandbox |
| `balanced` | 320 x 128 x 128 | 5.24 M | |
| `fine` | 384 x 160 x 160 | 9.83 M | About 2,400 MLUPS while rendering |
| `ultra` | 512 x 192 x 192 | 18.9 M | About 160 steps a second; for patient, mostly headless studies |

An unknown preset name gives `fast` (`tunnel_preset`).

## Tunnel settings

`TunnelSettings`, the app's single source of tunables. `solver_config(TunnelSettings)` turns them
into the app's `lbm::Config`, so that a gate claiming to test what the app runs constructs exactly
that solver.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `nx`, `ny`, `nz` | int | 256, 96, 96 | Grid size (set by the preset) |
| `tau` | float | 0.504 | Relaxation time; $\nu = (\tau - 0.5)/3$ (§1.2). Usable this close to 0.5 only with a regularised operator |
| `smagorinsky_cs` | float | 0.1 | Smagorinsky constant; 0 disables the LES model (§2.5) |
| `regularised` | bool | true | Projected-regularised collision (§2.3) |
| `recursive` | bool | false | Recursive-regularised collision (§2.4); implies `regularised` |
| `u_inlet` | float | 0.05 | Startup speed command (Mach 0.087) |
| `u_max` | float | 0.11 | Top of the speed slider (Mach 0.19) (§1.2) |
| `u_slew_per_step` | float | 1e-4 | Largest change of the applied speed per step (§9.1) |
| `ramp_steps` | int | 2000 | Ramp from rest at startup and after a reset (§9.1) |
| `floor_height` | int | 3 | Rolling road or fixed floor thickness, cells |
| `max_wall_speed` | float | 0.08 | Fastest a spinning surface may move (§9.3) |
| `sub_cell_walls` | bool | true | Walls at their true sub-cell position: link fractions from the mesh and interpolated bounce-back; false = half-way bounce-back (§3.6) |
| `max_jet_speed` | float | 0.12 | Fastest an engine port's face may blow or draw; every port is scaled together above it (§3.11) |
| `alm_eps` | float | 2 | Width of the Gaussian that spreads a rotor's blade forces onto the grid, cells (§3.12) |
| `diverge_speed` | float | 0.5 | The guard trips above this speed (§9.4) |
| `outlet_sponge` | int | 24 | Sponge width before the outlet, cells; 0 = off (§3.3) |
| `sponge_target` | int | 1 | 1 = relax to the full freestream; 0 = density only (§3.3) |
| `turbulence_length` | float | 8 | Eddy size of the inlet turbulence, cells (§7.1) |
| `force_ema_steps` | double | 120 | Dashboard smoothing time constant, solver steps (§4.4) |
| `health_every` | int | 10 | Batches between full-grid health checks (§9.4) |
| `dye_tau`, `dye_tau_plus` | float | 0.53, 1.0 | Dye relaxation times; $D = (\tau - 0.5)/4$ (§6.2) |
| `dye_nozzle_rate` | float | 0.5 | Nozzle relaxation towards $C = 1$ per step (§6.3) |
| `dye_nozzles` | int | 6 | Nozzles per side of the wand (a 6 x 6 rake) |
| `flow_cache_mb` | double | 512 | Disk budget of the settled-flow cache, MiB (§9.6) |
| `storage_f16` | bool | false | Half-precision distribution storage: an ungated approximation (§11.3) |
| `mach_min`, `mach_max` | float | 0.3, 1.6 | Range of the transonic Mach slider |
| `mach_slew` | float | 0.004 | Largest Mach change per batch (§9.1) |
| `develop_flow_throughs` | double | 2.0 | Transonic development time, flow-throughs |
| `force_ema_time` | double | 60 | Transonic dashboard smoothing, in cell sound-crossings |
| `analysis_every` | int | 25 | Batches between refreshes of the wake survey and the spectra (§12) |
| `history_flow_throughs` | double | 16 | Length of the signal history the spectra are taken over (§12.3) |

## Lattice Boltzmann solver options

`lbm::Config`. The app's values come from `solver_config` (above); the defaults below are those of a
bare solver, as the gates construct them.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `nx`, `ny`, `nz` | int | - | Grid size (required) |
| `tau` | float | 0.504 | Relaxation time (§1.2) |
| `u_inlet` | float | 0.0 | Inlet speed, $+x$ (`set_inlet_velocity` changes it per step) |
| `smagorinsky_cs` | float | 0.1 | Smagorinsky constant (§2.5) |
| `mode_x` | enum | `InletOutlet` | `InletOutlet` (§3.1, §3.2) or `Periodic` |
| `mode_y`, `mode_z` | enum | `FreeSlip` | `FreeSlip` (§3.4) or `Periodic` |
| `regularised` | bool | false | Projected-regularised collision (§2.3) |
| `recursive` | bool | false | Recursive-regularised collision (§2.4) |
| `use_ibb` | bool | false | Interpolated bounce-back; needs link fractions, `set_link_q` (§3.6) |
| `moving_boundaries` | bool | false | Moving-wall terms on `OBSTACLE` cells (§3.7) |
| `outlet_sponge` | int | 0 | Sponge width, cells (§3.3) |
| `sponge_target` | int | 0 | 1 = full freestream, 0 = density only (§3.3) |
| `opt_lazy_macro` | bool | true | Write $\rho$, $\mathbf{u}$ only when read; an exact identity (§11.1) |
| `opt_sparse_forces` | bool | true | Skip the force tree where no model link exists; an exact identity (§11.2) |
| `storage_f16` | bool | false | Half-precision storage; an ungated approximation (§11.3) |

Run-time setters (no recompilation): `set_inlet_velocity`, `set_body_force`, `set_lid_velocity`,
`set_torque_ref`, `set_inlet_turbulence`, `set_turbulence_convection`, `set_rotation`,
`set_wall_velocity` (a port's face, §3.11), `clear_wall_velocity`, `set_link_q`, `enforce_mass`.
`enable_force_field` switches the per-cell body force that actuator lines write (§3.12); it
recompiles the step through a specialisation constant, and off it is bit-identical.

## Compressible solver options

`euler::Config`. The app builds it with the far-field side faces and the tunnel inflow / outflow.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `nx`, `ny`, `nz` | int | - | Grid size (required) |
| `mach` | float | 0.8 | Freestream Mach number (`set_mach`) |
| `cfl` | float | 0.4 | Courant number of the device-side time step (§8.4) |
| `limiter` | enum | `VanLeer` | `VanLeer` or `Minmod` (§8.2) |
| `side_bc` | enum | `Farfield` | $y$ faces: `Farfield` or `Slip` (§8.5) |
| `z_bc` | int | -1 | $z$ faces; -1 = as `side_bc`. A two-dimensional section needs `Slip` |
| `x_bc` | enum | `Tunnel` | `Tunnel` (inflow / outflow) or `Transmissive` (zero gradient) |
| `wall` | enum | `Image` | `Image` (ghost cells, §8.6) or `Mirror` (grid-axis reflection) |

`set_flow_angle(degrees)` tilts the freestream in the $x$-$y$ plane; `set_dt_cap` caps the time step.
`set_ports` gives up to eight engine exhausts their exit states, primitive $(\rho, u, v, w, p)$, for
the cells flagged `kPortFlag + k` (§8.12).

## Catalogue models

The 30 procedural models, in menu order. *Size* is the default length as a fraction of $n_x$ (or
64 cells where none is given); the placement then shrinks a model that would not fit, and moves it
so the nose clears the inlet. The longest axis is taken over the mesh and any rotor's swept disc.
*Spinners* are the parts the *spinning parts* checkbox turns, *rotors* the actuator lines
(§3.12) and *ports* the engine faces (§3.11, §8.12).

| ID | Label | Group | Size | Ground | Spinners, rotors, ports |
|---|---|---|---|---|---|
| `sphere` | Sphere | Basic shapes | 0.15 | air | - |
| `cube` | Cube | Basic shapes | 0.12 | air | - |
| `cylinder` | Cylinder (cross-flow) | Basic shapes | 64 cells | air | - |
| `cone` | Cone (15 deg) | Basic shapes | 64 cells | air | - |
| `ball_spin` | Spinning ball (Magnus) | Basic shapes | 0.15 | air | 1 spinner (backspin) |
| `wing_naca0012` | NACA 0012 wing | Aerodynamic references | 64 cells | air | - |
| `ahmed_25deg` | Ahmed body (25 deg) | Aerodynamic references | 64 cells | air | - |
| `car_saloon` | Saloon (XE-like) | Road vehicles | 64 cells | rolling road | 4 wheels |
| `car_saloon_wing` | Saloon + rear wing | Road vehicles | 64 cells | rolling road | 4 wheels |
| `car_lmp` | Le Mans prototype | Road vehicles | 64 cells | rolling road | 1 exhaust |
| `car_open_wheel` | Open-wheel race car | Road vehicles | 64 cells | rolling road | 4 wheels; airbox and 2 radiator intakes, 1 exhaust |
| `lorry` | Lorry + trailer | Road vehicles | 0.40 | rolling road | - |
| `bullet_train` | Bullet train (leading car) | Road vehicles | 0.60 | rolling road | - |
| `airliner` | Airliner (A320-like) | Aircraft | 64 cells | air | 2 fan intakes, 2 exhausts |
| `concorde` | Supersonic delta (Concorde-like) | Aircraft | 0.50 | air | 2 intakes, 2 exhausts |
| `uav_reaper` | UAV (MQ-9-like) | Aircraft | 64 cells | air | 1 rotor (3-blade pusher propeller) |
| `uav_quad` | Quadcopter | Aircraft | 64 cells | air | 4 rotors (2 blades; diagonal pairs counter-rotate) |
| `capsule` | Re-entry capsule (Apollo-like) | Space | 0.15 | air | - |
| `rocket` | Rocket with grid fins (F9-like) | Space | 0.75 | air | 1 exhaust |
| `round_556` | 5.56 mm projectile | Munitions | 0.40 | air | - |
| `shell_105` | 105 mm howitzer shell | Munitions | 0.45 | air | 1 exhaust (base bleed) |
| `apfsds` | APFSDS dart | Munitions | 0.70 | air | - |
| `aim120` | AIM-120 AMRAAM-like | Munitions | 0.70 | air | 1 exhaust |
| `ballistic_missile` | Ballistic missile (Scud-like) | Munitions | 0.65 | air | 1 exhaust |
| `caarc_building` | Tall building (CAARC standard) | Wind engineering | 0.20 | fixed floor | - |
| `city_block` | City block | Wind engineering | 0.40 | fixed floor | - |
| `bridge_deck` | Bridge deck (Tacoma Narrows) | Wind engineering | 0.36 | air | - |
| `wind_turbine` | Wind turbine | Rotating | 0.117 | fixed floor | 1 rotor (3 blades, designed for a tip-speed ratio of 7) |
| `propeller` | Propeller | Rotating | 0.10 | air | 1 rotor (3 blades) |
| `frisbee` | Frisbee | Rotating | 64 cells | air | 1 spinner |

The rotors' design tip-speed ratios are 7 for the turbine and 4.5 for the propellers; the
quadcopter's rotors drive the air downwards. The engine ports, with the subsonic speed ratio (face
speed over the wind speed, times the throttle) and the transonic exit state (Mach number, and
static pressure and temperature over the freestream's):

| Model | Port | Speed ratio | Exit Mach | $p_e / p_\infty$ | $T_e / T_\infty$ |
|---|---|---|---|---|---|
| `car_lmp`, `car_open_wheel` | exhaust | 2.0 | 0.6 | 1.0 | 3.0 |
| `car_open_wheel` | airbox intake; radiator intakes | 0.6; 0.5 | - | - | - |
| `airliner` | fan intakes | 0.8 | - | - | - |
| `airliner` | exhausts (mixed turbofan jet) | 1.6 | 0.95 | 1.5 | 1.6 |
| `concorde` | intakes | 0.7 | - | - | - |
| `concorde` | exhausts (afterburning) | 2.0 | 1.5 | 1.8 | 4.0 |
| `rocket` | exhaust | 2.4 | 3.0 | 1.0 | 6.0 |
| `shell_105` | base bleed | 0.3 | 0.25 | 1.0 | 3.0 |
| `aim120` | exhaust | 2.4 | 2.5 | 1.5 | 5.0 |
| `ballistic_missile` | exhaust | 2.4 | 2.8 | 1.0 | 5.0 |

Intakes are walls in the transonic regime.

Road vehicles get a ride height of $\max(4, 0.06 L)$ cells, so the under-body gap is resolved.

## Command-line tools

All executables build into `build\<area>\RelWithDebInfo\` (for example
`build\tools\RelWithDebInfo\tunnel_run.exe`). The app and the tools find the repository root
themselves where they need it. Each prints its options with `-h` or `--help`, and refuses an
unknown option (exit code 2) rather than ignoring it.

### windoa_app

```
windoa_app [fast|balanced|fine|ultra] [subsonic|transonic] [--model ID] [--f16] [--no-vsync]
           [--frames N] [--shot FILE.png] [--warmup STEPS] [--show LIST] [--field NAME]
           [--spin R] [--power T] [--rotors L] [--aoa DEG] [--mach M] [--zoom F] [--view AZ,EL]
```

| Flag | Default | Meaning |
|---|---|---|
| `fast` / `balanced` / `fine` / `ultra` | `fast` | Grid preset |
| `subsonic` / `transonic` | `subsonic` | Starting regime |
| `--model ID` | `car_saloon` | Starting catalogue model; an unknown ID is refused with the list |
| `--f16` | off | Half-precision storage: an ungated approximation (§11.3) |
| `--no-vsync` | off | Present without waiting for the display; frames are then capped at 240 a second (uncapped, the window starved the solver's queue: 3,040 down to 1,280 MLUPS) |
| `--frames N` | - | Exit after N frames (scripts, smoke tests) |
| `--shot FILE` | - | With `--frames`: save the last frame as a PNG, panels included |
| `--warmup STEPS` | 0 | Run the solver this many steps before the first frame |
| `--show LIST` | - | Comma-separated view toggles: `q`, `dye`, `lines`, `nosmoke`, `nohaze`, `voxel`, `help`; `slice`, `hslice`, `xslice` (the vertical, horizontal and cross slice); `avg` (time averaging), `recirc` (mean reversed-flow shells), `lic` (slice texture), `arrows` (slice arrows), `timelines`, `nopaint`, `wallspeed`, `reversed`, `oil` (surface paint), `analysis` (the Analysis panel open), `probes` (two probes in the wake); `noui` (panels hidden), `noplots`, `nobox` (no tunnel outline), `nosurface` (the body hidden) |
| `--field NAME` | `speed` | `speed`, `pressure`, `vorticity`, `vortx`, `mach`, `schlieren`, `mean` (mean speed) or `turb` (turbulence intensity) |
| `--spin R` | - | Spin ratio for models with spinning parts |
| `--power T` | - | Engines on at throttle T, for models with engine ports |
| `--rotors L` | - | Rotors turning at tip-speed ratio L, for models with rotors |
| `--aoa DEG` | 0 | Starting pitch / angle of attack |
| `--mach M` | 0.8 | Starting transonic Mach number |
| `--zoom F` | - | Camera distance times F, aimed at the model (F < 1 is nearer) |
| `--view AZ,EL` | 35, 18 | Camera azimuth and elevation, degrees |

On exit the app prints its throughput and timings.

### tunnel_run

The app's `Tunnel` without a window: a headless sandbox session.

```
tunnel_run [--preset fast|balanced|fine|ultra] [--model ID] [--batches N] [--steps S]
           [--spin R] [--rotors L] [--power T] [--tu PCT] [--aoa DEG] [--size CELLS]
           [--speed U] [--walls sub|half] [--dye] [--f16] [--average]
           [--probe X,Y,Z ...] [--no-cache]
tunnel_run --list
tunnel_run --catalogue
```

| Flag | Default | Meaning |
|---|---|---|
| `--preset` | `fast` | Grid preset |
| `--model ID` | `ahmed_25deg` | Catalogue model, at its default placement |
| `--batches N`, `--steps S` | 40, 50 | Run N batches of S steps |
| `--spin R` | off | Spin ratio |
| `--rotors L` | parked | Rotors turning at tip-speed ratio L (§3.12) |
| `--power T` | off | Engines on at throttle T (§3.11) |
| `--tu PCT` | 0 | Inlet turbulence, per cent |
| `--aoa DEG` | 0 | Angle of attack |
| `--size CELLS` | the model's | Length of the model's longest axis, cells |
| `--speed U` | 0.05 | Freestream speed, lattice units (at most `u_max`, 0.11) |
| `--walls sub\|half` | `sub` | Sub-cell walls (§3.6) or half-way bounce-back (§3.5) |
| `--dye` | off | Run the dye from a wand ahead of the model |
| `--f16` | off | Half-precision storage (ungated) |
| `--average` | off | Time averaging once the flow has settled; reports the wake survey (§12.1, §12.2) |
| `--probe X,Y,Z` | - | A probe at the cell (X, Y, Z); up to four; reports its spectra (§12.4) |
| `--no-cache` | off | Neither restore nor save a settled flow, so the run develops from rest |
| `--list` | - | Print the catalogue's model IDs, groups and names |
| `--catalogue` | - | Voxelise every catalogue model at its default placement and report cells, areas and bounds (CTest `P2_catalogue`) |

It prints progress ten times, then the mean and standard deviation of Cd and Cl over the second half
of the run; the lift spectrum's strongest peaks as Strouhal numbers on the body's height, with the
acoustic-mode spacing and the shedding peak below the first mode (§12.3); each probe's peak per
component; the wake survey with `--average`; the rotors' tip-speed ratio, smoothed $C_T$ and
$C_P$ and swept area, and the engines' jet scale, for models that have them; and what the flow
cache did.

### lbm_run

A sphere (or any STL) in a tunnel, reporting throughput, health, Cd and a mass check.

```
lbm_run [nx ny nz] [--steps N] [--bgk] [--rr] [--ibb] [--spin S] [--u U] [--tau T]
        [--no-sponge] [--stl PATH [--size CELLS]]
```

Defaults: 256 x 96 x 96, 4,000 steps, regularised collision, $\tau = 0.504$, $C_s = 0.1$,
$U = 0.05$, a 24-cell sponge relaxing to the full freestream.

### euler_run

```
euler_run sphere [MACH]                     a sphere at Mach M (default 0.8)
euler_run profile                           GPU time per kernel category
euler_run equiv --save DIR | --check DIR    the Euler bit-identity baseline (§11.5)
```

### bench

```
bench [fast|balanced|fine|ultra|all] [--steps N]
```

Throughput of the lattice Boltzmann solver (plain, with moving boundaries, with dye, with f16
storage) and of the Euler solver, at each preset. Defaults: `all` (every preset but `ultra`),
1,000 steps. A measurement, not
a gate: run it twice before believing a difference.

### lbm_equiv

```
lbm_equiv [--f16] [--save DIR | --check DIR]
```

Checks every performance switch against the reference path bit for bit (CTest `P4_equiv`).
`--check build\p4_orig` also compares against results saved from the pre-optimisation kernel;
`--f16` reports the half-precision deviations.

### device_info and compute_selftest

`device_info` prints every Vulkan GPU and the capabilities the engine relies on;
`device_info --require-compute` exits non-zero without a compute-capable discrete GPU (CTest
`P0_device_info`). `compute_selftest` runs a SAXPY kernel end to end and checks it bit for bit
against the CPU (CTest `P0_compute`).

### The gates

`V1_poiseuille` to `V31_actuator_lines` take no arguments and exit non-zero on failure. Each prints
what it compared against and the measured value. See [`docs/VALIDATION.md`](VALIDATION.md).

## App controls

### Mouse and keys

| Input | Action |
|---|---|
| Right mouse drag | Orbit about the focus point |
| Middle mouse drag | Pan |
| Wheel, or Q / E | Zoom |
| W A S D | Pan the focus point |
| F | Focus on the model |
| Space | Pause / resume the solver |
| H | Hide the panels (the legends stay) |
| P | Save a PNG screenshot, panels included |
| F1 | Help window |
| Esc | Quit |
| Ctrl + click a slider | Type an exact value |

### Panels

| Panel | Control | Range | Meaning |
|---|---|---|---|
| Tunnel | regime | subsonic / transonic | The solver (§8) |
| Tunnel | flow speed | 0.005 - `u_max` | Freestream command; a change over 10 % re-develops (§9.2) |
| Tunnel | inlet turbulence % | 0 - 2 | Synthetic inlet turbulence (§7) |
| Tunnel | sim rate cap | flat out - 5,000 steps/s | Slow the solver to watch the flow evolve |
| Tunnel | skip developing, pause, reset flow | - | Leave the detector early; pause; restart from rest (never from the cache) |
| Tunnel (transonic) | Mach | 0.3 - 1.6 | Freestream Mach number; restart flow |
| Model | model | the catalogue | Applies the model's natural size and placement |
| Model | placement | aviation / rolling road / fixed ground | Free air, a floor moving at the wind speed, or a stationary floor |
| Model | size [cells] | 12 - 0.8 $n_x$ | Length of the longest axis |
| Model | pitch / AoA, yaw, roll | -180 - 180 deg | Attitude; a turn over 20 degrees restarts from rest (§9.2) |
| Model | pos x; pos y (air) or ride height (ground) | 0.1 - 0.8; 0.15 - 0.85 or 0 - 0.25 $n_y$ | Position |
| Model | ref area | frontal / planform / manual (10 - 5,000) | The coefficients' reference area (§4.4) |
| Model | spinning parts; spin ratio | 0 - 3 | Rim speed over wind speed (§9.3) |
| Model | rotors turning; tip-speed ratio | 0 - 12 (starts at 6) | Tip speed over wind speed; shows $C_T$, $C_P$ and a warning past 5 % of the section swept (§3.12) |
| Model | engines (jets / intakes); throttle | 0 - 3 | Each port's speed ratio (subsonic) or exit pressure (transonic) times the throttle (§3.11, §8.12) |
| Compare | save as A, save as B | - | Snapshot the coefficients and show B - A |
| View | surface | hidden / voxel / smooth | How the body is drawn |
| View | surface paint | none / pressure (Cp) / near-wall speed / reversed flow / oil flow | What the surface shows (§10.2, §10.7) |
| View | field | speed / pressure / vorticity / streamwise vorticity / Mach / schlieren / mean speed / turbulence intensity | The field for the haze and the slice (§10.1, §12.1); the last two need time averaging |
| View | flow haze; strength; floor | 0.2 - 4; 0 - 0.5 | The translucent field and its cut-off |
| View | field slice; slice pos | off / x-y / x-z / y-z; 0.02 - 0.98 | An opaque plane of the field |
| View | flow texture (LIC); velocity arrows; arrow spacing | -; -; 2 - 12 cells | The slice's in-plane flow as a texture or as arrows (§10.8, §10.9) |
| View | smoke; smoke as; pulse every | streaklines / timelines; 20 - 300 steps | Particle smoke from the wand, continuous or as pulsed lines (§10.9) |
| View | streamlines | - | Instantaneous field lines |
| View | dye smoke; density; colour by speed | 0.2 - 10 | The transported dye (§6) |
| View | vortex cores (Q); Q threshold | 0.1 - 10 | Q-criterion shells; threshold in multiples of the adaptive level (§10.3) |
| View | mean reversed flow | - | Shells where the time-averaged streamwise flow runs backwards (§12.1) |
| View | smoke tracks / fit model; height, width, y, z, detail | - | The smoke wand's position and size |
| View | time plots; refit plots; tunnel box | - | The Cd / Cl / Cm strips; the domain outline |
| View | render quality; render scale; field of view | 32 - 256 steps; 0.25 - 1; 20 - 90 deg | Ray-march samples; resolution; camera |
| Analysis | average the flow; restart | - | Time averaging of the developed flow (§12.1) |
| Analysis | show planes; auto plane; survey plane x | -; -; 1 - $n_x - 2$ | The wake survey's planes, drawn in the view, and the survey plane's position (§12.2) |
| Analysis | probes; x y z | 0 - 4; cells | Probe positions, drawn as coloured crosses (§12.4) |
| Analysis | signal; component | lift / a probe; u_x, u_y, u_z, rho | The signal whose spectrum and history are plotted (§12.3) |

## Test presets and labels

| Command | Runs |
|---|---|
| `ctest --preset dev` | Everything: the smoke checks and all 31 gates (about 19 minutes) |
| `ctest --preset quick` | Everything except the label `long` (about 105 s) |
| `ctest --preset dev -L gate` | The gates only |
| `ctest --preset dev -R V17` | One test by name |

| Test | Label | What |
|---|---|---|
| `P0_device_info`, `P0_compute` | - | The toolchain and the compute path |
| `P2_catalogue` | - | Every catalogue model voxelises inside the tunnel |
| `P3_tunnel_smoke` | - | A short headless sandbox session, with dye |
| `P4_equiv` | - | The performance switches are exact identities |
| `V1` - `V31` | `gate` (and `long` for the 16 slow ones) | The validation gates |

## Files written at run time

All are gitignored and safe to delete.

| Path | Written by | What |
|---|---|---|
| `cache/flow/` | the app, `tunnel_run` | Settled flows (about 86 MB each at the fast preset) |
| `screenshots/` | the app (P key) | PNG screenshots |
| `windoa_imgui.ini` | the app | The panel layout |
| `models/` | the user | STL files listed under *Imported* in the model menu |

## The flow-cache key

Everything that determines a settled flow (§9.6): the model's content hash; the length, angle of
attack, yaw, roll and position; the ground mode and ride height; whether it spins, the spin ratio
and every spinner's definition; whether the rotors turn, the tip-speed ratio, the kernel width and
every rotor's definition; whether the engines run, the throttle, the jet-speed cap and every
port's definition; the wall-speed cap; the inlet turbulence; the speed; the grid; $\tau$, $C_s$,
the collision operator, the wall mode, the sponge, the floor height and the storage precision;
plus a hash of the physics source code and the cache format. Any change to the solver, voxeliser or
tunnel source therefore invalidates every entry.
