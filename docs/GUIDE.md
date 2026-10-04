# User guide

How to set the tunnel up, drive it, get defensible numbers out of it and keep it healthy. It is the
single source for operating wind-oa; it assumes no C++ and no fluid dynamics. How the parts work is
explained in [`docs/MODEL.md`](MODEL.md), every flag and default is listed in
[`docs/REFERENCE.md`](REFERENCE.md), and the physics is specified in [`docs/THEORY.md`](THEORY.md).

> **The tunnel runs at sandbox Reynolds numbers, about $10^3 - 10^4$, not at full scale.** Absolute
> coefficients are therefore qualitative; comparisons and trends at the same grid and speed are
> what it is for (Part 3).

## Contents

- **Part 1 - Getting it running**
  - [1. Setting up](#1-setting-up)
  - [2. Running the app](#2-running-the-app)
  - [3. Driving the app](#3-driving-the-app)
- **Part 2 - Building situations**
  - [4. Models](#4-models)
  - [5. Transonic mode](#5-transonic-mode)
  - [6. The flow cache](#6-the-flow-cache)
- **Part 3 - Getting numbers out**
  - [7. Study design: the two rules](#7-study-design-the-two-rules)
  - [8. The headless tools](#8-the-headless-tools)
  - [9. Reading the output](#9-reading-the-output)
  - [10. A worked study, end to end](#10-a-worked-study-end-to-end)
  - [11. Averages, the wake survey and spectra](#11-averages-the-wake-survey-and-spectra)
  - [12. Keeping findings honest](#12-keeping-findings-honest)
- **Part 4 - Maintaining it**
  - [13. Testing](#13-testing)
  - [14. Extending it](#14-extending-it)
  - [15. Troubleshooting](#15-troubleshooting)

# Part 1 - Getting it running

## 1. Setting up

### 1.1 Install the toolchain

wind-oa builds on Windows with three tools, and nothing is downloaded at build time:

1. **Visual Studio 2022** with the "Desktop development with C++" workload (MSVC).
2. **CMake 3.25 or newer** (the installer's default location, `C:\Program Files\CMake`, is
   expected).
3. **The LunarG Vulkan SDK** (1.4.x), which provides the Vulkan headers and loader, the `glslc`
   shader compiler and the validation layers.

A Vulkan 1.3 GPU is required; the project is developed on an AMD Radeon RX 7900 XT. The only
third-party library, Dear ImGui, is already vendored in `third_party/`.

### 1.2 Editor setup

Any editor works. With VS Code, the Microsoft C/C++ extension provides IntelliSense and the
`clang-format` binary the project's `.clang-format` uses (format on save). Source files are ASCII
only.

### 1.3 First build

From the repository root, in any command prompt:

```
cmake --preset dev
cmake --build --preset dev
```

The first command configures a Visual Studio 2022 solution in `build\`; the second builds it in
RelWithDebInfo. `--build --preset debug` or `release` selects the other configurations. Shaders are
compiled to SPIR-V at build time and embedded in the executables, so nothing is compiled when the
app starts.

### 1.4 Check it worked

```
build\tools\RelWithDebInfo\device_info
ctest --preset quick
```

`device_info` lists the GPUs the engine can see and the capabilities it relies on (an asynchronous
compute queue, f16 storage, linear filtering of 32-bit float images). `ctest --preset quick` runs the
smoke checks and the 15 short validation gates in about 105 s; every line should read `Passed`.

## 2. Running the app

```
build\app\RelWithDebInfo\windoa_app [fast|balanced|fine|ultra] [transonic] [--model ID]
```

| Preset | Grid | Cells | Typical speed while rendering |
|---|---|---|---|
| fast (default) | 256 x 96 x 96 | 2.4 M | about 3,150 MLUPS |
| balanced | 320 x 128 x 128 | 5.2 M | |
| fine | 384 x 160 x 160 | 9.8 M | about 2,400 MLUPS with vortex cores on |
| ultra | 512 x 192 x 192 | 18.9 M | about 160 steps a second: for patient studies, mostly headless |

MLUPS is millions of lattice cell updates per second. The app finds the repository root itself:
the flow cache (`cache/`), screenshots (`screenshots/`), imported STL files (`models/`) and the
panel layout (`windoa_imgui.ini`) live there, and all are gitignored. It starts with the saloon on a
rolling road; `--model ID` starts with any catalogue model (`tunnel_run --list` prints the IDs, and
REFERENCE describes them). `windoa_app --help` lists every option; an unknown option or model ID
is refused with the list rather than ignored.

The solver runs flat out on the GPU's asynchronous compute queue and the window draws on the
graphics queue, so the window stays smooth however hard the solver works.

## 3. Driving the app

### 3.1 Controls

| Input | Action |
|---|---|
| Right mouse drag | Orbit about the focus point |
| Middle mouse drag | Pan |
| Wheel, or Q / E | Zoom |
| W A S D | Pan the focus point |
| F | Focus on the model |
| Space | Pause / resume the solver |
| H | Hide the panels (the legends stay, for screenshots) |
| P | Save a PNG screenshot, panels included |
| F1 | Help |
| Esc | Quit |
| Ctrl + click a slider | Type an exact value |

Panels can be dragged, docked and resized, and the layout is remembered. A layout that does not
fit the window (one saved on a larger screen, or after the window shrinks) is put back to the
starting arrangement for the current size; *reset layout* in the View panel does the same on
demand.

### 3.2 The Tunnel panel

The regime (subsonic lattice Boltzmann or transonic Euler), the run status, the force dashboard and
the flow controls.

- **DEVELOPING** after any change means the wake is forming and washing through the tunnel, which
  takes 2 - 4.5 flow-throughs; that is physics, not slowness. The numbers can be trusted once it
  reads **SETTLED**. *skip developing* hands control back early.
- **Cd, Cl, Cs** are the drag, lift and side-force coefficients, over $q A_{\mathrm{ref}}$; **Cm_z**
  is the pitching moment about the model's centre, over $q A_{\mathrm{ref}} L$.
- **Re_sim** is the Reynolds number actually simulated, about $10^3$. A car at motorway speed is
  about $4 \times 10^6$, so absolute Cd is qualitative and comparisons are trustworthy (Part 3).
- **flow speed** is in lattice units (Mach = $u \sqrt{3}$, kept under about 0.19). Changes are
  slew-limited so the lattice never shocks; a change of more than 10 % re-develops.
- **inlet turbulence %** adds real-tunnel freestream turbulence, a divergence-free gust field
  carried in through the inlet (0 is a clean inlet; real tunnels run about 0.1 - 2 %).
- **sim rate cap** slows the solver, to watch a flow evolve.
- If the flow blows up (most often from a very high speed, a thin under-body gap or spin), it is
  reset from rest with a note saying why. A second blow-up within a minute pauses instead.

### 3.3 The Model panel

- Choosing a model applies its natural setting: a size that resolves it and a placement (free
  air, a rolling road for road vehicles, a fixed floor for buildings and the turbine), fitted so
  the nose clears the inlet.
- *placement*: aviation (free air), rolling road (the floor moves at the wind speed, as in an
  automotive tunnel) or fixed ground (a stationary floor). Keep a few cells of ride height: a body
  flush on the road makes a one-cell under-body channel the grid cannot resolve.
- *size*, *pitch / yaw / roll* and *position* re-voxelise the model half a second after a slider
  stops moving. A new body, or a turn of more than 20 degrees, restarts the flow from rest.
- *ref area*: frontal (the car convention), planform (the wing convention) or manual. A wing's Cl
  only looks right against its planform area.
- *spinning parts* appears on models with wheels or other spinning bodies. The spin ratio is rim
  speed over wind speed; surfaces are capped at 0.08 lattice units for stability, and the panel
  says when the cap is limiting.
- *rotors turning* appears on models with rotors (the turbine, the propellers, the quadcopter).
  The blades are not solid: they are lines of lift and drag forces computed from the local flow,
  drawn as turning outlines (section 4.6). The *tip-speed ratio* is tip speed over wind speed,
  starting at 6 (the turbine is designed for 7, the propellers for 4.5); parked, the blades still
  take the wind. The panel shows the rotor's thrust and power coefficients and warns when the
  rotor sweeps more than 5 % of the tunnel's section, where blockage moves them from their
  free-air values.
- *engines (jets / intakes)* appears on models with engine ports (the jets, the race cars, the
  rockets and missiles). Exhausts blow and intakes draw at their speed ratio times the *throttle*
  times the wind speed, capped at 0.12 lattice units (the panel says when); in transonic mode the
  exhausts emit a hot, under-expanded jet instead (section 4.7). The force balance then reads the
  net force, thrust included.

### 3.4 The View panel

| View | Shows |
|---|---|
| *surface* smooth / voxel | The body at the wall the solver uses (smooth), or its cells |
| *surface paint* | What the model is painted with: **pressure (Cp)**, red compression and blue suction against an upstream reference plane (a Pitot-static); **near-wall speed**, the flow one cell off the surface, which scales with the skin friction; **reversed flow**, blue where the near-wall flow runs upstream (separated); or **oil flow**, streaks along the near-wall flow as in a tunnel oil-film test |
| *field* + *flow haze* | The chosen field through the tunnel, translucent; undisturbed flow is invisible |
| *field slice* | The same field on a plane: vertical, horizontal or cross-stream; optionally with a **flow texture (LIC)**, noise smeared along the in-plane flow so every streamline shows at once, and **velocity arrows** on a grid |
| *smoke* | Particles from the smoke wand (the cyan rectangle): **streaklines**, released continuously like tunnel smoke, or **timelines**, a line released across the wand every few steps like a pulsed hydrogen-bubble wire, whose bending shows the velocity profile |
| *dye smoke* | A transported concentration from nozzles on the wand, which fills the wake; coloured by local speed (violet still, yellow fast) or white |
| *streamlines* | Instantaneous field lines; they differ from smoke when the wake is unsteady |
| *vortex cores (Q)* | Where rotation beats strain, as translucent shells (green at the threshold, amber at 4x); the threshold adapts to the flow's own vorticity |
| *mean reversed flow* | Translucent teal shells where the time-averaged streamwise flow runs backwards: the mean recirculation bubbles (needs time averaging) |

The fields are **speed** (deviation from the freestream: blue slower, red faster), **pressure**
(Cp), **|vorticity|**, **streamwise vorticity** (signed, so trailing vortex pairs show as blue and
orange tubes), **Mach** (blue subsonic, white at the sonic line, red supersonic), **schlieren**
(density gradients as white light, which shows shocks), and two that need time averaging: **mean
speed** and **turbulence intensity** (the rms of the fluctuations, 0 - 20 % of U). The legends
along the top follow what is on.

### 3.5 Compare and the plots

*save as A* and *save as B* snapshot the coefficients and show the difference: the honest way to use
a sandbox at this Reynolds number. Along the bottom the Cd, Cl and Cm plots scroll; their scale only
ever expands, so small fluctuations never look like waves, and *refit plots* refocuses on the recent
history (which happens by itself once settled).

### 3.6 The Analysis panel

- *average the flow* accumulates the mean and the variance of every cell once the flow has
  settled; a new operating point restarts the window, which refills once the flow settles again.
  It feeds the mean-speed and turbulence fields, the reversed-flow shells and the wake survey.
- The **wake survey** measures the drag a second way: from the fall in the averaged flow's
  momentum between an upstream plane and a survey plane behind the model (both drawn in the
  view). It is shown beside the force balance's mean over the same window, with the wake profile
  at the survey plane. The two agree within about 1 % once a few flow-throughs are averaged.
- **Probes** (up to four, coloured crosses in the view) record the velocity and density at a
  point. The **spectrum** of the lift or of a probe component gives the shedding frequency as a
  Strouhal number, $\mathrm{St} = f h / U$ on the body's height. Dashed lines mark the tunnel's
  acoustic modes: sound ringing between the side walls, which shows in every spectrum and is not
  shedding.

### 3.7 Screenshots and scripted runs

P saves a PNG of the window, panels included, to `screenshots/` (compressed: about 0.2 - 0.5 MB at
1600 x 900). For a reproducible picture, the app can be scripted from the command line:

```
windoa_app --model airliner --aoa 6 --show "q,lines,nosmoke" --warmup 8000 --frames 120 --shot out.png
```

`--warmup` develops the flow before the first frame, `--frames` exits after that many and `--shot`
saves the last one. `--zoom` moves the camera nearer the model and `--view AZ,EL` sets its azimuth
and elevation in degrees; `--size` sets the model's length in cells, `--mach` the transonic Mach
number, and `--rotors` and `--power` start the rotors and the engines. The view flags (`--show`) cover
every overlay, the slices (`slice`, `hslice`, `xslice`) and the clean-picture switches (`noui`,
`noplots`, `nobox`, `nosurface`); REFERENCE lists them all.

The README's images come from these runs (each restores a settled flow from the cache when it
has one, so a second run is quicker; the transonic runs take their frames after the warm-up because
the Mach number slews once per batch, and the warm-up runs as a few long batches):

```
windoa_app --model car_saloon --zoom 0.6 --view 40,22 --warmup 30000 --frames 240 --shot app.png
windoa_app --model cylinder --field mean --show "avg,hslice,lic,recirc,nohaze,nosmoke,noui,noplots,nobox" --zoom 0.55 --view 90,75 --warmup 40000 --frames 120 --shot cylinder_mean_wake.png
windoa_app fine --model wind_turbine --rotors 7 --show "q,nohaze,nosmoke,noui,noplots,nobox" --zoom 0.3 --view 70,22 --warmup 25000 --frames 120 --shot turbine_tip_vortices.png
windoa_app --model ahmed_25deg --show "timelines,nohaze,noui,noplots" --zoom 0.5 --view 60,30 --warmup 25000 --frames 240 --shot ahmed_timelines.png
windoa_app --model car_saloon --show "oil,nohaze,nosmoke,noui,noplots" --zoom 0.3 --view 150,30 --warmup 30000 --frames 60 --shot saloon_oil_flow.png
windoa_app transonic --mach 0.8 --model wing_naca0012 --aoa 2 --field mach --show "slice,noui,noplots,nobox" --zoom 0.3 --view 90,0 --warmup 8000 --frames 120 --shot transonic_wing.png
windoa_app transonic --mach 1.5 --model aim120 --size 90 --power 1 --field schlieren --show "slice,noui,noplots,nobox,nosurface" --zoom 0.55 --view 90,0 --warmup 20000 --frames 1000 --shot missile_plume.png
```

# Part 2 - Building situations

## 4. Models

### 4.1 The catalogue

Thirty stylised procedural models in eight groups: basic shapes, aerodynamic references (the NACA
0012 wing and the Ahmed body), road vehicles, aircraft, space, munitions, wind engineering and
rotating parts. Each is built in memory from boxes, cylinders, lofts, aerofoil sections and bodies
of revolution, so nothing is downloaded or shipped. They have the right proportions; they are not
replicas. The full list, with sizes, ground modes, spinning parts, rotors and engine ports, is
in REFERENCE.

### 4.2 Placement and attitude

A model is scaled so that its longest axis equals the size slider, centred at the position sliders,
and rotated by pitch, yaw and roll about its centre (roll applied first, then pitch, then yaw).
Positive pitch is nose-up for a model facing the wind. In the two ground modes the model is then
dropped onto the floor, its lowest point at the ride height. The Model panel warns if a model
extends outside the tunnel.

### 4.3 Reference areas

Frontal area is the number of cells in the model's shadow along the flow; planform area is its
shadow from above. Choose the convention of the field the model belongs to (frontal for vehicles,
planform for wings) and keep it fixed across any comparison.

### 4.4 Spinning parts

Road cars' wheels, the open-wheel car's wheels, the frisbee and the Magnus ball can spin. Wheels
roll so the contact patch moves downstream; the ball has backspin (Magnus lift upwards). All parts
are scaled by one common factor if any would exceed the wall-speed cap, so their relative speeds
are preserved. Rotors are not spinning walls (section 4.6).

### 4.5 Imported models

Drop a binary or ASCII STL file into `models/` and restart the app; it appears under *Imported*.
Author it nose towards $-x$ (the flow runs in $+x$). It is scaled so its longest axis equals the
size slider, so its units do not matter. Imperfect meshes are tolerated (a winding-rule fill with a
two-of-three axis vote), as are overlapping parts and zero-thickness sheets; parts thinner than a
cell are kept one cell thick. Imported STL files are gitignored and never committed.

### 4.6 Rotors

A real rotor's tips move several times faster than the wind, faster than any wall the lattice can
move. The turbine, the propeller, the UAV's pusher propeller and the quadcopter's four rotors are
therefore actuator lines (THEORY §3.12): each blade is a line of points that read the local flow,
take lift and drag from an aerofoil section's polar at the local angle of attack, and push back on
the air. The blades are drawn as turning outlines, and their wake (the helical tip vortices behind
a turbine, the downwash under a quadcopter) is real flow. The turbine's blades follow a
blade-element design for a tip-speed ratio of 7; the propellers and the quadcopter's rotors are
pitched to drive the air.

The thrust and power coefficients are over the swept area, $C_T = T / (\tfrac12 \rho U^2 A)$ and
$C_P = P / (\tfrac12 \rho U^3 A)$, positive power meaning the wind drives the rotor. The
dashboard's Cd, Cl and Cm remain the body's (the tower, nacelle or fuselage); the wake survey's
force balance adds the rotor's streamwise force, since the air feels both. A closed
tunnel holds the flow round a large rotor, so these depart from free-air values as the swept area
grows: past about 5 % of the section a turbine's power coefficient can even pass the Betz limit
(16/27). The catalogue turbine is sized below that; enlarge it for pictures, not for numbers.

### 4.7 Engines

Engine ports are discs on the surface: intakes at the jets' fan faces and the race cars' airboxes
and radiators, exhausts at the nozzles, the motors of the rockets and missiles, and the base-bleed
unit of the 105 mm shell. In the subsonic regime each face blows (or draws) at its speed ratio
times the throttle times the wind speed, a velocity boundary (THEORY §3.11); the lattice is
isothermal, so a subsonic jet carries mass and momentum but no heat. In transonic mode an exhaust
emits its exit state, a Mach number, pressure and temperature (THEORY §8.12): an under-expanded jet
then shows its barrel shock, Mach disc and shock diamonds in the *schlieren* field. Intakes are
walls in transonic mode.

## 5. Transonic mode

The lattice Boltzmann solver is trusted to about Mach 0.3 and cannot form shocks. *regime:
TRANSONIC* hands the same tunnel and model to a compressible Euler solver (finite volume: MUSCL,
HLLC, SSP-RK2, with image-point walls from the exact distance to the surface).

- A *Mach* slider (0.3 - 1.6) replaces the speed. *schlieren* shows the shocks and *Mach* the
  supersonic pockets; the panel shows the peak local Mach number.
- Cd, Cl and Cm_z are pressure forces and their moment only: the solver is inviscid, so there is
  wave drag but no skin friction.
- After a switch or a new model the flow develops for two flow-throughs.
- Smoke, dye, streamlines, vortex cores, spin and rotors are subsonic-only (engines work in both
  regimes); switching back resumes the paused lattice flow.
- At the default model sizes, read it for shock patterns and trends. The NACA 0012 at 128 cells a
  chord reads its lift about 17 % low, because the walls lie on a voxel grid, and the shortfall
  sits aft, so the pitching moment's centre reads ahead of the quarter chord (THEORY §8.9).

## 6. The flow cache

Once a configuration has settled, its flow is saved (about 86 MB at the fast preset; a 512 MB budget,
oldest dropped first). Returning to exactly that configuration (the same model, placement, speed,
spin, rotors, engines, walls and turbulence) restores it in well under a second. A change to the solver or voxeliser source
invalidates every entry automatically. *reset flow* always starts from rest, and `cache/` is safe
to delete.

# Part 3 - Getting numbers out

## 7. Study design: the two rules

### Rule 1 - wait for SETTLED, then average

A coefficient read during DEVELOPING describes a wake that has not formed. Wait for SETTLED, then
average over a window long enough to cover the wake's own unsteadiness: a shedding body's
instantaneous drag oscillates, and only its mean is a property of the configuration. The headless
tools report the mean and standard deviation over the second half of a run for this reason.

### Rule 2 - compare, do not quote

At $\mathrm{Re} \sim 10^3$ an absolute drag coefficient is not a prediction for the full-scale
object: the Ahmed body reads about 1.4 at $\mathrm{Re}_L = 1{,}250$ against 0.285 in experiment at
$4.3 \times 10^6$, and the tunnel's walls add blockage. A *difference* between two configurations
on the same grid, at the same speed and with the same reference area is far more reliable, because
most of the Reynolds-number and blockage effects are common to both.

### The corollary that catches people

Everything except the variable under study must match: grid, speed, size, reference area, ground
mode and storage precision. A comparison across presets, or between a model at 64 cells and the same
model at 96, measures the resolution as much as the configuration.

## 8. The headless tools

The tools drive the engine without a window, so a study can run unattended and repeat exactly.

| Tool | For |
|---|---|
| `tunnel_run` | The app's sandbox headless: any catalogue model, size, speed, spin, rotors, engines, angle, turbulence, dye, f16 or half-way walls, reporting settled means, the lift spectrum, the rotors' coefficients and, with `--average`, the wake survey |
| `lbm_run` | A sphere or any STL in a bare solver, with the collision and boundary options exposed |
| `euler_run sphere M` | A sphere in the compressible solver at Mach M |
| `bench` | Throughput at each preset |

Each prints its options with `--help` and refuses an unknown one; the flags are also in REFERENCE.
A typical `tunnel_run`:

```
build\tools\RelWithDebInfo\tunnel_run --model ahmed_25deg --batches 300 --steps 100
```

## 9. Reading the output

`tunnel_run` prints ten progress lines and a summary:

```
step   30000  u 0.0500  Cd +0.6798  Cl +0.0981  Cm +0.0172  max|u| 0.066  SETTLED after 2.5 flow-throughs  (3240 MLUPS)
second-half mean: Cd 0.6779 (sd 0.0013)  Cl 0.0960 (sd 0.0102)  over 150 batches
lift spectrum over 16000 steps: peak St 0.135 (period 2523 steps, amplitude 0.013); next peaks at St 0.265 0.393  (h 17 cells; acoustic modes every St 1.022)
shedding (strongest peak below the first acoustic mode): St 0.135
spin scale 1.000, cache: saved to cache (0.5 s) (1 entries)
```

| Field | Meaning |
|---|---|
| `u` | The applied inlet speed (it ramps from zero over the first 2,000 steps) |
| `Cd`, `Cl`, `Cm` | The dashboard's smoothed coefficients at that moment |
| `max|u|` | The fastest fluid cell; a value approaching 0.3 is a warning, and the guard trips at 0.5 |
| phase | RAMPING, DEVELOPING, SETTLED (with the flow-throughs it took to settle) or NOT SETTLED |
| MLUPS | Throughput so far |
| second-half mean | The mean and standard deviation of the smoothed coefficients over the second half of the run |
| lift spectrum | The strongest peaks of the lift's spectrum over the developed flow, as Strouhal numbers on the body's height, and the spacing of the tunnel's acoustic modes |
| shedding | The strongest lift peak below the first acoustic mode: the shedding frequency |
| spin scale | Below 1 when the wall-speed cap limited the requested spin |
| rotors | Models with rotors: turning (`--rotors L`) or parked, the tip-speed ratio, the smoothed $C_T$ and $C_P$ at the end of the run, and the swept area as a fraction of the section |
| engines | Models with engine ports: on (`--power T`) or off, and the jet scale, below 1 when the jet-speed cap limited the throttle |

The standard deviation is of smoothed, correlated values, so it understates the uncertainty of the
mean; use it to compare the scatter of two runs, not as an error bar on its own. A repeated run
reproduces these numbers exactly, because every reduction in the solver is deterministic; only the
throughput varies from run to run.

## 10. A worked study, end to end

**Question:** does half-precision storage change the Ahmed body's drag?

```
build\tools\RelWithDebInfo\tunnel_run --model ahmed_25deg --batches 300 --steps 100 --no-cache
build\tools\RelWithDebInfo\tunnel_run --model ahmed_25deg --batches 300 --steps 100 --no-cache --f16
```

| Storage | Throughput | Settled after | Second-half mean Cd | sd |
|---|---|---|---|---|
| f32 | 3,240 MLUPS | 2.5 flow-throughs | 0.6779 | 0.0013 |
| f16 | 4,022 MLUPS | 2.5 flow-throughs | 0.6768 | 0.0052 |

Half precision is 24 % faster per step, and its mean is 0.2 % lower. However, its scatter is 4
times larger, so matching f32's confidence in the mean needs $(0.0052 / 0.0013)^2 = 16$ times as
many samples: at 1.24 times the speed, the same confidence costs about 13 times the wall-clock
time. Its lift spectrum also misses the shedding: f32 finds a clean peak at St 0.135 with its
harmonics at 0.265 and 0.393, while f16 reports St 0.119 and a spurious peak at 0.64. The faster
storage is therefore slower for this kind of question, which is why it stays opt-in and ungated
(every gate runs f32); it remains useful where a picture, not a mean, is the goal.

The same study answers a second question with one flag: `--walls half` runs the half-way staircase
instead of the sub-cell walls. It reads Cd 0.7135 (sd 0.0021), 5.3 % more drag, with two thirds of the
lift and a weaker shedding peak (St 0.116): on a body with sharp edges and a slant, where the
wall sits matters at this resolution (THEORY §3.9).

## 11. Averages, the wake survey and spectra

Three measurements sit on top of the flow (THEORY §12). They read the solver's fields and never
write them, so switching them on changes nothing about the flow.

- **Time averages** turn an unsteady wake into its mean: the mean recirculation bubble, the mean
  wake deficit and the turbulence intensity. Average over several shedding cycles; a window
  shorter than a few periods leaves the last cycle's imprint.
- **The wake survey** is a second measurement of the drag, from a control-volume momentum
  balance of the averaged flow. In free air it agrees with the force balance within about 1 %
  (V25); a larger difference means the window is still short or the survey plane sits too close
  to the body. In ground mode the floor's shear lies inside the volume, so the survey includes
  it.
- **Spectra** of the lift and of probes give frequencies. The Strouhal number reported for the
  lift is the strongest peak below the tunnel's first acoustic mode; peaks at St of order one and
  above, at the dashed lines, are sound between the side walls.

Headless, `tunnel_run --average --probe X,Y,Z` reports all three at the end of the run:

```
build\tools\RelWithDebInfo\tunnel_run --model ahmed_25deg --batches 1500 --steps 21 --average
```

## 12. Keeping findings honest

- A finding should name its grid, speed, size, reference area and storage precision, and the
  window it averaged over.
- If a result depends on a model the gates do not cover (for example a new catalogue model at an
  unusual attitude), treat it as provisional.
- The gates are the evidence that the solver computes the right physics at all
  ([`docs/VALIDATION.md`](VALIDATION.md)); run `ctest --preset quick` after any change before
  trusting a new number.

# Part 4 - Maintaining it

## 13. Testing

| Command | Runs | Time |
|---|---|---|
| `ctest --preset quick` | The smoke checks and the 15 short gates | about 105 s |
| `ctest --preset dev` | Everything, all 31 gates included | about 19 minutes |
| `ctest --preset dev -R V17` | One test by name | - |

### The validation layers

`cmake --preset validation-layers` then `cmake --build --preset validation-layers` builds a Debug
tree in `build-vl\` with the Vulkan validation layers enabled at run time. Running the app from that
tree reports any API misuse on the console. Synchronisation validation is enabled through a
temporary `vk_layer_settings.txt` in the repository root containing
`khronos_validation.validate_sync = true` (delete it afterwards).

### The bit-identity baselines

```
build\tools\RelWithDebInfo\lbm_equiv --check build\p4_orig
build\tools\RelWithDebInfo\euler_run equiv --check build\p4_euler
```

Both must print PASS after any change that is not meant to alter the physics. The baselines live in
the build tree; if it is wiped, regenerate them from a known-good build with `--save` before making
changes.

## 14. Extending it

### 14.1 Adding a catalogue model

Add a builder function to `engine/src/catalogue.cpp` that returns a `geometry::Mesh` authored nose
towards $-x$, and register it with `add(group, id, label, builder, size, ground, spinners)` in
`build_entries()`. Then run `tunnel_run --catalogue`: every model must voxelise inside the tunnel at
its default placement (CTest `P2_catalogue` runs the same check).

### 14.2 Adding a tunable

Add the field, with a comment giving its unit and reason, to `TunnelSettings` in
`engine/include/windoa/tunnel.hpp`; tunables live there rather than as literals in the code. If it
changes the flow, add it to the flow-cache key in `Tunnel::operating_point_key`, and list it in
REFERENCE.

### 14.3 Adding a validation gate

New physics needs a new gate before it is made fast. Name the external reference first, write the
gate as an executable under `validation/` that prints what it compared against and exits non-zero
on failure, and register it in `validation/CMakeLists.txt` with `windoa_gate(...)` (adding `LONG` if
it takes over a minute). The full procedure is in [`docs/VALIDATION.md`](VALIDATION.md#adding-one).

### 14.4 Adding a shader

Write the GLSL under the module's `shaders/` directory and add it with
`windoa_add_shaders(<target> shaders/x.comp)`; it is compiled at build time and available as
`windoa::spv::x` from `#include "x_spv.hpp"`. Shaders are never loaded from disk at run time.

## 15. Troubleshooting

### Setup

- **`cmake` is not recognised.** A terminal opened before CMake was installed does not have it on
  its path: open a new one, or prepend `C:\Program Files\CMake\bin`.
- **`device_info` finds no GPU.** The Vulkan driver or the SDK's loader is missing; reinstall the
  GPU driver, then the SDK.

### Running

- **The flow keeps diverging.** The note on the Tunnel panel names the likely cause. Reduce the
  speed, raise the ride height of a road vehicle, or lower the spin ratio.
- **DEVELOPING never ends.** A strongly unsteady configuration may not settle; the detector gives
  up at 6 flow-throughs and says NOT SETTLED. Average over a long window instead (Part 3).
- **The model is clipped by the tunnel.** The Model panel warns when it extends outside; reduce the
  size or the turn.
- **A panel has gone missing.** It may be folded or docked somewhere unexpected: *reset layout* in
  the View panel puts every panel back (H toggles them all).

### Studies

- **Two runs disagree.** Check that everything except the studied variable matched (section 7),
  including the storage precision and the reference area.
- **The spectrum's strongest peak is at St of order one or above.** That is an acoustic mode of
  the tunnel (the dashed lines), not shedding; the shedding Strouhal number is reported
  separately, below the first mode.
- **The wake survey and the force balance differ by more than about 1 %.** Average for longer,
  move the survey plane further behind the body, and check the ground mode (the floor's shear
  counts in the survey).
- **A gate fails after a change.** Understand why before changing any tolerance;
  [`docs/VALIDATION.md`](VALIDATION.md) explains what each gate checks.
