# Validation

Every model in wind-oa ships with a **gate**: an executable that checks it against an external
reference and exits non-zero if it misses. There are 31 of them, V1 - V31, and they are the backbone
of the project: they are the evidence that the solvers compute the right physics, not merely
plausible pictures.

This page is the catalogue and the method. The models themselves are specified in
[`docs/THEORY.md`](THEORY.md), where each section lists its own gates alongside the equations they
hold, so a reader studying a model finds its gates on the same page. This page is for the complete
list, for what a gate is *for*, and for how to add one.

- [Why a gate is not a regression test](#why-a-gate-is-not-a-regression-test)
- [The three kinds of reference](#the-three-kinds-of-reference)
- [Running them](#running-them)
- [The catalogue](#the-catalogue)
- [Where they live, and why there](#where-they-live-and-why-there)
- [Adding one](#adding-one)
- [What is deliberately not gated](#what-is-deliberately-not-gated)

## Why a gate is not a regression test

A regression test records what the code did yesterday and complains when that changes. A gate names
an **external reference**, such as an analytic solution, published data or an identity the model
must satisfy, and checks against that.

- A regression test says *the answer changed*.
- A gate says *the answer is wrong*.

The working rule follows from the difference: **if a change makes a gate fail, the reason is
understood before anything is re-baselined.** A tolerance is never adjusted to make a failure go
away.

## The three kinds of reference

| Kind | What it proves | Example |
|---|---|---|
| **Closed form** | The solver converges to the analytic answer | The Poiseuille parabola; the exact Riemann solution of the shock tube; the oblique-shock relations of the wedge; the iso-surface of a sphere's distance field; a standing sound wave between the side walls |
| **Published data** | The solver reproduces experiment or benchmark computation | The Ghia cavity profiles; sphere drag at Re 100 and 300; the Strouhal number at Re 200; the NACA 0012 at Mach 0.8 |
| **Identity or invariant** | A property that must hold whatever the numbers | Mass conservation, globally and through every plane; a switched-off subsystem equals its absence bit for bit; a cache restore matches a never-restored control; the wake survey's momentum balance against the force balance |

## Running them

```
ctest --preset dev                 # every gate and the smoke checks (about 19 minutes)
ctest --preset quick               # skip the gates labelled long (about 105 s)
ctest --preset dev -L gate         # the gates only
ctest --preset dev -R V17          # one gate
```

Each gate prints what it compared against and one `[PASS]` or `[FAIL]` line per check, with the
measured value and the band, and ends with the gate's verdict.

## The catalogue

Grouped by the theory section that specifies each model. *Result* is the measured value as the gate
last printed it.

### §1 - §2 The lattice and collision

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V1 | BGK + Guo recovers the analytic channel flow | Poiseuille parabola, relative L2 < 1 % | L2 0.074 % |
| V3 | Lid-driven cavity at Re 1000, with the mass anchor | Ghia et al. (1982), RMS < 0.03 | RMS 0.0064 |
| V4 | Regularised collision: Poiseuille, the LES cavity, spheres at Re 1000 and 10,000 | Parabola; Ghia (RMS < 0.05); sphere drag, bands 50 and 60 %, and stability | L2 0.065 %; RMS 0.0085; Cd 0.676 / 0.658, stable |
| V5 | Recursive regularisation recovers the same viscosity | Parabola, L2 < 1 % | L2 0.050 %; at cs = 0 the driven cavity is stable to tau 0.52 (plain regularised) and 0.53 (RR) |

### §3 Boundary conditions

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V2 | The open tunnel holds its density and conserves mass locally; the anchor pins it and leaves u exact | Drift < 1e-8 per step; the mass flux through every x-face equal to 1e-4; u bit-identical | Drift +2.7e-10 per step; face mass flux spread 4.2e-5; u unchanged |
| V8 | Moving walls produce the Magnus force | Sign, monotonic growth, 1.5 < abs(Cl) < 9 at spin ratio 2 | Cl -2.78 / -5.59 at ratios 1 / 2 |
| V9 | Interpolated bounce-back places the wall at fraction q | Shifted-wall Poiseuille, u_max within 1.5 % | 0.15 - 0.70 %; informational: sphere Cd 1.171 half-way -> 1.125 interpolated (error 7.5 % -> 3.2 %) |
| V10 | The sponge absorbs sound without changing the answer | Reflection < 0.05; Ahmed Cd within 1 % | R 0.687 -> 0.010; Cd +0.03 %, force noise down 4x |
| V28 | A mesh's link fractions place its wall at the surface (§3.6), and the sandbox runs on them | Ray intersection with the true sphere, RMS within 0.01 of a link and worst within 0.05; the true sphere's drag within 0.5 %; 8,000 healthy steps at U 0.11 for six thin-featured models | RMS 0.0030, worst 0.016 (the signed distance alone: 0.036, 0.176); Cd within 0.23 %, half-way 6.0 % higher; all six healthy, max abs(u) 0.20 - 0.28 |
| V29 | An engine port injects the mass it prescribes, and the body feels the jet (§3.11) | The injection within 1 %; the momentum balance within 2 % for an exhaust, an intake and the control; in the app's tunnel, the survey within 2 % and the mass gained within 5 % | Mass 0.11 % (exhaust), 0.22 % (intake); force 0.04 % in all three; app survey 0.14 %, mass 1.4 % |
| V31 | Rotors as actuator lines obey momentum theory and load the air as the balance requires (§3.12) | Linear theory, a = C_T / 4 within 3 %; the thrust against the momentum deficit within 2 %; the app's turbine: survey within 3 %, 0 < C_P < 16/27, healthy at U 0.11 | a 2.0 % below C_T / 4; deficit 1.07 % below the thrust; survey 0.98 %, C_T 0.824, C_P 0.565; healthy. Measured, not gated: at C_T 0.5 the induction 8.0 % below 1-D theory |

### §4 Forces and coefficients

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V6 | Sphere drag at Re 100 and 300 | Schiller-Naumann, within 15 % | Cd 1.212 / 0.766 (11 / 12 %) |
| V7 | Vortex shedding off a cylinder at Re 200 | St = 0.196, within 15 % | St 0.212 (8 %) |
| V14 | Ahmed body: low-Re drag band, slant lift, recirculating wake | Low-Re bands: 0.6 < Cd < 1.8, Cl > 0, reverse flow behind the base | Cd 1.400, Cl +0.168, reverse flow behind the base |
| V15 | Ahmed drag falls with Reynolds number | The published trend | Cd 1.411 -> 1.136 -> 1.212 over Re 1,250 - 20,000 |

### §5 Geometry

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V12 | Thick bodies exact, thin sheets sealed, overlapping parts a union | The exact centre-inside set; a D3Q19 flood fill; the union plus a fillet | All three exact |
| V13 | The triangle-mesh pipeline reproduces the analytic sphere | Solid count and drag within 5 % | 2,176 cells (3.2 %), Cd 1.235 (1.8 %) |

### §6 - §7 Transport

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V20 | Dye advects at U, spreads with D = (tau - 1/2)/4, never leaks, stays bounded | The analytic Gaussian blob; exact zero beyond a wall; 0 <= C <= 1.1 | Variance within 0.9 %, mass 0.04 %; leak exactly 0; C <= 1.070 |
| V21 | Inlet turbulence is solenoidal, at the requested intensity, and quiet | Closed form; the request within 25 %; non-solenoidal noise | Divergence 4e-8; Tu 1.75 % for 2 %; one fifth of the noise's sound |

### §8 The compressible Euler solver

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V17 | Shock tube: all three wave families | The exact Riemann solution | L1 0.28 %, shock on the exact cell, symmetric |
| V18 | Mach 2 over a 15 degree voxel wedge | theta-beta-M | Shock 45.50 against 45.34 degrees; wall pressure exact |
| V19 | NACA 0012 at Mach 0.8, alpha 1.25 degrees | AGARD-AR-211 bands | Cl 0.291, Cd 0.0147, upper shock 0.64c |
| V27 | Lift and pitching moment of a NACA 0012 at Mach 0.5, alpha +-2 degrees (§8.11) | Mirror antisymmetry; the moment transfer to the leading edge; Kutta-Joukowski within 5 % | Sums 1.5e-6 / 7e-7; transfer exact to 8e-9; circulation lift 2.9 % below the surface lift. Measured, not gated: the aerodynamic centre at 0.225 c (§8.9) |
| V30 | An under-expanded sonic jet places its Mach disc (§8.12) | Ashkenas and Sherman, x_M / D = 0.67 sqrt(p0 / p_a), within 10 % at p0 / p_a = 20 and 40; peak axial Mach above 3 | 2.92 D vs 3.00 (2.4 %); 4.12 D vs 4.24 (2.7 %); peak Mach 4.7 and 5.7 |

### §9 The sandbox loop

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V11 | The guard sees a blow-up; spin at the cap is stable; long windows are exact | Poisoned fields; 8,000 healthy steps; a double-precision sum | All detected; healthy at U 0.05 and 0.11; window exact to 5e-7 |
| V22 | The settling detector is right on synthetic and real signals | Five synthetic invariants; the long-run mean within 2 % | All five; cold start settles within 0.4 %, warm restart within 0.3 % |
| V23 | A restored flow is the settled flow | A never-restored control | Means identical to 0.00 %, frames within 0.1 % |

### §10 What the renderer derives

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V16 | Marching cubes extracts the iso-surface of a sampled field | Two analytic spheres: closed and oriented, V - E + F = 2, vertices within 0.02 cells, normals outward, area and volume within 1 % | Closed, V - E + F = 2; radial error 0.006 / 0.010 cells; area -0.08 / -0.21 %, volume -0.15 / -0.39 % |

### §12 Statistics and signals

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V24 | Time averages are exact, and the mean vortex street has its published structure | A double-precision host accumulation; a steady flow's identity; the symmetric mean wake, v_rms peaking on the centreline and u_rms off it | Accumulator within 2.7e-8; mean = instantaneous to the f32 floor; mean wake symmetric to 1e-4 U (instantaneous 1.9 U); u_rms peaks 0.82 D off the centreline |
| V25 | The wake survey recovers the drag | The force balance over the same window: steady sphere within 1 %, the app's tunnel within 2 % | 0.34 % and 0.56 % at two planes; 0.60 % on the app's Ahmed body; mass flux equal to 4e-5 |
| V26 | Spectra find frequencies, the shedding Strouhal number and the acoustic modes | Tones at uneven sampling; St 0.196 at Re 200; v at f and u at 2f on the centreline; c_s / (2 n_y) for a standing sound wave | Tones within 0.08 %; St 0.212 (zero crossings 0.212); probes exact to 0.03 %; sound wave within 0.01 % |

## Where they live, and why there

Each gate is one executable under `validation/`, registered with CTest by `windoa_gate(...)` in
`validation/CMakeLists.txt`, with the label `gate` (and `long` for the sixteen slow ones). Gates use only the engine's public headers: they exercise exactly the API the app uses,
including `solver_config(TunnelSettings)` where a gate claims to test what the app runs.

- `validation/gate.hpp` is the shared harness: the pass / fail ledger and the small numerics the
  gates share.
- `validation/cases.hpp` and `validation/euler_cases.hpp` hold the setups that several gates share,
  so that each configuration is written once.

The gates are plain executables rather than a test framework's cases, for two reasons: the project
avoids dependencies, and each gate prints its evidence in full, which is what makes a failure
diagnosable.

## Adding one

1. **Name the external reference first:** the equation, the paper or the identity, and the pass
   criterion.
2. **Specify the model** in the right section of [`docs/THEORY.md`](THEORY.md), and list the gate
   in that section's *Validation gates* table.
3. **Write the gate** as `validation/VN_name.cpp`, using `gate::run` and one `g.check` per
   property, printing the measured value and the band. If its setup is shared, put it in
   `cases.hpp`.
4. **Register it** with `windoa_gate(VN_name VN_name.cpp)` in `validation/CMakeLists.txt`,
   adding `LONG` if it takes over a minute.
5. **Add its row** to the catalogue above.

## What is deliberately not gated

- **The renderer.** It is a view, not a model; what its colours mean is recorded in §10 so that a
  picture can still be read quantitatively.
- **Performance.** Throughput is a measurement, not a correctness property (`tools/bench`). What
  is protected is that the performance work does not change the physics: `P4_equiv` and the
  Euler baseline check it bit for bit (§11.5).
- **The interactive UI.** Its numbers come from the same `Tunnel` the gates drive.
- **A heavily loaded rotor against one-dimensional momentum theory.** At $C_T = 0.5$ the actuator
  disc's induction reads 8 % below $(1 - \sqrt{1 - C_T})/2$. Blockage, the kernel's width and the
  viscosity were each varied and none accounts for it, so its cause is open and it is reported
  rather than gated (§3.12). The thrust itself is gated against the momentum the air gains.
- **An exhaust plume beyond its Mach disc.** The inviscid solver holds the plume's shear layer by
  numerical diffusion, so its spreading and the decay of its shock cells are not predicted (§8.12).
- **The catalogue's geometric detail.** The models are stylised; `P2_catalogue` checks that every
  one voxelises inside the tunnel, and V28 that the thin-featured ones run at top speed.

### The two worth singling out

- **Half-precision storage.** It is an approximation, not an identity, and every gate runs f32. Its
  effect on the Ahmed body is measured ([`docs/GUIDE.md`](GUIDE.md#10-a-worked-study-end-to-end)):
  a third faster per step, but slower to a given confidence in a mean, with a spurious lift peak.
  It therefore stays opt-in and ungated, a tool for pictures rather than numbers. Gating it would
  mean running the ladder a second time with f16 storage.
- **Three-dimensional transonic bodies.** The compressible solver is gated on a shock tube, a
  two-dimensional wedge and a two-dimensional aerofoil. A 3-D body at Mach 0.8 runs (a sphere
  briefly reaches a local Mach of about 2.2 early in its impulsive start), but no external reference
  covers it yet.
