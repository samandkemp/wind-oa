# Validation

Every model in wind-oa ships with a **gate**: an executable that checks it against an external
reference and exits non-zero if it misses. There are 23 of them, V1 - V23, and they are the backbone
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
| **Closed form** | The solver converges to the analytic answer | The Poiseuille parabola; the exact Riemann solution of the shock tube; the oblique-shock relations of the wedge; the iso-surface of a sphere's distance field |
| **Published data** | The solver reproduces experiment or benchmark computation | The Ghia cavity profiles; sphere drag at Re 100 and 300; the Strouhal number at Re 200; the NACA 0012 at Mach 0.8 |
| **Identity or invariant** | A property that must hold whatever the numbers | Mass conservation; a switched-off subsystem equals its absence bit for bit; a cache restore matches a never-restored control |

## Running them

```
ctest --preset dev                 # every gate and the smoke checks (about 13 minutes)
ctest --preset quick               # skip the gates labelled long (about 40 s)
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
| V4 | Regularised collision: Poiseuille, the LES cavity, spheres at Re 1000 and 10,000 | Parabola; Ghia (RMS < 0.05); sphere drag, bands 50 and 60 %, and stability | L2 0.065 %; RMS 0.0085; Cd 0.667 / 0.665, stable |
| V5 | Recursive regularisation recovers the same viscosity | Parabola, L2 < 1 % | L2 0.050 %; at cs = 0 the driven cavity is stable to tau 0.52 (plain regularised) and 0.53 (RR) |

### §3 Boundary conditions

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V2 | The open tunnel holds its density; the anchor pins it and leaves u exact | Drift < 1e-8 per step; u bit-identical | Drift +1.5e-10 per step; u unchanged |
| V8 | Moving walls produce the Magnus force | Sign, monotonic growth, 1.5 < abs(Cl) < 9 at spin ratio 2 | Cl -2.78 / -5.59 at ratios 1 / 2 |
| V9 | Interpolated bounce-back places the wall at fraction q | Shifted-wall Poiseuille, u_max within 1.5 % | 0.16 - 0.70 %; sphere Cd 1.172 -> 1.165 with IBB |
| V10 | The sponge absorbs sound without changing the answer | Reflection < 0.05; Ahmed Cd within 1 % | R 0.685 -> 0.010; Cd +0.18 %, force noise down 7x |

### §4 Forces and coefficients

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V6 | Sphere drag at Re 100 and 300 | Schiller-Naumann, within 15 % | Cd 1.214 / 0.768 (11 / 12 %) |
| V7 | Vortex shedding off a cylinder at Re 200 | St = 0.196, within 15 % | St 0.212 (8 %) |
| V14 | Ahmed body: low-Re drag band, slant lift, recirculating wake | Low-Re bands: 0.6 < Cd < 1.8, Cl > 0, reverse flow behind the base | Cd 1.413, Cl +0.221, reverse flow behind the base |
| V15 | Ahmed drag falls with Reynolds number | The published trend | Cd 1.424 -> 1.138 -> 1.176 over Re 1,250 - 20,000 |

### §5 Geometry

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V12 | Thick bodies exact, thin sheets sealed, overlapping parts a union | The exact centre-inside set; a D3Q19 flood fill; the union plus a fillet | All three exact |
| V13 | The triangle-mesh pipeline reproduces the analytic sphere | Solid count and drag within 5 % | 2,176 cells (3.2 %), Cd 1.236 (1.9 %) |

### §6 - §7 Transport

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V20 | Dye advects at U, spreads with D = (tau - 1/2)/4, never leaks, stays bounded | The analytic Gaussian blob; exact zero beyond a wall; 0 <= C <= 1.1 | Variance within 0.9 %, mass 0.04 %; leak exactly 0; C <= 1.062 |
| V21 | Inlet turbulence is solenoidal, at the requested intensity, and quiet | Closed form; the request within 25 %; non-solenoidal noise | Divergence 4e-8; Tu 1.75 % for 2 %; one fifth of the noise's sound |

### §8 The compressible Euler solver

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V17 | Shock tube: all three wave families | The exact Riemann solution | L1 0.28 %, shock on the exact cell, symmetric |
| V18 | Mach 2 over a 15 degree voxel wedge | theta-beta-M | Shock 45.50 against 45.34 degrees; wall pressure exact |
| V19 | NACA 0012 at Mach 0.8, alpha 1.25 degrees | AGARD-AR-211 bands | Cl 0.291, Cd 0.0147, upper shock 0.64c |

### §9 The sandbox loop

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V11 | The guard sees a blow-up; spin at the cap is stable; long windows are exact | Poisoned fields; 8,000 healthy steps; a double-precision sum | All detected; healthy at U 0.05 and 0.11; window exact to 5e-7 |
| V22 | The settling detector is right on synthetic and real signals | Five synthetic invariants; the long-run mean within 2 % | All five; cold start settles within 1.4 %, warm restart within 0.3 % |
| V23 | A restored flow is the settled flow | A never-restored control | Means identical to 0.00 %, frames within 0.1 % |

### §10 What the renderer derives

| Gate | Property | Checked against | Result |
|---|---|---|---|
| V16 | Marching cubes extracts the iso-surface of a sampled field | Two analytic spheres: closed and oriented, V - E + F = 2, vertices within 0.02 cells, normals outward, area and volume within 1 % | Closed, V - E + F = 2; radial error 0.006 / 0.010 cells; area -0.08 / -0.21 %, volume -0.15 / -0.39 % |

## Where they live, and why there

Each gate is one executable under `validation/`, registered with CTest by `windoa_gate(...)` in
`validation/CMakeLists.txt`, with the label `gate` (and `long` for the twelve that take over a
minute). Gates use only the engine's public headers: they exercise exactly the API the app uses,
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

### The two worth singling out

- **Half-precision storage.** It is an approximation, not an identity, and every gate runs f32. Its
  effect on the Ahmed body is measured ([`docs/GUIDE.md`](GUIDE.md#10-a-worked-study-end-to-end)),
  but it is not gated, which is the main reason it stays opt-in. Gating it would mean running the
  ladder a second time with f16 storage.
- **Three-dimensional transonic bodies.** The compressible solver is gated on a shock tube, a
  two-dimensional wedge and a two-dimensional aerofoil. A 3-D body at Mach 0.8 runs (a sphere
  briefly reaches a local Mach of about 2.2 early in its impulsive start), but no external reference
  covers it yet.
