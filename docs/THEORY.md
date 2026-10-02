# Theory

The quantitative specification: every model in the tunnel, worked from the general form of its
mathematics down to the particular form the code evaluates, with the dials it exposes and the gate
that holds it honest.

This is the bottom layer of the documentation. [`README.md`](../README.md) says what the project
is; [`docs/GUIDE.md`](GUIDE.md) says how to set it up and drive it; [`docs/MODEL.md`](MODEL.md)
walks the same models in plain terms with worked numbers and the code path for each. **Read those
first.** This page is for the reader who wants to know *why* an equation has the form it does,
what was rejected on the way to it, and what it is checked against.

It is one file on purpose. The source cites these sections as `THEORY N.M` (the source is kept
ASCII, so the section sign appears only here), and landing on the right section of one document
is better than landing on a table of contents that sends the reader somewhere else. Sections are
therefore never renumbered.

## Contents

- [Conventions these sections share](#conventions-these-sections-share)
- [0. The strands](#0-the-strands)
- [1. The lattice and its units](#1-the-lattice-and-its-units)
- [2. Collision](#2-collision)
- [3. Boundary conditions](#3-boundary-conditions)
- [4. Forces and coefficients](#4-forces-and-coefficients)
- [5. Geometry](#5-geometry)
- [6. The passive scalar (dye)](#6-the-passive-scalar-dye)
- [7. Synthetic inlet turbulence](#7-synthetic-inlet-turbulence)
- [8. The compressible Euler solver](#8-the-compressible-euler-solver)
- [9. The sandbox loop](#9-the-sandbox-loop)
- [10. What the renderer derives](#10-what-the-renderer-derives)
- [11. Performance identities](#11-performance-identities)
- [References](#references)

## Conventions these sections share

- **Lattice units.** Unless a section says otherwise, lengths are in cells and times in solver
  steps, so the lattice spacing and the time step are both 1. Velocities are therefore fractions
  of a cell per step, and densities are normalised so that the undisturbed fluid has
  $\rho = 1$. The compressible solver uses a different, stated system (§8.1).
- **Cell order and centres.** Fields are stored in C order, `[x][y][z]` with `z` fastest. Cell
  $(i, j, k)$ occupies the unit cube whose centre is $(i + \tfrac12, j + \tfrac12, k + \tfrac12)$.
  The analytic shapes of §5.5 are the one exception: they test the cell *index*, the convention
  the gates that use them are specified in.
- **Distributions.** The distributions $f_i$ are stored direction-major, `[Q][x][y][z]`, in the
  direction order of §1.1.
- **Flags.** Every cell is `FLUID` (0), `OBSTACLE` (1, a model surface whose forces are
  counted), `WALL` (2, a solid with no force bookkeeping, such as a fixed floor) or `LID` (3, a
  solid with a prescribed velocity, such as a rolling road).
- **Post-collision values** carry a star: $f_i^*$ is the value leaving a cell after collision,
  and streaming moves it to the neighbour along $\mathbf{e}_i$.

## 0. The strands

Six strands of modelling make up the tunnel, and each does a job the others cannot.

### 0.1 Lattice Boltzmann flow - the subsonic air

A D3Q19 lattice Boltzmann method evolves particle distributions on the grid; their moments are the
density and velocity of a weakly compressible fluid whose viscosity is set by a relaxation time.
It is fast, local and naturally parallel, which is what makes an interactive tunnel possible
(§1, §2).

### 0.2 Boundary conditions - what the air meets

A velocity inlet, a pressure outlet with an absorbing sponge, free-slip side walls, and bounce-back
at every solid link (half-way or interpolated), with moving-wall terms for spinning parts and a
rolling road (§3).

### 0.3 Compressible Euler - the transonic air

The lattice Boltzmann method is trusted only to about Mach 0.3 and cannot form shocks. A
finite-volume Euler solver on the same grid and flags carries the tunnel from Mach 0.3 to 1.6:
MUSCL reconstruction, the HLLC flux and SSP-RK2 time stepping, with image-point ghost cells at the
walls (§8).

### 0.4 Geometry - from triangles to cells

A triangle mesh becomes cell flags by a winding-number fill with a two-of-three vote, a thin-feature
shell that keeps sub-cell sheets solid, and an exact signed distance for the compressible walls
(§5).

### 0.5 Transport - what the air carries

A passive dye on its own D3Q7 lattice (§6) and a synthetic, divergence-free inlet turbulence
(§7).

### 0.6 Measurement - what the numbers are

Forces by momentum exchange, reduced deterministically (§4); a detector that decides when the
coefficients have settled; and a cache that restores a settled flow exactly (§9).

### 0.7 Two properties that cut across all six

- **The identity discipline.** Every optional subsystem (spin, turbulence, sponge, dye, the
  performance switches) reduces to an exact identity when switched off: the solver steps bit for
  bit as if the feature did not exist. Compile-time switches are Vulkan specialisation
  constants, so a disabled feature is absent from the compiled kernel rather than skipped at run
  time.
- **Determinism.** No floating-point atomics are used in any reduction. Sums are per-workgroup
  trees in a fixed order followed by a single-group reduction, and host-side sums of partials
  are taken in double precision. A run is therefore repeatable bit for bit on the same GPU and
  driver.

## 1. The lattice and its units

### 1.1 The D3Q19 lattice

The nineteen lattice velocities $\mathbf{e}_i$ are the rest vector, the six face neighbours and the
twelve edge neighbours of a cube, in this order:

| $i$ | $\mathbf{e}_i$ | $w_i$ |
|---|---|---|
| 0 | $(0, 0, 0)$ | $1/3$ |
| 1 - 6 | $(\pm 1, 0, 0)$, $(0, \pm 1, 0)$, $(0, 0, \pm 1)$ | $1/18$ |
| 7 - 18 | $(\pm 1, \pm 1, 0)$, $(\pm 1, 0, \pm 1)$, $(0, \pm 1, \pm 1)$ in pairs | $1/36$ |

The lattice sound speed is $c_s^2 = 1/3$. The tables in `engine/shaders/lattice.glsl` also give,
for each direction, its opposite (for bounce-back) and its specular reflections across a $y$ or
$z$ face (for free-slip walls).

### 1.2 Lattice units and the physical mapping

The kinematic viscosity follows from the relaxation time $\tau$:

$$\nu = c_s^2 \left(\tau - \tfrac12\right),$$

so the Reynolds number of a body of length $L$ cells in a freestream of $U$ cells per step is

$$\mathrm{Re} = \frac{U L}{\nu}, \qquad \tau = \frac{3 U L}{\mathrm{Re}} + \frac12,$$

and the Mach number is $\mathrm{Ma} = U / c_s = U \sqrt{3}$. The tunnel's default speed
$U = 0.05$ is Mach 0.087, and the speed slider stops at $U = 0.11$ (Mach 0.19), because the
method's compressibility error grows as $\mathrm{Ma}^2$.

The app's default $\tau = 0.504$ gives $\nu = 0.00133$, so a 64-cell body at $U = 0.05$ is
simulated at $\mathrm{Re} = 2{,}400$: about three orders of magnitude below a road car at
motorway speed. This is the single most important fact for reading the tunnel's numbers, and the
reason absolute coefficients are qualitative (§4.5).

### 1.3 Grid conventions

The grid presets are $256 \times 96 \times 96$ (fast), $320 \times 128 \times 128$ (balanced) and
$384 \times 160 \times 160$ (fine). The flow runs in $+x$; $+y$ is up and $+z$ is lateral. The
inlet is the plane $x = 0$ and the outlet the plane $x = n_x - 1$.

### 1.4 Deliberate limitations

- **Weak compressibility.** The method recovers the Navier-Stokes equations to $O(\mathrm{Ma}^2)$
  and has no energy equation; it is not used above Mach 0.19 (the compressible solver of §8
  takes over).
- **Uniform grid.** There is no grid refinement, so resolution near a body is the same as in the
  far field.

### 1.5 Validation gates (V1)

| Gate | Property | Checked against |
|---|---|---|
| V1 | The viscosity of §1.2 is what the solver produces | The analytic Poiseuille parabola, relative L2 < 1 % |

## 2. Collision

### 2.1 Moments and the equilibrium

After streaming, the density and velocity of a fluid cell are the moments of its pulled
distributions, with half of the body force $\mathbf{g}$ added (the Guo scheme):

$$\rho = \sum_i f_i, \qquad \rho \mathbf{u} = \sum_i f_i \mathbf{e}_i + \tfrac12 \mathbf{g}.$$

The equilibrium is the second-order expansion of a Maxwellian:

$$f_i^{\mathrm{eq}}(\rho, \mathbf{u}) = w_i \rho \left(1 + 3\, \mathbf{e}_i \cdot \mathbf{u}
+ \tfrac92 (\mathbf{e}_i \cdot \mathbf{u})^2 - \tfrac32\, \mathbf{u} \cdot \mathbf{u}\right).$$

### 2.2 BGK with Guo forcing

The single-relaxation-time (BGK) collision relaxes every distribution towards equilibrium at the
rate $\omega = 1 / \tau_{\mathrm{eff}}$ and adds the forcing term of Guo, Zheng and Shi (2002):

$$f_i^* = f_i - \omega \left(f_i - f_i^{\mathrm{eq}}\right) + \left(1 - \tfrac{\omega}{2}\right) S_i,
\qquad S_i = w_i \left(3 (\mathbf{e}_i - \mathbf{u}) \cdot \mathbf{g}
+ 9 (\mathbf{e}_i \cdot \mathbf{u})(\mathbf{e}_i \cdot \mathbf{g})\right).$$

BGK is accurate and cheap, but it becomes unstable as $\tau \to \tfrac12$, which caps the
Reynolds number a given grid can reach. That ceiling is what the regularised operators lift.

### 2.3 Projected regularisation

The regularised operator of Latt and Chopard (2006) discards the non-hydrodynamic content of the
non-equilibrium distribution and rebuilds it from the viscous stress alone. The non-equilibrium
momentum flux is measured, and the force dipole that Guo forcing adds to it is removed, so that
only the viscous part remains:

$$\Pi^{\mathrm{neq}}_{\alpha\beta} = \sum_i \left(f_i - f_i^{\mathrm{eq}}\right) e_{i\alpha} e_{i\beta}
+ \tfrac12 \left(u_\alpha g_\beta + u_\beta g_\alpha\right).$$

The rebuilt non-equilibrium part and the collision are then

$$f_i^{\mathrm{neq}} = \frac{w_i}{2 c_s^4} \left(e_{i\alpha} e_{i\beta} - c_s^2 \delta_{\alpha\beta}\right)
\Pi^{\mathrm{neq}}_{\alpha\beta}, \qquad
f_i^* = f_i^{\mathrm{eq}} + (1 - \omega) f_i^{\mathrm{neq}} + \tfrac12 S_i,$$

with $1/(2 c_s^4) = 4.5$. The forcing coefficient is deliberately $\tfrac12$ rather than Guo's
$(1 - \omega/2)$: writing the collision in terms of the clean non-equilibrium part turns Guo's
term into $S_i/2$, and using $(1 - \omega/2)$ here lost momentum at a rate of about
$(1-\omega)F/2$ per step (a 46 % Poiseuille error at $\tau = 0.52$).

### 2.4 Recursive regularisation

The recursive-regularised (RR) operator of Jacob, Malaspinas and Sagaut (2018) also rebuilds the
third-order non-equilibrium Hermite coefficients, recursively from the second-order ones. D3Q19
supports only the six "two-same-one-different" coefficients:

$$a^{(3)}_{\alpha\alpha\beta} = 2 u_\alpha \Pi^{\mathrm{neq}}_{\alpha\beta} + u_\beta \Pi^{\mathrm{neq}}_{\alpha\alpha}
\quad (\alpha \neq \beta),$$

and they add

$$\Delta f_i^{\mathrm{neq}} = \frac{w_i}{2 c_s^6} \sum_{\alpha \neq \beta} a^{(3)}_{\alpha\alpha\beta}\,
e_{i\beta} \left(e_{i\alpha}^2 - c_s^2\right),$$

with $1/(2 c_s^6) = 13.5$ (the index multiplicity folds $1/(6 c_s^6)$ into it). RR recovers the
same Navier-Stokes viscosity as BGK (V5). It does not, on its own, extend the stable Reynolds
number of an enclosed driven cavity at zero eddy viscosity; the hybrid variant (HRR) that would
is deliberately not implemented.

### 2.5 Smagorinsky large-eddy model

An eddy viscosity adds to the molecular one, computed from the magnitude of the clean
non-equilibrium flux (Hou et al., 1996):

$$|\Pi| = \sqrt{\Pi^{\mathrm{neq}}_{\alpha\beta} \Pi^{\mathrm{neq}}_{\alpha\beta}}, \qquad
\tau_t = \tfrac12 \left(\sqrt{\tau_0^2 + 18 \sqrt{2}\, C_s^2\, |\Pi| / \rho} - \tau_0\right), \qquad
\omega = \frac{1}{\tau_0 + \tau_t}.$$

$C_s$ is the Smagorinsky constant (0.1 in the app; 0 disables the model). The model needs no
velocity gradients from neighbouring cells, because $\Pi^{\mathrm{neq}}$ is local, which keeps
the collision a purely local operation.

### 2.6 Deliberate limitations

- **No wall model.** Near-wall turbulence is not modelled; at sandbox Reynolds numbers the
  boundary layers are laminar or transitional in any case.
- **Smagorinsky everywhere.** The eddy viscosity is not damped at walls (no van Driest
  correction); it is small where the flow is smooth, because $|\Pi|$ is.

### 2.7 Validation gates (V1, V3 - V5)

| Gate | Property | Checked against |
|---|---|---|
| V1 | BGK + Guo recovers the analytic channel flow | Poiseuille parabola, L2 < 1 % |
| V3 | BGK at Re 1000 in a driven cavity | Ghia, Ghia and Shin (1982), RMS < 0.03 |
| V4 | Regularised: Poiseuille, the LES cavity, spheres at $\tau \to \tfrac12$ stable | Parabola; Ghia (RMS < 0.05); published sphere drag (bands 50 - 60 %) |
| V5 | RR recovers the same viscosity | Poiseuille parabola, L2 < 1 % |

## 3. Boundary conditions

Streaming is a *pull*: each fluid cell gathers $f_i$ from its upwind neighbour $\mathbf{x} -
\mathbf{e}_i$. A boundary condition is therefore the rule for what to pull when that neighbour is
outside the grid or solid.

### 3.1 Velocity inlet

The plane $x = 0$ is a hard Dirichlet condition: every distribution there is set to
$f_i^{\mathrm{eq}}(1, \mathbf{u}_{\mathrm{in}})$ each step, and a cell that would pull from
$x < 0$ pulls the same equilibrium. A soft inlet (pulling the equilibrium alone) lets the tunnel
settle about 15 % below the commanded speed under load, which is why the plane is pinned.

### 3.2 Pressure outlet

A cell that would pull from $x \geq n_x$ pulls the equilibrium at the reference density with the
velocity of the last interior plane, taken from the **previous** step:

$$f_i = f_i^{\mathrm{eq}}\left(1, \mathbf{u}(n_x - 2, y, z, t - 1)\right).$$

Pinning $\rho = 1$ anchors the tunnel's pressure level; without it the tunnel over-pressurised
during spin-up and the flow choked about 25 % below the commanded speed. The macroscopic fields
are double-buffered alongside $f$, so the outlet reads exactly the previous step; with a single
buffer the read would race with the same kernel's writes, and its outcome would depend on
scheduling. Anything that writes the distributions directly must also refresh $\rho$ and
$\mathbf{u}$, or the outlet imposes a stale velocity for one step.

### 3.3 Absorbing sponge

Acoustically, the pressure outlet is a pressure-release end, which reflects sound: 68 % of an
incident pressure pulse returns up the tunnel. A band of width $W$ cells before the outlet
therefore relaxes the post-collision state towards a target equilibrium:

$$f_i^* \leftarrow f_i^* + \sigma \left(f_i^{\mathrm{eq}}(1, \mathbf{u}_t) - f_i^*\right), \qquad
\sigma = 0.35 \left(1 - \frac{d}{W}\right)^2, \quad d = n_x - 1 - x < W.$$

The target velocity $\mathbf{u}_t$ is the full freestream (target 1, the app's choice) or the
local velocity (target 0, density only). With $W = 24$ and target 1 the reflection falls to
1.0 %, and the Ahmed body's drag moves by 0.18 % (V10).

### 3.4 Free-slip and periodic faces

On a free-slip $y$ or $z$ face, a pull that leaves the grid is reflected specularly: the
direction's normal component is flipped (the `SPEC_Y` / `SPEC_Z` tables) and the pull is taken
from inside the domain. This is an inviscid wall, so the tunnel's side walls grow no boundary
layer. Any axis may instead be periodic, in which case the pull wraps around.

### 3.5 Half-way bounce-back

At a solid neighbour, the distribution that left the cell towards the wall returns reversed:

$$f_i(\mathbf{x}, t + 1) = f^*_{\bar\imath}(\mathbf{x}, t),$$

where $\bar\imath$ is the direction opposite to $i$ (the one pointing into the wall). This places
the no-slip wall half-way along the link, to second order. A body is therefore represented by its
voxel staircase, with the wall half a cell outside the solid cells' centres.

### 3.6 Interpolated bounce-back

Where the true surface is known, the linear interpolated bounce-back of Bouzidi, Firdaouss and
Lallemand (2001) places the wall at the fraction $q \in (0, 1]$ of the link, measured from the
fluid cell:

$$f_i = \begin{cases}
2q\, f^*_{\bar\imath}(\mathbf{x}) + (1 - 2q)\, f^*_{\bar\imath}(\mathbf{x} + \mathbf{e}_i), & q < \tfrac12,\\[4pt]
\dfrac{1}{2q} f^*_{\bar\imath}(\mathbf{x}) + \dfrac{2q - 1}{2q} f^*_i(\mathbf{x}), & q \geq \tfrac12.
\end{cases}$$

The first branch needs the cell one step away from the wall to be fluid; where it is not (a gap
of one cell), the link falls back to half-way bounce-back. Link fractions are stored as 8-bit
values $q = n/255$; the default $n = 128$ stands for $q = 0.502$, effectively half-way. Only the
analytic sphere fills them today (§5.5); the sandbox uses half-way bounce-back.

### 3.7 Moving walls

A wall moving with velocity $\mathbf{u}_w$ adds the momentum term of Ladd (1994) to every link
it reflects:

$$f_i \leftarrow f_i + 6\, w_i\, \mathbf{e}_i \cdot \mathbf{u}_w.$$

`LID` cells move with one prescribed velocity (the rolling road, at the freestream speed).
`OBSTACLE` cells may carry their own: a spinning part sets
$\mathbf{u}_w = \boldsymbol{\Omega} \times (\mathbf{x}_c - \mathbf{p})$ at each captured cell
centre $\mathbf{x}_c$, about the axis point $\mathbf{p}$. The term implicitly assumes $\rho = 1$
at the wall and runs out of stability margin near a wall speed of 0.10 - 0.12, which is the
origin of the wall-speed cap (§9.3).

### 3.8 The closed-domain mass anchor

An open tunnel holds its density level unaided, because the outlet pins it. A closed domain (the
lid-driven cavity) has no such anchor, and any small per-link imbalance integrates without
bound. `enforce_mass` rescales every distribution by $k = \rho_{\mathrm{target}} /
\bar\rho_{\mathrm{fluid}}$. Because $\rho$ and $\rho\mathbf{u}$ scale together, the velocity
field is left exactly unchanged (V2 checks it bit for bit); only the arbitrary absolute density
level moves.

### 3.9 Deliberate limitations

- **Staircase walls.** The sandbox's bodies are voxel staircases with half-way bounce-back.
  Interpolated bounce-back is certified (V9) but buys only about 0.6 points of sphere drag, the
  wall-placement term of the error budget (blockage contributes about 3.8 points and streamwise
  confinement about 1.2).
- **Blockage and confinement.** The tunnel is a few body lengths long and wide; drag is inflated
  by blockage in the same way as in a small physical tunnel, and no correction is applied.

### 3.10 Validation gates (V2, V8 - V10)

| Gate | Property | Checked against |
|---|---|---|
| V2 | The open tunnel holds its density; the anchor pins it and leaves u exact | Invariants: drift < 1e-8 per step; u bit-identical |
| V8 | Moving walls give the Magnus force with the right sign and growth | Physical sign and monotonicity; 1.5 < abs(Cl) < 9 at spin ratio 2 |
| V9 | Interpolated bounce-back places the wall at fraction q | Shifted-wall Poiseuille, u_max within 1.5 % |
| V10 | The sponge absorbs sound without changing the answer | Reflection < 0.05; Ahmed Cd within 1 % |

## 4. Forces and coefficients

### 4.1 Momentum exchange

The force on the model is the momentum exchanged across every link between a fluid cell and an
`OBSTACLE` cell (Mei et al., 2002), measured relative to the ambient state:

$$\mathbf{F} = \sum_{\text{links}} \left(f^*_{\bar\imath}(\mathbf{x}) + f_i(\mathbf{x}, t + 1)
- 2 w_{\bar\imath}\right) \mathbf{e}_{\bar\imath}.$$

The reference $2 w_{\bar\imath}$ sums to zero over a closed body, so it changes nothing there;
for a face resting against a wall or a floor, it removes a spurious ambient-pressure buoyancy.
Only `OBSTACLE` links count: a fixed floor is `WALL` and is excluded, because once its pressure was
counted as a car's lift (Cl -124).

### 4.2 Torque

Each link's force acts at the link midpoint, so the torque about the reference point
$\mathbf{r}_0$ (the model's centre) is

$$\mathbf{T} = \sum_{\text{links}} \left(\tfrac12 (\mathbf{x} + \mathbf{x}_s) + \tfrac12 - \mathbf{r}_0\right)
\times \mathbf{F}_{\text{link}},$$

where $\mathbf{x}_s$ is the solid cell's index and the $\tfrac12$ moves an index to a cell centre.

### 4.3 Deterministic accumulation

Each workgroup of 256 cells sums its links in a fixed-order shared-memory tree; a single workgroup
then folds the partials into a step total and a window sum. Summing per *step* into the window
(one addition of order 0.4) rather than per link keeps long windows exact in f32: when per-link
additions vanish into a large sum, an Ahmed Cd of 0.716 can read 0.603.
V11 checks that a 10,000-step window equals the double-precision sum of the per-step forces.

### 4.4 Coefficients and reference areas

With $\rho_\infty = 1$, the dynamic pressure is $q = \tfrac12 U^2$ and

$$C_D = \frac{F_x}{q A}, \quad C_L = \frac{F_y}{q A}, \quad C_S = \frac{F_z}{q A}, \quad
C_{m,z} = \frac{T_z}{q A L}.$$

$A$ is the frontal area (the number of cells in the model's projection along $x$, the road-vehicle
convention), the planform area (the projection along $y$, the wing convention) or a manual value.
The dashboard smooths the per-batch mean forces with an exponential moving average whose time
constant is 120 solver steps, and evaluates $q$ with $\max(u_{\mathrm{applied}},
\tfrac14 u_{\mathrm{command}})$, so the ramp from rest does not divide by a vanishing speed.

### 4.5 Deliberate limitations

- **Sandbox Reynolds numbers.** At $\mathrm{Re} \sim 10^3 - 10^4$ (§1.2) boundary layers are thick
  and separation early, so absolute coefficients are qualitative. The Ahmed body reads
  $C_D \approx 1.4$ at $\mathrm{Re}_L = 1{,}250$ against 0.285 in experiment at
  $4.3 \times 10^6$; the trend with Reynolds number is right (V15), the absolute value is not.
  Comparisons between configurations at the same grid and speed are what the tunnel is for.
- **Pressure and friction together.** Momentum exchange measures the total force; it does not
  separate pressure drag from skin friction.

### 4.6 Validation gates (V6, V7, V11, V14, V15)

| Gate | Property | Checked against |
|---|---|---|
| V6 | Sphere drag at Re 100 and 300 | Schiller-Naumann correlation, within 15 % |
| V7 | Shedding frequency of a cylinder at Re 200 | St = 0.196, within 15 % |
| V11 | Long force windows are exact | Double-precision host sum, relative 1e-3 |
| V14 | Ahmed body: low-Re drag band, slant lift, recirculating wake | Published geometry; low-Re bands: 0.6 < Cd < 1.8, Cl > 0, reverse flow behind the base |
| V15 | Ahmed drag falls with Reynolds number | The published trend |

## 5. Geometry

### 5.1 Placement

A mesh is fitted to the tunnel by a uniform scale and a translation: with $s = L / \ell$, where
$\ell$ is the longest side of the mesh's bounding box and $L$ the requested length in cells,

$$\mathbf{v}' = (\mathbf{v} - \mathbf{m})\, s + \mathbf{c},$$

where $\mathbf{m}$ is the bounding-box centre and $\mathbf{c}$ the target centre. It is then rotated
about $\mathbf{c}$ by $R = R_y(\psi)\, R_z(\theta)\, R_x(\phi)$ (yaw, pitch, roll, applied roll
first). A positive angle of attack is nose-up for a model pointing into the wind, which is a
negative rotation about $+z$, so the pitch argument is $-\alpha$. In ground mode the mesh is
finally dropped so that its lowest point sits at the floor height plus the ride height.

The arithmetic is float32 throughout, with a centre computed in double precision and rounded to
float32 once. This matters: cells lying exactly on a boundary flip with a one-ulp shift of the
centre (§5.5).

### 5.2 The winding-number fill

For each axis in turn, one thread per grid column collects every triangle crossing of the column's
line, with the sign of the triangle's normal along the axis. A cell centre is inside if the
winding number, the signed count of crossings beyond it, is non-zero, and the cell receives one
vote. A cell with at least two of the three votes becomes `OBSTACLE`.

The winding rule, rather than parity, is what lets the procedural models be built as unions of
overlapping closed parts: parity calls the overlap of two parts *outside* and punched a hole at
every junction. The two-of-three vote tolerates the imperfections of real STL files (small gaps,
duplicated faces), where a single axis can miscount.

### 5.3 The thin-feature shell

A plate thinner than a cell has almost no cell centres inside it, so the fill alone leaves gaps
that no bounce-back link sees. Every cell whose unit cube a triangle intersects (the conservative
"supercover") records its nearest marking triangle. A shell cell is then made solid only where
the feature is genuinely thin: stepping 0.5, 1 and 1.5 cells from the nearest surface point along
the normal finds a shell cell whose nearest triangle faces the other way (normals more than
120 degrees apart), or no solid exists near it at all (a single-sided sheet). A thick body's skin
sees only its own face, or a perpendicular one at a convex edge, so it keeps its exact
centre-inside voxels. Four simpler rules were tried and rejected.

### 5.4 The exact signed distance

The compressible walls (§8.6) need the true distance to the surface, not an estimate from the
flags (an estimate from the flags alone puts the wedge's shock 13 % off). One thread per triangle
computes the exact point-to-triangle distance to every cell centre within a band of 3 cells of
its bounding box and keeps the minimum per cell (in fixed point, 1/4096 of a cell, via one atomic
minimum). The result $\phi$ is negative where the flags are solid and clamped to $\pm 4$; cells
with no triangle in the band hold $+4$.

### 5.5 Analytic shapes

The validation gates also place spheres and cylinders analytically, testing the cell *index*
against the shape: a sphere marks $(i, j, k)$ where
$(i - c_x)^2 + (j - c_y)^2 + (k - c_z)^2 \leq r^2$, evaluated in double precision. Centres must be
passed in double precision: `200 * 0.3f` is 60.0000038, not 60.0, and that shift moves the five
cells lying exactly at $r$ outside (2,104 cells instead of 2,109), which moves a sphere's drag by
1 %. For interpolated bounce-back the exact link fraction of
§3.6 is solved from $|\mathbf{x}_0 + t\,\mathbf{e} - \mathbf{c}| = r$ at the cell centres.

### 5.6 Deliberate limitations

- **One cell is the floor.** Features thinner than a cell are kept one cell thick; their true
  thickness is lost.
- **No surface map.** The per-cell nearest-triangle map is not kept after voxelisation, so forces
  cannot be painted back onto individual triangles; the renderer samples the flow at the
  surface instead (§10.2).

### 5.7 Validation gates (V12, V13)

| Gate | Property | Checked against |
|---|---|---|
| V12 | A rotated box is exact; a 0.3-cell plate is sealed; overlapping parts fill their union | The exact centre-inside set; a D3Q19 flood fill; the union plus a fillet |
| V13 | The triangle-mesh pipeline reproduces the analytic sphere | Solid count within 5 %, drag within 5 % of the analytic run |

## 6. The passive scalar (dye)

### 6.1 The D3Q7 lattice and its equilibrium

The dye concentration $C$ is carried on its own seven-velocity lattice (rest and the six face
neighbours), with $w_0 = 1/4$, $w_{1..6} = 1/8$ and $c_s^2 = 1/4$. It is advected by the flow's
velocity through a second-order equilibrium:

$$g_i^{\mathrm{eq}} = w_i C \left(1 + 4\, \mathbf{e}_i \cdot \mathbf{u} + 8 (\mathbf{e}_i \cdot \mathbf{u})^2
- 2\, \mathbf{u} \cdot \mathbf{u}\right).$$

The second-order form stays positive upstream at any tunnel speed; the linear form goes negative
past $|\mathbf{u}| = 0.25$, and clipping the negative values then manufactures dye ($C = 1.625$
upstream of a body).

### 6.2 Two-relaxation-time collision

The populations split into even and odd parts about each opposite pair,
$g^\pm_i = \tfrac12 (g_i \pm g_{\bar\imath})$, which relax at different rates:

$$g_i^* = g_i - \omega_+ \left(g_i^+ - g_i^{\mathrm{eq}+}\right) - \omega_- \left(g_i^- - g_i^{\mathrm{eq}-}\right),
\qquad \omega_- = \frac{1}{\tau}, \quad \omega_+ = \frac{1}{\tau_+}.$$

The odd rate sets the diffusivity, $D = c_s^2 (\tau - \tfrac12) = (\tau - \tfrac12)/4$. The app
uses $\tau = 0.53$ ($D = 0.0075$, a small and physical value rather than a scheme's numerical smear)
and $\tau_+ = 1$. Both were chosen by measurement: plain BGK at $\tau = 0.515$ flips population
signs, and the usual large $\tau_+$ is unstable at the tunnel's cell Peclet number of about 80.

### 6.3 Boundaries, sources and the limiter

The dye streams by the same pull, on the same flags, as the flow: the inlet brings clean air
($C = 0$), side walls and bodies bounce back (no flux, so dye can never leak into a solid), and the
outlet is an equilibrium outflow at the cell's own $C$ and $\mathbf{u}$ (a plain zero-gradient
outlet recycled dye and blew up at the outlet corners). Nozzle cells relax towards $C = 1$ at the
rate $r$ per step, distributing the added concentration by the weights:

$$\Delta C = r (1 - C), \qquad g_i \leftarrow g_i + w_i\, \Delta C.$$

Because every source holds $C = 1$, a value above one is only the scheme's small overshoot at a
steep front (up to about 1.07). Next to spinning walls, however, the moving-wall term makes the
flow locally compressible and the conservative scalar piled up without bound (C 760 beside a
wheel). The populations are therefore rescaled whenever $C$ exceeds $C_{\max} = 1.25$; this is an
exact identity for every validated case, since none reaches the limit.

### 6.4 Deliberate limitations

- **Not monotone.** Steep fronts beside a nozzle overshoot by a few per cent; populations are
  clipped at zero only to guard against round-off.
- **One-way coupling.** The dye does not affect the flow.

### 6.5 Validation gate (V20)

| Gate | Property | Checked against |
|---|---|---|
| V20 | Advection at U, spreading with $D = (\tau - \tfrac12)/4$, no leak through walls, bounded with sources | Analytic Gaussian blob; exact zero beyond a wall; $0 \leq C \leq 1.1$ |

## 7. Synthetic inlet turbulence

### 7.1 The solenoidal patch

Real tunnel air carries a little turbulence, and a perfectly clean inlet never disturbs a
separating shear layer. A periodic patch $\mathbf{u}'(s, y, z)$ of span $S$ planes is built once:
three independent white-noise fields form a vector potential $\mathbf{A}$, each filtered by a
periodic Gaussian of width $\sigma = L$ cells (the eddy size), and

$$\mathbf{u}' = \nabla \times \mathbf{A}$$

by periodic central differences, with the mean removed and the field scaled to unit RMS per
component. The discrete divergence of a discrete curl vanishes identically, so the patch is
divergence-free to round-off (4e-8 against a typical gradient of 9e-2, V21).

### 7.2 Convection through the inlet

After each step the inlet plane is overwritten with

$$\mathbf{u}_{\mathrm{inlet}}(y, z) = U \left(\hat{\mathbf{x}} + I\, \mathbf{u}'(s, y, z)\right),$$

where $I$ is the requested intensity and $s$ advances by $U$ planes per step (Taylor's frozen
turbulence), interpolating linearly between planes. The curl is the reason the scheme is quiet:
the inlet is a hard Dirichlet plane, and any compressive part of an injected fluctuation would
leave as sound. Solenoidal fluctuations make about one fifth of the density fluctuation of the same
intensity of scrambled, non-solenoidal noise (V21).

### 7.3 Deliberate limitations

- **A synthetic spectrum.** The patch has a Gaussian, not a Kolmogorov, spectrum; it seeds
  instability rather than reproducing a particular tunnel's turbulence.
- **Decay.** The intensity decays downstream under the LES and molecular viscosity (2 % at the
  inlet measures 1.75 % at 8 cells and 1.62 % at 60).

### 7.4 Validation gate (V21)

| Gate | Property | Checked against |
|---|---|---|
| V21 | The patch is solenoidal and unit RMS; the measured intensity matches; it is quiet; 1 % changes the Ahmed drag by less than 10 % | Closed form; the request within 25 %; non-solenoidal noise; a sanity bound |

## 8. The compressible Euler solver

### 8.1 Equations, state and units

The Euler equations of an ideal gas with $\gamma = 1.4$ are solved for the conserved state
$\mathbf{U} = (\rho, \rho u, \rho v, \rho w, E)$, with pressure

$$p = (\gamma - 1)\left(E - \tfrac12 \rho\, |\mathbf{u}|^2\right),$$

on the same grid and flags as the lattice Boltzmann solver. The units are chosen so that the
freestream has $\rho_\infty = 1$ and sound speed $c_\infty = 1$ (so $p_\infty = 1/\gamma$), and a
cell has unit size: the freestream speed *is* the Mach number, and time is measured in cell
sound-crossings.

### 8.2 MUSCL reconstruction

The primitive state $\mathbf{W} = (\rho, u, v, w, p)$ is reconstructed to second order at each
face from the four-cell stencil along the face normal:

$$\mathbf{W}_L = \mathbf{W}_{i} + \tfrac12\, \mathrm{lim}\left(\mathbf{W}_i - \mathbf{W}_{i-1},\,
\mathbf{W}_{i+1} - \mathbf{W}_i\right),$$

and its mirror image for $\mathbf{W}_R$, with the van Leer limiter (minmod is available):

$$\mathrm{lim}(a, b) = \begin{cases} \dfrac{2ab}{a + b}, & ab > 0,\\ 0, & ab \leq 0. \end{cases}$$

If either reconstructed state has a non-positive density or pressure, the face falls back to first
order. A solid cell in the stencil is replaced by its wall state (§8.6), so walls and the interior
share one code path.

### 8.3 The HLLC flux

The flux through each face is the HLLC approximate Riemann solver (Toro, 2009), which restores the
contact wave missing from HLL and so keeps contacts and shear layers sharp. The outer wave speeds
are the Davis estimates

$$S_L = \min(u_L - c_L,\; u_R - c_R), \qquad S_R = \max(u_L + c_L,\; u_R + c_R),$$

and the contact speed is

$$S_* = \frac{p_R - p_L + \rho_L u_L (S_L - u_L) - \rho_R u_R (S_R - u_R)}
{\rho_L (S_L - u_L) - \rho_R (S_R - u_R)},$$

with $u$ the velocity normal to the face. The flux is the physical flux of the left or right state
when $0 \leq S_L$ or $0 \geq S_R$, and otherwise that of the star state on the appropriate side of
the contact.

### 8.4 Time integration

The two-stage strong-stability-preserving Runge-Kutta scheme (SSP-RK2) advances the state:

$$\mathbf{U}^{(1)} = \mathbf{U}^n + \Delta t\, R(\mathbf{U}^n), \qquad
\mathbf{U}^{n+1} = \tfrac12 \left(\mathbf{U}^n + \mathbf{U}^{(1)} + \Delta t\, R(\mathbf{U}^{(1)})\right).$$

The time step is computed on the device every step, from the fluid cells,

$$\Delta t = \frac{\mathrm{CFL}}{\max\left(|u| + |v| + |w| + 3c\right)}, \qquad \mathrm{CFL} = 0.4,$$

optionally capped (the Sod gate uses the cap to land exactly on its end time).

### 8.5 Domain boundaries

- **Inflow** ($x < 0$): the freestream state.
- **Outflow** ($x \geq n_x$): zero-gradient, except that a subsonic outflow holds $p = p_\infty$.
- **Side faces** ($y$, $z$): far field (the freestream state) or slip (the normal velocity
  mirrored). A two-dimensional section needs slip $z$ faces: a far-field $z$ made it a two-cell
  wing with no lift.
- **Transmissive** $x$ (the shock tube): zero-gradient at both ends.

### 8.6 Image-point ghost cells

Each solid cell next to fluid holds a *ghost* state built from the flow. With $d = -\phi \geq 0$ the
ghost centre's depth below the true wall and $\mathbf{n} = \nabla\phi / |\nabla\phi|$, the image
point lies across the wall at

$$\mathbf{x}_I = \mathbf{x}_g + \Delta n\, \mathbf{n}, \qquad \Delta n = d + \max(d, 0.75),$$

at least 0.75 cells out so that it samples fluid. The image state is interpolated trilinearly over
fluid cells only (the weights renormalised, and at least 0.2 of the total weight required), and
the ghost holds it with the normal velocity reversed, corrected for curvature (§8.7).

A face uses the image ghost only if the ghost is valid, the fluid cell lies on the fluid side of
the ghost's normal, and the solid is more than one cell thick across the face. A one-cell sheet,
such as a trailing edge or a fin, has a distance normal pointing *along* the sheet; its image
reversed the streamwise velocity and let flow through the trailing edge (the NACA 0012 read Cl
0.26 instead of 0.35), so such faces use the impermeable grid-axis mirror instead. Two simpler wall
rules were rejected: a mirror about the grid axis everywhere builds a spurious boundary layer, and
reflecting the adjacent cell makes trailing edges porous.

### 8.7 Curvature-corrected symmetry

A plain mirror assumes $\partial p / \partial n = 0$ at the wall. On a curved wall the normal
momentum balance gives $\partial p / \partial n = \rho\, u_t^2\, \kappa_t$, so the curvature-corrected
symmetry technique of Dadone and Grossman (2004) sets the ghost pressure

$$p_g = \max\left(p_I - \rho_I\, |\mathbf{u}_t|^2\, \kappa_t\, \Delta n,\; 0.3\, p_I\right),$$

then the density from constant entropy, $\rho_g = \rho_I (p_g / p_I)^{1/\gamma}$, and the speed from
constant total enthalpy, with the normal velocity reversed. The tangential curvature
$\kappa_t = \mathbf{t} \cdot \mathrm{H}(\phi)\, \mathbf{t}$ (clamped to one per cell) is taken at the
*image* point's cell and carried to the wall as $\kappa / (1 - \phi\kappa)$: inside a thin body
$\phi$ has a ridge whose Hessian over-corrected the nose into thrust. Without the correction the
NACA 0012's lift converged like $\sqrt{h}$ and read 25 % low at 64 cells a chord.

### 8.8 Pressure forces

The force on the model is the sum, over every face between a fluid cell and an `OBSTACLE` cell, of
the wall pressure times the face's area vector; the staircase's projected areas sum to the body's
exactly. The wall pressure is the image ghost's value interpolated to the true wall,
$p_w = p_I + \tfrac{\Delta n - d}{\Delta n} (p_g - p_I)$, where the face uses the image (the
ghost's own pressure lies beyond the wall, and gave a NACA 0012 a thrust), and otherwise the star
pressure of the Riemann problem between the cell and its mirror image. As in §4.1, a floor is
`WALL` and is excluded.

### 8.9 Deliberate limitations

- **Inviscid.** There are no boundary layers and no skin friction: drag is pressure and wave drag
  only.
- **Voxel walls cost lift.** At 128 cells a chord the NACA 0012 reads Cl 0.291 against about 0.35
  for the published Euler solutions; the far field must also sit at least about six chords away,
  since a closer one inflates lift.
- **No pitching moment** is computed in the transonic regime yet.

### 8.10 Validation gates (V17 - V19)

| Gate | Property | Checked against |
|---|---|---|
| V17 | Shock tube: all three wave families | The exact Riemann solution: L1(rho) < 2 %, shock within 2 cells, transverse symmetry |
| V18 | Mach 2 over a 15 degree voxel wedge | theta-beta-M: shock angle within 1.5 degrees, wall pressure within 5 % |
| V19 | NACA 0012 at Mach 0.8, alpha 1.25 degrees | AGARD-AR-211 bands for Cl, Cd and the upper shock; converged lift |

## 9. The sandbox loop

### 9.1 Ramp and slew

A flow starts from rest. The applied inlet speed follows the command through a linear ramp over the
first 2,000 steps and is otherwise slew-limited to $10^{-4}$ per step, so a slider jump never
shocks the lattice:

$$u_{\mathrm{desired}} = u_{\mathrm{command}} \min\left(1, \frac{n}{n_{\mathrm{ramp}}}\right), \qquad
|\Delta u_{\mathrm{applied}}| \leq 10^{-4}\ \text{per step}.$$

An impulsive start can reach a local speed of 0.31, which is why a new body always
restarts from rest. In ground mode the rolling road moves at the applied speed. The Mach number of
the transonic regime is slewed similarly, by 0.004 per batch.

### 9.2 Operating points and restarts

An *operating point* is everything that determines the settled flow. A change that leaves the flow
recognisably the same (a small turn, a speed change, spin, turbulence) keeps the flow and restarts
only the settling detector; a speed change of more than 10 % does so automatically. A *new body*,
or the same one turned by more than 20 degrees on any axis or resized by more than 25 %, restarts
the flow from rest with the ramp.

### 9.3 Spinning parts and the wall-speed cap

A spinning part has a centre, an axis, a rim radius $r$ and a sense; its angular velocity follows
the spin ratio $\lambda$ (rim speed over wind speed), $\Omega = \pm \lambda U / r$. The cells it
captures lie within $1.2\,r$ of the axis and $1.4$ times its half-length along it. Every part's
$\Omega$ is then scaled by one common factor so that no captured cell moves faster than the cap
of 0.08 (0.025 for thin rotor blades, where 0.04 diverged): rigid rotation and the relative speeds
of the parts are preserved, and the panel reports when the cap is limiting.

### 9.4 The divergence guard

Every tenth batch the solver's health is read: the number of non-finite fluid cells (tested on the
bit pattern, because `x != x` is folded away by fast-math) and the maximum speed. A non-finite cell
or a speed above 0.5 resets the flow from rest with a note naming the likely cause; a second
blow-up within 60 s pauses the tunnel instead of resetting into it again. The transonic regime
restarts from the freestream if the peak local Mach exceeds 5 or any cell goes bad.

### 9.5 The settling detector

The coefficient history is cut into consecutive windows of 0.5 flow-throughs of *simulated* time,
where one flow-through is $n_x / U$ steps. With window means $w_1, w_2, w_3$ (the last three) and
differences $d_1 = w_2 - w_1$, $d_2 = w_3 - w_2$, a coefficient is settled when

$$|d_2| < \mathrm{tol} = \max\left(0.01\, |w_3|,\; 0.01\right),$$

and, on a monotone approach ($d_1 d_2 > 0$), the geometric tail estimate of the remaining change is
inside the tolerance too:

$$r = \frac{d_2}{d_1} < 0.9 \quad\text{and}\quad \frac{|d_2|\, r}{1 - r} < \mathrm{tol}.$$

The flow is declared settled when every coefficient (Cd and Cl) is settled and at least 1.5
flow-throughs have passed, which guards the early overshoot where two windows straddling a peak
look alike. Opposite signs indicate oscillation about a mean, where the plain difference test is
right. The detector gives up at 6 flow-throughs and says so. The tail test exists because two
consecutive windows can agree to within the tolerance on a slow approach while the value is still
several tolerances away (declared 2.9 % off on a real Ahmed history before the test was added).

### 9.6 The settled-flow cache

A settled flow is saved under a key built from a canonical description of its operating point: the
model's content hash, the placement, the spin and the spinner definitions, the speed, the
turbulence, the grid, every solver setting and a hash of the physics source code. The key is the
first 96 bits of a 128-bit FNV-1a hash of that description (two independent 64-bit streams), so
any change to the solver or voxeliser source invalidates every entry. The distributions are stored as binary16 of
$f_i - w_i$: the deviation from rest is about 100 times smaller than $f_i$, so half precision keeps
about 32 times more of it (a maximum error of 3.8e-6 on a developed Ahmed flow). A restore sets the
distributions and refreshes $\rho$ and $\mathbf{u}$ (§3.2); entries beyond a 512 MB budget are
evicted oldest first.

### 9.7 Deliberate limitations

- **Exact matches only.** A neighbouring cached state is not used as a starting point: it saves
  only about 18 % of the settling time, because settling is physics.
- **A heuristic detector.** The thresholds were tuned on real Ahmed histories; a strongly
  unsteady flow may never settle and is released at 6 flow-throughs.

### 9.8 Validation gates (V11, V22, V23)

| Gate | Property | Checked against |
|---|---|---|
| V11 | The guard detects a blow-up; spin at the cap is stable; long windows are exact | Poisoned fields; 8,000 healthy steps; a double-precision sum |
| V22 | The detector settles correctly on synthetic and real signals | Five synthetic invariants; the long-run mean within 2 % |
| V23 | A restored flow is the settled flow | A never-restored control: means within 1.5 % |

## 10. What the renderer derives

The renderer is a view, not a model, and is not gated; this section records what its colours
mean, so that a picture can be read quantitatively. It also specifies the one geometric tool
beside the renderer that is gated: marching cubes (§10.5).

### 10.1 Transfer functions

Each field is computed once per cell into a signed, normalised value; a ray then accumulates
opacity in proportion to the deviation above a noise floor, so undisturbed flow is transparent:

| Field | Value per cell |
|---|---|
| speed | $(|\mathbf{u}| - U) / U$ |
| pressure | $C_p = (p - p_{\mathrm{ref}}) / (\tfrac12 U^2)$, with $p = \rho / 3$ on the lattice |
| vorticity | $|\nabla \times \mathbf{u}| / U$ |
| streamwise vorticity | $10\, \omega_x / U$ |
| Mach | $(M - 1) / 0.5$, with $M = |\mathbf{u}| \sqrt{3}$ on the lattice |
| schlieren | $4\, |\nabla \rho| / U^2$ |

In the transonic regime the same fields use the Euler state: $U$ is the Mach number, $p$ is the
pressure itself, and $M = |\mathbf{u}| / \sqrt{\gamma p / \rho}$.

### 10.2 Pressure coefficient and its reference

The reference pressure is the mean density on an upstream plane, half-way between the inlet and the
model's front face (at least 3 cells in), refreshed every fourth batch: a Pitot-static reference
rather than the freestream value, so the tunnel's own pressure gradient does not tint the model.
While the inlet ramps the dynamic pressure is tiny, so the surface stays unpainted until the
applied speed reaches a third of the command.

### 10.3 Vortex cores

Vortex cores are drawn where the Q criterion is positive and large:

$$Q = \tfrac12 \left(|\boldsymbol{\Omega}|^2 - |\mathbf{S}|^2\right),$$

with $\boldsymbol{\Omega}$ and $\mathbf{S}$ the antisymmetric and symmetric parts of the velocity
gradient, from central differences of a velocity smoothed by a $3 \times 3 \times 3$ box over fluid
cells only. At sandbox Reynolds numbers one-cell velocity wiggles have large gradients that raw Q
flags as vortices everywhere. Q is zeroed in and next to solids, where the gradient is the voxel
staircase rather than vorticity. The threshold adapts to the flow: three times the RMS of the
positive Q, sampled across the domain, times the user's sensitivity. Q is refreshed every third
frame.

### 10.4 Deliberate limitations

- **Snapshots.** The renderer draws snapshots published by the solver at most every 12 ms, so the
  picture lags the solver by up to a batch.
- **Smoothed cores.** The box filter that makes Q usable also widens the cores; they show where
  vortices are, not how thin they are.

### 10.5 Iso-surfaces by marching cubes

Marching cubes turns a sampled scalar field back into triangles, the reverse of §5.2. Given $f$ at
the cell centres and a level $f_0$, each unit cube between eight neighbouring centres is
classified by which of its corners lie below $f_0$: an 8-bit case index, 256 cases. The classic
tables (Lorensen and Cline 1987, in Bourke's public-domain form) give, for each case, the edges
the surface crosses and up to five triangles joining them. On a crossed edge from corner $a$ to
corner $b$ the vertex is placed by linear interpolation,

$$\mathbf{x} = \mathbf{x}_a + t\,(\mathbf{x}_b - \mathbf{x}_a), \qquad t = \frac{f_0 - f_a}{f_b - f_a},$$

always from the edge's lower corner, so the two cubes that share an edge compute the same vertex
to the bit and the triangles weld into a closed surface. Each triangle is wound so that its
right-hand normal points towards decreasing $f$. On a smooth field a vertex lies within about
$h^2 \kappa / 8$ of the true surface, for curvature $\kappa$ and spacing $h = 1$: the sagitta of a
chord one cell long, about 0.006 cells on a sphere of radius 20.

The extraction runs on the CPU (`marching_cubes`). The renderer does not use it, since vortex
cores are ray-marched (§10.3); it is a headless building block, gated on its own.

### 10.6 Validation gate (V16)

| Gate | Property | Checked against |
|---|---|---|
| V16 | Marching cubes extracts the iso-surface of a sampled field | Two analytic spheres (R 20 about a cell corner, R 12.3 off-grid): closed and oriented, V - E + F = 2, vertices within 0.02 cells, every normal outward, area and volume within 1 % |

## 11. Performance identities

The performance work changed how the solver runs without changing what it computes. Each change is
either an exact identity, checked bit for bit, or an approximation that is opt-in and judged by the
gates.

### 11.1 Lazy macroscopic writes

$\rho$ and $\mathbf{u}$ are written only on the steps that read them: the last step of a
submission, every step when a per-step consumer such as the dye needs them, and always on the plane
$x = n_x - 2$, which the next step's outlet reads. This saves 16 of about 176 bytes per cell per
step and is an exact identity.

### 11.2 Sparse force reduction

The per-workgroup force tree is skipped in workgroups where no thread touched a model link (about
99 % of them); they write exact zeros, so this too is an identity.

### 11.3 Half-precision storage

Optionally, the distributions are stored as binary16 of $f_i - w_i$ (§9.6), with all arithmetic in
f32. This halves the dominant memory traffic (about 1.3 times faster in the sandbox), but it is an
approximation: on the Ahmed body the mean drag moved by 0.2 - 0.7 %, and the batch-to-batch scatter
grew by a factor of 2.2 in one measurement and about 5 in another. The same confidence in a mean
therefore needs at least five times as many samples, which the speed-up does not repay, and the
option stays opt-in for that reason.

### 11.4 The fused Euler stage

The six face fluxes and the Runge-Kutta stage are computed in one kernel, which reads about 60 bytes
per cell per stage instead of about 180. The arithmetic is reordered at the level of one or two
units in the last place; the shock tube, wedge and NACA results are unchanged to the printed digit.

### 11.5 How each is checked

`tools/lbm_equiv` runs every lattice Boltzmann switch on and off in the same binary and requires
every bit to agree (CTest `P4_equiv`), and compares the reference path against results saved from
the kernel as it was before the performance work (`--check`). `tools/euler_run equiv` does the
same for the Euler solver. A code motion that "cannot change the arithmetic" did once (streaming
the collision outputs changed bits through fused multiply-add contraction), which is why both
checks exist.

## References

- Bourke, P. (1994). Polygonising a scalar field. paulbourke.net/geometry/polygonise.
- Bouzidi, M., Firdaouss, M. and Lallemand, P. (2001). Momentum transfer of a Boltzmann-lattice
  fluid with boundaries. *Physics of Fluids* 13, 3452.
- Dadone, A. and Grossman, B. (2004). Ghost-cell method for inviscid two-dimensional flows on
  Cartesian grids. *AIAA Journal* 42(12), 2499.
- Ghia, U., Ghia, K. N. and Shin, C. T. (1982). High-Re solutions for incompressible flow using
  the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics* 48, 387.
- Guo, Z., Zheng, C. and Shi, B. (2002). Discrete lattice effects on the forcing term in the
  lattice Boltzmann method. *Physical Review E* 65, 046308.
- Hou, S., Sterling, J., Chen, S. and Doolen, G. D. (1996). A lattice Boltzmann subgrid model for
  high Reynolds number flows. *Fields Institute Communications* 6, 151.
- Jacob, J., Malaspinas, O. and Sagaut, P. (2018). A new hybrid recursive regularised
  Bhatnagar-Gross-Krook collision model for lattice Boltzmann method-based large eddy simulation.
  *Journal of Turbulence* 19, 1051.
- Ladd, A. J. C. (1994). Numerical simulations of particulate suspensions via a discretized
  Boltzmann equation. *Journal of Fluid Mechanics* 271, 285.
- Latt, J. and Chopard, B. (2006). Lattice Boltzmann method with regularized pre-collision
  distribution functions. *Mathematics and Computers in Simulation* 72, 165.
- Lorensen, W. E. and Cline, H. E. (1987). Marching cubes: a high resolution 3D surface
  construction algorithm. *Computer Graphics* 21(4), 163.
- Mei, R., Yu, D., Shyy, W. and Luo, L.-S. (2002). Force evaluation in the lattice Boltzmann
  method involving curved geometry. *Physical Review E* 65, 041203.
- Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd edition.
  Springer.
- AGARD Advisory Report 211 (1985). Test cases for inviscid flow field methods.
