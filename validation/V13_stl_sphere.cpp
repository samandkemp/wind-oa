// V13 -- the triangle-mesh pipeline end to end.
// Specified in THEORY 5.7 (docs/THEORY.md).
//
// Closed-form reference: the analytic voxel sphere. The faceted catalogue
// sphere (catalogue::make_sphere, 3,968 triangles) is fitted to D 16 at
// V6's Re 100 position, voxelised by the GPU Voxeliser, and re-run through
// V6's drag case. It must reproduce the analytic ball: solid count within
// 5 % of 2,109 (a shell of edge cells legitimately differs at D 16) and Cd
// within 5 % of the analytic-sphere run (1.213, V6).
#include "cases.hpp"
#include "windoa/catalogue.hpp"
#include "windoa/mesh.hpp"
#include "windoa/voxeliser.hpp"

using namespace windoa;

int main() {
    return gate::run("V13_stl_sphere", [](gate::Gate& g) {
        Context ctx;
        constexpr int NX = 160, NY = 80, NZ = 80;
        constexpr float D = 16.0f, U_IN = 0.05f, RE = 100.0f;
        constexpr double CD_ANALYTIC = 1.213;
        constexpr int N_SOLID_ANALYTIC = 2109;
        const auto mesh =
            geometry::fit_to_box(catalogue::make_sphere(), {NX * 0.3f, NY / 2.0f, NZ / 2.0f}, D);
        Voxeliser vox(ctx, NX, NY, NZ);
        std::vector<std::uint8_t> flags(std::size_t(NX) * NY * NZ, lbm::FLUID);
        const auto st = vox.voxelise(mesh, flags);
        const double count_err = std::abs(double(st.n_solid) - N_SOLID_ANALYTIC) / N_SOLID_ANALYTIC;
        g.check(count_err < 0.05,
                "%zu triangles -> %zu solid cells (analytic %d): %.1f %% (< 5 %%)", st.n_tris,
                st.n_solid, N_SOLID_ANALYTIC, count_err * 100.0);

        lbm::Config c;
        c.nx = NX;
        c.ny = NY;
        c.nz = NZ;
        c.tau = shapes::tau_for_reynolds(U_IN, D, RE);
        c.u_inlet = 0.0f;
        c.smagorinsky_cs = 0.0f;
        lbm::Solver s(ctx, c);
        s.set_flags(flags);
        s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
        constexpr int RAMP = 2000, N_STEPS = 40'000, N_AVG = 5'000;
        for (int k = 0; k < RAMP; ++k) {
            s.set_inlet_velocity(U_IN * float(k + 1) / RAMP);
            s.step(1);
        }
        s.step(N_STEPS - RAMP - N_AVG);
        s.read_mean_forces();
        s.step(N_AVG);
        const double cd =
            s.read_mean_forces().force[0] / (0.5 * U_IN * U_IN * gate::kPi * (D / 2.0) * (D / 2.0));
        const double cd_err = std::abs(cd - CD_ANALYTIC) / CD_ANALYTIC;
        g.check(cd_err < 0.05, "Cd %.3f vs the analytic-sphere run %.3f: %.1f %% (< 5 %%)", cd,
                CD_ANALYTIC, cd_err * 100.0);
    });
}
