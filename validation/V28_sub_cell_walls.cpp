// V28 -- sub-cell walls from the signed distance (the sandbox's walls).
// Specified in THEORY 3.6 (docs/THEORY.md).
//
//   A. Closed form: a finely tessellated sphere (D 16, 96 x 192 facets) is
//      voxelised by the GPU Voxeliser and its link fractions taken as the
//      sandbox takes them (shapes::link_fractions: mesh crossings, the signed
//      distance where none). On the same flags the true sphere's fractions
//      come from ray intersection (shapes::sphere_link_fractions). Every
//      link filled, RMS within 0.01 of a link, worst within 0.05 (u8 steps
//      are 0.004). Reported beside it: the signed distance alone (linear
//      along a link, it errs where links graze the curved surface) and the
//      catalogue's coarser sphere (32 x 64 facets, up to 0.01 cells inside
//      the sphere, which grazing links magnify).
//   B. Closed form, carried into the flow: V9's Re 100 sphere drag on the
//      catalogue sphere with the sandbox's fractions against the true
//      sphere's, on the same flags, within 0.5 %; the half-way wall is
//      reported beside them.
//   C. Invariant: the app's tunnel (Tunnel, fast grid, sub_cell_walls on) at
//      its top speed stays healthy for 8,000 steps on the thin-featured
//      models -- wings, blades, rotors spinning -- with no guard restart.
#include <filesystem>

#include "cases.hpp"
#include "windoa/catalogue.hpp"
#include "windoa/mesh.hpp"
#include "windoa/tunnel.hpp"
#include "windoa/voxeliser.hpp"

using namespace windoa;

namespace {

constexpr int NX = 160, NY = 112, NZ = 112;
constexpr double D = 16.0, CX = 48.0, CY = 56.0, CZ = 56.0;

double sphere_cd(Context& ctx, const std::vector<std::uint8_t>& flags,
                 const std::vector<std::uint8_t>* q) {
    constexpr float U = 0.05f, RE = 100.0f;
    constexpr int RAMP = 1500, TOTAL = 14'000, AVG = 4'000;
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = shapes::tau_for_reynolds(U, float(D), RE);
    c.u_inlet = 0.0f;
    c.smagorinsky_cs = 0.0f;
    c.use_ibb = q != nullptr;
    lbm::Solver s(ctx, c);
    s.set_flags(flags);
    if (q)
        s.set_link_q(*q);
    s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    for (int k = 0; k < RAMP; ++k) {
        s.set_inlet_velocity(U * float(k + 1) / RAMP);
        s.step(1);
    }
    s.step(TOTAL - RAMP - AVG);
    s.read_mean_forces();
    s.step(AVG);
    const auto f = s.read_mean_forces();
    return f.force[0] / (0.5 * U * U * gate::kPi * (D / 2.0) * (D / 2.0));
}

} // namespace

int main() {
    return gate::run("V28_sub_cell_walls", [](gate::Gate& g) {
        Context ctx;
        Voxeliser vox(ctx, NX, NY, NZ);
        const std::size_t cells = std::size_t(NX) * NY * NZ;
        struct Fractions {
            std::vector<std::uint8_t> flags, mesh, phi, exact;
            int n_mesh = 0, n_crossed = 0, n_exact = 0;
        };
        auto fractions = [&](int n_lat, int n_lon) {
            Fractions f;
            f.flags.assign(cells, lbm::FLUID);
            const geometry::Mesh mesh = geometry::fit_to_box(
                catalogue::make_sphere(n_lat, n_lon), {float(CX), float(CY), float(CZ)}, float(D));
            vox.voxelise(mesh, f.flags);
            const std::vector<float> phi = vox.signed_distance(f.flags);
            f.mesh =
                shapes::link_fractions(f.flags, phi, NX, NY, NZ, &mesh, &f.n_mesh, &f.n_crossed);
            f.phi = shapes::link_fractions(f.flags, phi, NX, NY, NZ);
            f.exact =
                shapes::sphere_link_fractions(f.flags, NX, NY, NZ, CX, CY, CZ, D / 2.0, &f.n_exact);
            return f;
        };
        auto compare = [&](const std::vector<std::uint8_t>& q, const std::vector<std::uint8_t>& ref,
                           double& rms, double& worst) {
            double sum2 = 0.0;
            int n = 0;
            worst = 0.0;
            for (std::size_t i = cells; i < 19 * cells; ++i) { // skip the rest direction
                if (q[i] == 128 && ref[i] == 128)
                    continue;
                const double d = (double(q[i]) - double(ref[i])) / 255.0;
                sum2 += d * d;
                worst = std::max(worst, std::abs(d));
                ++n;
            }
            rms = std::sqrt(sum2 / std::max(n, 1));
        };

        g.section("A: link fractions against the true sphere");
        const Fractions fine = fractions(96, 192);
        double rms, worst, rms_phi, worst_phi;
        compare(fine.mesh, fine.exact, rms, worst);
        compare(fine.phi, fine.exact, rms_phi, worst_phi);
        g.check(fine.n_mesh == fine.n_exact && rms < 0.01 && worst < 0.05,
                "96 x 192 facets: %d links (ray: %d), %d by mesh crossings; RMS difference "
                "%.4f, worst %.3f (< 0.01, < 0.05)",
                fine.n_mesh, fine.n_exact, fine.n_crossed, rms, worst);
        g.note("signed distance alone: RMS %.4f, worst %.3f", rms_phi, worst_phi);
        const Fractions cat = fractions(32, 64);
        double rms_cat, worst_cat;
        compare(cat.mesh, cat.exact, rms_cat, worst_cat);
        g.note("catalogue sphere (32 x 64 facets): RMS %.4f, worst %.3f -- its facets", rms_cat,
               worst_cat);
        const std::vector<std::uint8_t>& flags = cat.flags;
        const std::vector<std::uint8_t>& q_mesh = cat.mesh;
        const std::vector<std::uint8_t>& q_exact = cat.exact;

        g.section("B: sphere drag, Re 100, D 16");
        const double cd_exact = sphere_cd(ctx, flags, &q_exact);
        const double cd_sdf = sphere_cd(ctx, flags, &q_mesh);
        const double cd_half = sphere_cd(ctx, flags, nullptr);
        const double e = std::abs(cd_sdf - cd_exact) / cd_exact;
        g.note("half-way wall: Cd %.4f (%+.2f %% from the true-sphere walls)", cd_half,
               100.0 * (cd_half - cd_exact) / cd_exact);
        g.check(e < 0.005, "sandbox walls Cd %.4f vs true-sphere walls %.4f: %.2f %% (< 0.5 %%)",
                cd_sdf, cd_exact, 100.0 * e);

        g.section("C: the app's tunnel at top speed, thin features");
        const std::filesystem::path cache =
            std::filesystem::temp_directory_path() / "windoa_v28_nocache";
        std::filesystem::remove_all(cache);
        TunnelSettings ts = tunnel_preset("fast");
        Tunnel tunnel(ctx, ts, cache);
        for (const char* id :
             {"car_open_wheel", "airliner", "propeller", "uav_quad", "wind_turbine", "frisbee"}) {
            const catalogue::Entry* en = catalogue::find(id);
            Model m = model_from_catalogue(*en);
            const Placement p = default_placement(*en, ts, m.mesh);
            const bool spin = !m.spinners.empty();
            tunnel.set_model(std::move(m), p);
            tunnel.set_speed(ts.u_max);
            tunnel.set_spin(spin, spin ? 2.0f : 0.0f);
            float peak = 0.0f;
            bool finite = true;
            for (int k = 0; k < 40; ++k) {
                tunnel.advance(200);
                const TunnelStatus st = tunnel.status();
                peak = std::max(peak, st.max_speed);
                finite = finite && std::isfinite(st.cd) && std::isfinite(st.cl);
            }
            const TunnelStatus st = tunnel.status();
            g.check(finite && st.health_note.empty() && peak < 0.3f,
                    "%-15s U %.2f%s, 8,000 steps: max |u| %.3f (< 0.3), Cd %+.3f, no restart%s", id,
                    ts.u_max, spin ? ", spinning" : "", peak, st.cd,
                    st.health_note.empty() ? "" : (" -- " + st.health_note).c_str());
        }
        std::filesystem::remove_all(cache);
    });
}
