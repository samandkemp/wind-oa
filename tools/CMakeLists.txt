add_executable(device_info device_info.cpp)
target_link_libraries(device_info PRIVATE windoa_engine)

# Smoke check (P0): the toolchain builds and the engine sees a compute-capable GPU.
add_test(NAME P0_device_info COMMAND device_info --require-compute)

# Smoke check (P0): the compute path end to end (build-time SPIR-V -> pipeline ->
# dispatch -> readback), checked bit for bit against the CPU.
add_executable(compute_selftest compute_selftest.cpp)
windoa_add_shaders(compute_selftest shaders/saxpy.comp)
target_link_libraries(compute_selftest PRIVATE windoa_engine)
add_test(NAME P0_compute COMMAND compute_selftest)

# Development tool (not a gate): a sphere in the tunnel, MLUPS + Cd + health.
add_executable(lbm_run lbm_run.cpp)
target_link_libraries(lbm_run PRIVATE windoa_engine)

# Development tool (not a gate): the headless Tunnel (the app's sandbox),
# and a voxelisation check over the whole procedural catalogue (--catalogue).
add_executable(tunnel_run tunnel_run.cpp)
target_link_libraries(tunnel_run PRIVATE windoa_engine)

# Smoke checks, not physics validation (the gates are in validation/): every
# catalogue model voxelises inside the tunnel at its default placement, and
# the headless Tunnel runs a short sandbox session.
add_test(NAME P2_catalogue COMMAND tunnel_run --catalogue)
add_test(NAME P3_tunnel_smoke COMMAND tunnel_run --model sphere --batches 5 --steps 40 --dye)

# Development tool (not a gate): the Euler solver -- a sphere at a chosen
# Mach, kernel profile, and the P4 bit-identity baselines. (Sod, the wedge
# and the NACA 0012 are the gates V17-V19.)
add_executable(euler_run euler_run.cpp)
target_link_libraries(euler_run PRIVATE windoa_engine)

# Throughput benchmark (a measurement, not a gate): LBM (plain / moving
# walls / dye) and Euler per preset.
add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE windoa_engine)

# P4 safety net: every LBM performance switch that claims to be an identity
# is checked bit for bit against the reference path in the same binary.
add_executable(lbm_equiv lbm_equiv.cpp)
target_link_libraries(lbm_equiv PRIVATE windoa_engine)
add_test(NAME P4_equiv COMMAND lbm_equiv)
