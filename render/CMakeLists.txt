# The ray-march renderer and the Lagrangian tracers: compute kernels drawing
# field snapshots into an image. Depends on the engine only (no window, no
# GUI) -- the app blits the result to its swapchain.
add_library(windoa_render STATIC
    src/volume.cpp
    src/tracers.cpp
)
windoa_add_shaders(windoa_render
    shaders/vol_prepare.comp
    shaders/vol_occ.comp
    shaders/vol_blur.comp
    shaders/vol_smooth.comp
    shaders/vol_q.comp
    shaders/vol_qstat.comp
    shaders/vol_march.comp
    shaders/vol_splat.comp
    shaders/smoke.comp
    shaders/streamlines.comp
)
target_include_directories(windoa_render PUBLIC include)
target_link_libraries(windoa_render PUBLIC windoa_engine)
