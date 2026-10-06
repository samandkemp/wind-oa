# The headless engine: Vulkan compute context, solver buffers and pipelines.
# No window, no GUI: the tools and the validation gates drive it directly.
add_library(windoa_engine STATIC
    src/context.cpp
    src/device.cpp
    src/lbm.cpp
    src/mesh.cpp
    src/voxeliser.cpp
    src/convergence.cpp
    src/dye.cpp
    src/catalogue.cpp
    src/flow_cache.cpp
    src/tunnel.cpp
    src/euler.cpp
    src/shapes.cpp
    src/isosurface.cpp
    src/flow_stats.cpp
    src/spectrum.cpp
    src/rotor.cpp
)
windoa_add_shaders(windoa_engine
    shaders/lbm_init.comp
    shaders/lbm_linkq.comp
    shaders/lbm_step.comp
    shaders/lbm_reduce.comp
    shaders/lbm_stats.comp
    shaders/lbm_scale.comp
    shaders/lbm_refresh.comp
    shaders/vox_winding.comp
    shaders/vox_votes.comp
    shaders/vox_shell_mark.comp
    shaders/vox_shell_apply.comp
    shaders/vox_dist_band.comp
    shaders/vox_phi.comp
    shaders/stats_accumulate.comp
    shaders/dye_step.comp
    shaders/lbm_turb.comp
    shaders/turb_blur.comp
    shaders/euler_ghost.comp
    shaders/euler_update.comp
    shaders/euler_stage.comp
    shaders/euler_dt.comp
    shaders/euler_force.comp
    shaders/euler_diag.comp
    shaders/alm_elements.comp
    shaders/alm_spread.comp
)
target_include_directories(windoa_engine PUBLIC include)

# Physics-source hash for the flow cache (cmake/physics_hash.cmake): the
# solver + voxeliser sources and shaders; regenerated whenever one changes.
set(WINDOA_PHYSICS_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lbm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/voxeliser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tunnel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shapes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rotor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/alm_elements.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/alm_spread.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/windoa/lbm.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/windoa/voxeliser.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lattice.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lbm_params.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lbm_step.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lbm_reduce.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lbm_init.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lbm_linkq.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lbm_refresh.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lbm_turb.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/turb_blur.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vox_common.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vox_winding.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vox_votes.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vox_shell_mark.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vox_shell_apply.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vox_dist_band.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vox_phi.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/voxeliser.cpp)
set(physics_hash ${CMAKE_CURRENT_BINARY_DIR}/generated/physics_hash.hpp)
add_custom_command(
    OUTPUT ${physics_hash}
    COMMAND ${CMAKE_COMMAND} -DOUT=${physics_hash} "-DFILES=${WINDOA_PHYSICS_FILES}"
            -P ${PROJECT_SOURCE_DIR}/cmake/physics_hash.cmake
    DEPENDS ${WINDOA_PHYSICS_FILES} ${PROJECT_SOURCE_DIR}/cmake/physics_hash.cmake
    COMMENT "physics source hash"
    VERBATIM)
target_sources(windoa_engine PRIVATE ${physics_hash})
target_include_directories(windoa_engine PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(windoa_engine PUBLIC Vulkan::Vulkan)
target_compile_definitions(windoa_engine PUBLIC
    $<$<BOOL:${WINDOA_VALIDATION_LAYERS}>:WINDOA_VALIDATION_LAYERS=1>)
