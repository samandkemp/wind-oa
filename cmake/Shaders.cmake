# GLSL shaders (compute; the renderer's one vertex / fragment pair) ->
# SPIR-V at build time -> embedded C++ arrays; glslc takes the stage from
# the extension (.comp, .vert, .frag), so each name must be unique.
# No runtime shader files and no JIT: a first-use compile would stall the
# first solver steps and frames. Usage:
#   windoa_add_shaders(<target> shaders/foo.comp ...)
# then  #include "foo_spv.hpp"  ->  windoa::spv::foo  (const uint32_t[]).
# Shaders may #include any *.glsl next to them; every .comp rebuilds when
# one of those changes.
set(WINDOA_EMBED_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/embed_spirv.cmake)

function(windoa_add_shaders target)
    set(outdir ${CMAKE_CURRENT_BINARY_DIR}/shaders)
    set(headers "")
    foreach(src IN LISTS ARGN)
        get_filename_component(name ${src} NAME_WE)
        get_filename_component(srcdir ${CMAKE_CURRENT_SOURCE_DIR}/${src} DIRECTORY)
        file(GLOB includes CONFIGURE_DEPENDS ${srcdir}/*.glsl)
        set(spv ${outdir}/${name}.spv)
        set(hdr ${outdir}/${name}_spv.hpp)
        add_custom_command(
            OUTPUT ${hdr}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${outdir}
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O -Werror
                    -I ${srcdir} -o ${spv} ${CMAKE_CURRENT_SOURCE_DIR}/${src}
            COMMAND ${CMAKE_COMMAND} -DSPV=${spv} -DHDR=${hdr} -DNAME=${name}
                    -P ${WINDOA_EMBED_SCRIPT}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src} ${includes} ${WINDOA_EMBED_SCRIPT}
            COMMENT "glslc ${src}"
            VERBATIM)
        list(APPEND headers ${hdr})
    endforeach()
    target_sources(${target} PRIVATE ${headers})
    target_include_directories(${target} PRIVATE ${outdir})
endfunction()
