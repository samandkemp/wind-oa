cmake_minimum_required(VERSION 3.25)
project(wind_oa VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WINDOA_VALIDATION_LAYERS "Enable VK_LAYER_KHRONOS_validation at runtime" OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Vulkan REQUIRED COMPONENTS glslc)
include(cmake/Shaders.cmake)

enable_testing()

# Dependency arrows point one way: app -> render -> engine, and tools /
# validation -> engine. engine/ never links a window or GUI library.
add_subdirectory(engine)
add_subdirectory(tools)
add_subdirectory(render)
if(WIN32)
    add_subdirectory(third_party)
    add_subdirectory(app)
endif()
add_subdirectory(validation)
