# cmake -DSPV=in.spv -DHDR=out.hpp -DNAME=ident -P embed_spirv.cmake
# SPIR-V is a stream of little-endian 32-bit words; emit them as uint32s.
file(READ "${SPV}" hex HEX)
string(LENGTH "${hex}" nchars)
math(EXPR rem "${nchars} % 8")
if(NOT rem EQUAL 0)
    message(FATAL_ERROR "${SPV}: size is not a whole number of 32-bit words")
endif()
string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1u," words "${hex}")
string(REGEX REPLACE "((0x........u,){8})" "\\1\n    " words "${words}")
file(WRITE "${HDR}"
"// Generated from ${NAME}.comp by cmake/embed_spirv.cmake -- do not edit.
#pragma once
#include <cstdint>
namespace windoa::spv {
inline constexpr std::uint32_t ${NAME}[] = {
    ${words}
};
}  // namespace windoa::spv
")
