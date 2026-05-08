# Test build, gated on -DPCR_BUILD_TESTS=ON.
#
# The tests link against the same source files the binaries use, rather
# than against a separate library target. This keeps the existing
# CMakeLists.txt structure untouched: PCR_RENDERER_SOURCES is defined in
# code/CMakeLists.txt and listed once per binary; we list the same files
# here. If the source list grows, both places update together.
#
# Header-only modules (Spectrum, Material, Optics, Vec3f, Ray, Triangle,
# Sphere, Plane) need only the include path. The .cpp modules
# (CIE, RGBToSpectrum, Bvh) get compiled into each test binary. That's
# wasteful in the CI build sense (every test recompiles them) but trivial
# in wall-clock terms (a handful of TUs, sub-second compile each), and
# avoids the cross-target object-library plumbing.

include(CTest)

set(PCR_TEST_SOURCES_FOR_LINK
    ${CMAKE_SOURCE_DIR}/CIE.cpp
    ${CMAKE_SOURCE_DIR}/RGBToSpectrum.cpp
    ${CMAKE_SOURCE_DIR}/Bvh/Bvh.cpp
    ${CMAKE_SOURCE_DIR}/Scenes/MeshLoader.cpp
)

set(PCR_TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/Includes
)

foreach(t IN ITEMS colorspace rgbtospec spectrum bvh optics)
    add_executable(test_${t} ${CMAKE_CURRENT_SOURCE_DIR}/unit/${t}.cpp ${PCR_TEST_SOURCES_FOR_LINK})
    target_include_directories(test_${t} PRIVATE ${PCR_TEST_INCLUDE_DIRS})
    target_link_libraries(test_${t} PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(test_${t} PRIVATE /W4 /bigobj)
    else()
        target_compile_options(test_${t} PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME unit_${t} COMMAND test_${t})
endforeach()
