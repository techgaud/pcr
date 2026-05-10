cmake_minimum_required(VERSION 3.20)
project(pcr CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Apple-only: enable Objective-C++. The Metal backend
# (Gpu/Metal/MetalRenderer.mm) is .mm so it can hold id<MTL...>
# Foundation/Metal objects in its pImpl. ARC is set per-source-file
# below on the .mm only, since the rest of the codebase is plain C++.
if(APPLE)
    enable_language(OBJCXX)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Disable MSVC link-time code generation. The spectral header
# infrastructure (Spectrum, CIE, RGBToSpectrum) is all header-only
# with heavy inline operators on 61-element arrays plus a Newton-
# Raphson solver, and MSVC's /GL whole-program optimizer crashes
# in the "Generating Code..." phase trying to chew through the
# combined inline footprint across every translation unit. CL.exe
# exits with code -529706956 (0xE06D7363, an unhandled C++
# exception inside the compiler itself). Disabling IPO sidesteps
# the compiler bug at marginal release-build performance cost.
#
# Linux/macOS toolchains handle the same code fine; this is MSVC-
# specific. Long-term the right fix is to move the heavy operators
# (RGBToSpectrum::fitCoefficients, CIE::spectrumToXYZ) out of
# headers into .cpp files, which would also drop the LTCG load on
# MSVC even with /GL re-enabled.
if(MSVC)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

    # MSBuild's default Release flags include /Zi (separate PDB
    # debug info). MSVC's PDB type-server (CL!CloseTypeServerPDB
    # in the crash stack we kept seeing) chokes on this codebase.
    # Strip /Zi out of Release so no type-server PDB gets opened
    # at all. We don't ship Release-config debug info anyway, so
    # losing /Zi here costs nothing visible.
    string(REGEX REPLACE "/Zi" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REGEX REPLACE "/ZI" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

    # /bigobj raises the per-object-file section limit. Modern
    # template-heavy C++ routinely exceeds the default; the
    # spectral code with std::array<float, 61> arithmetic and
    # Material's two-spectrum payload is a textbook trigger.
    add_compile_options(/bigobj)
endif()

find_package(Threads REQUIRED)

# --- Optional Intel Open Image Denoise (OIDN) ---------------------------
# Fetches an OS-appropriate prebuilt OIDN binary release from GitHub on
# first configure (~30 sec download). Define PCR_HAVE_OIDN for code that
# uses oidn::FilterRef etc; falls back gracefully when OFF (renderer
# builds without OIDN, --oidn flag prints a clear "not available" error).
option(PCR_USE_OIDN "Build with Intel Open Image Denoise support" ON)
if(PCR_USE_OIDN)
    include(FetchContent)
    set(_oidn_ver "2.3.0")
    if(WIN32)
        set(_oidn_url "https://github.com/RenderKit/oidn/releases/download/v${_oidn_ver}/oidn-${_oidn_ver}.x64.windows.zip")
    elseif(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(_oidn_url "https://github.com/RenderKit/oidn/releases/download/v${_oidn_ver}/oidn-${_oidn_ver}.arm64.macos.tar.gz")
        else()
            set(_oidn_url "https://github.com/RenderKit/oidn/releases/download/v${_oidn_ver}/oidn-${_oidn_ver}.x86_64.macos.tar.gz")
        endif()
    else()
        set(_oidn_url "https://github.com/RenderKit/oidn/releases/download/v${_oidn_ver}/oidn-${_oidn_ver}.x86_64.linux.tar.gz")
    endif()
    FetchContent_Declare(oidn URL ${_oidn_url})
    FetchContent_MakeAvailable(oidn)

    find_path(OIDN_INCLUDE_DIR OpenImageDenoise/oidn.h
              HINTS ${oidn_SOURCE_DIR}/include NO_DEFAULT_PATH)
    if(WIN32)
        find_library(OIDN_LIBRARY NAMES OpenImageDenoise
                     HINTS ${oidn_SOURCE_DIR}/lib NO_DEFAULT_PATH)
        file(GLOB OIDN_RUNTIME_DLLS "${oidn_SOURCE_DIR}/bin/*.dll")
    else()
        find_library(OIDN_LIBRARY NAMES OpenImageDenoise libOpenImageDenoise
                     HINTS ${oidn_SOURCE_DIR}/lib NO_DEFAULT_PATH)
        file(GLOB OIDN_RUNTIME_DLLS "${oidn_SOURCE_DIR}/lib/*.so*"
                                    "${oidn_SOURCE_DIR}/lib/*.dylib")
    endif()

    if(NOT OIDN_INCLUDE_DIR OR NOT OIDN_LIBRARY)
        message(FATAL_ERROR
                "OIDN download succeeded but headers/library not found in "
                "${oidn_SOURCE_DIR}. Re-run with -DPCR_USE_OIDN=OFF to build "
                "without denoiser support.")
    endif()
    add_library(oidn::oidn UNKNOWN IMPORTED)
    set_target_properties(oidn::oidn PROPERTIES
        IMPORTED_LOCATION ${OIDN_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${OIDN_INCLUDE_DIR}
    )
    message(STATUS "OIDN: include=${OIDN_INCLUDE_DIR} lib=${OIDN_LIBRARY}")
endif()

# Helper: link OIDN to a target + deploy its runtime libraries into a
# sibling `lib/` subdirectory of the binary, on every OS, so the binary
# folder stays tidy and only the .exe is at the root.
#
# On Linux/macOS this also matches OIDN's hardcoded RPATH of
# `$ORIGIN/../lib`, which is how OIDN resolves its dlopen-ed device
# plugins (_device_cpu / _device_cuda / _device_sycl) at runtime. The
# binary's own RPATH is set to `$ORIGIN/lib` so the dynamic loader finds
# `libOpenImageDenoise.so` from there.
#
# On Windows there's no RPATH; DllSearch.h calls SetDllDirectory("lib")
# at process startup to add the lib/ subdir to the loader search path.
# That call runs in main(), AFTER the Windows image loader has already
# resolved the .exe's import table, so it can't be used for normally-
# imported DLLs. To work around: /DELAYLOAD OpenImageDenoise.dll on
# MSVC, which makes the DLL lazy-loaded on first OIDN call. By that
# time main() has run and the search path includes lib/. OIDN's main
# DLL then dlopens its device plugins out of the same dir, also via
# the SetDllDirectory path.
#
# No-op when PCR_USE_OIDN=OFF.
function(pcr_link_oidn target)
    if(NOT PCR_USE_OIDN)
        return()
    endif()
    target_link_libraries(${target} PRIVATE oidn::oidn)
    target_compile_definitions(${target} PRIVATE PCR_HAVE_OIDN)
    if(WIN32)
        if(MSVC)
            target_link_options(${target} PRIVATE
                /DELAYLOAD:OpenImageDenoise.dll
            )
            target_link_libraries(${target} PRIVATE delayimp)
        endif()
        foreach(_dll ${OIDN_RUNTIME_DLLS})
            get_filename_component(_dll_name ${_dll} NAME)
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                        $<TARGET_FILE_DIR:${target}>/lib
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${_dll} $<TARGET_FILE_DIR:${target}>/lib/${_dll_name}
                VERBATIM
            )
        endforeach()
    else()
        set_property(TARGET ${target} APPEND PROPERTY
                     BUILD_RPATH "$ORIGIN/lib")
        set_property(TARGET ${target} APPEND PROPERTY
                     INSTALL_RPATH "$ORIGIN/lib")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    $<TARGET_FILE_DIR:${target}>/lib
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${oidn_SOURCE_DIR}/lib
                    $<TARGET_FILE_DIR:${target}>/lib
            VERBATIM
        )
    endif()
endfunction()

# --- Shared sources used by all binaries ---------------------------------
# These compile cleanly into every target even though some targets don't
# call all the symbols (linker drops dead code).
set(PCR_RENDERER_SOURCES
    Renderer.cpp
    NumGen.cpp
    lodepng.cpp
    Denoise.cpp
    Scenes/Cornell.cpp
    Scenes/CornellSpheres.cpp
    Scenes/CornellLargeLight.cpp
    Scenes/SceneLoader.cpp
    Scenes/SceneDiscovery.cpp
    Scenes/MeshLoader.cpp
    Bvh/Bvh.cpp
    OidnDenoise.cpp
    CIE.cpp
    RGBToSpectrum.cpp
    LutDiscovery.cpp
    SpdLoader.cpp
)

# --- Scene file deployment ----------------------------------------------
# Copy repo-root Scenes/*.json next to each built binary so running the
# binary by path (with no special cwd) still resolves scenes via the
# executable-dir fallback in SceneDiscovery. Uses per-file copy_if_different
# rather than copy_directory because parallel builds otherwise race when
# multiple targets POST_BUILD into the same destination.
#
# Mesh files in repo-root models/ are deployed the same way; JSON scenes
# reference them via relative paths (e.g. "../models/foo.obj") which the
# CMake post-build resolves by mirroring the layout.
set(PCR_SCENES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Scenes)
set(PCR_MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../models)
set(PCR_LUTS_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/../luts)
set(PCR_SPD_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/../spd)
file(GLOB PCR_SCENE_JSON_FILES "${PCR_SCENES_DIR}/*.json")
file(GLOB PCR_MODEL_FILES "${PCR_MODELS_DIR}/*.obj" "${PCR_MODELS_DIR}/*.mtl")
file(GLOB PCR_LUT_FILES "${PCR_LUTS_DIR}/*.lut")
# .spd files may live in subdirectories (spd/cornell/, spd/illuminants/, etc.).
file(GLOB_RECURSE PCR_SPD_FILES RELATIVE "${PCR_SPD_DIR}" "${PCR_SPD_DIR}/*.spd")
function(pcr_deploy_scenes target)
    foreach(scene_file ${PCR_SCENE_JSON_FILES})
        get_filename_component(scene_name ${scene_file} NAME)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    $<TARGET_FILE_DIR:${target}>/Scenes
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${scene_file}
                    $<TARGET_FILE_DIR:${target}>/Scenes/${scene_name}
            VERBATIM
        )
    endforeach()
    foreach(model_file ${PCR_MODEL_FILES})
        get_filename_component(model_name ${model_file} NAME)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    $<TARGET_FILE_DIR:${target}>/models
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${model_file}
                    $<TARGET_FILE_DIR:${target}>/models/${model_name}
            VERBATIM
        )
    endforeach()
    # luts/ ships shippable LUT files (PLUT binary format). Same deploy
    # pattern as Scenes/ and models/. The directory may be empty (no
    # shipped LUTs) but the function still creates the target dir so a
    # GUI "save built LUT here" path resolves to a real folder.
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
                $<TARGET_FILE_DIR:${target}>/luts
        VERBATIM
    )
    foreach(lut_file ${PCR_LUT_FILES})
        get_filename_component(lut_name ${lut_file} NAME)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${lut_file}
                    $<TARGET_FILE_DIR:${target}>/luts/${lut_name}
            VERBATIM
        )
    endforeach()
    # spd/ ships measured-spectrum data files (PLAIN-TEXT .spd format)
    # in possibly-nested subdirs. Mirror the tree next to the binary so
    # albedo_spd: "cornell/white-paint" resolves whether the user runs
    # from the source tree or from the build/install directory. Always
    # creates the spd/ root even when no .spd files are shipped.
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
                $<TARGET_FILE_DIR:${target}>/spd
        VERBATIM
    )
    foreach(spd_rel ${PCR_SPD_FILES})
        get_filename_component(spd_subdir ${spd_rel} DIRECTORY)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    $<TARGET_FILE_DIR:${target}>/spd/${spd_subdir}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${PCR_SPD_DIR}/${spd_rel}
                    $<TARGET_FILE_DIR:${target}>/spd/${spd_rel}
            VERBATIM
        )
    endforeach()
endfunction()

# --- CLI: frank-based-rendering-cli ---------------------------------------
# CPU only, headless. Use this for scripting and homelab servers.
add_executable(frank-based-rendering-cli
    Cli/Main.cpp
    ${PCR_RENDERER_SOURCES}
)
target_include_directories(frank-based-rendering-cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Includes
)
target_link_libraries(frank-based-rendering-cli PRIVATE Threads::Threads)
target_compile_definitions(frank-based-rendering-cli PRIVATE
    PCR_BINARY_NAME="frank-based-rendering-cli"
)

if(MSVC)
    target_compile_options(frank-based-rendering-cli PRIVATE /W4)
else()
    target_compile_options(frank-based-rendering-cli PRIVATE -Wall -Wextra)
endif()
pcr_deploy_scenes(frank-based-rendering-cli)
pcr_link_oidn(frank-based-rendering-cli)

# --- GUI dependencies (shared by both GUI binaries) ---------------------
# Fetch GLFW for static linking. First build downloads + builds it (~30s).
include(FetchContent)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
# X11 only. Wayland support would require libwayland-dev + wayland-protocols
# at build time, which not every Linux box has and CI doesn't bother with.
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)

# Propagate the static-CRT setting to GLFW (and any other FetchContent deps)
# *before* their config runs, so we don't end up with our /MT linking against
# GLFW's /MD and getting CRT-mismatch link errors on MSVC.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
endif()

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glfw)

find_package(OpenGL REQUIRED)

set(IMGUI_SOURCES
    Includes/imgui/imgui.cpp
    Includes/imgui/imgui_draw.cpp
    Includes/imgui/imgui_widgets.cpp
    Includes/imgui/imgui_tables.cpp
    Includes/imgui/backends/imgui_impl_glfw.cpp
    Includes/imgui/backends/imgui_impl_opengl3.cpp
)

set(PCR_GUI_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Includes
    ${CMAKE_CURRENT_SOURCE_DIR}/Includes/imgui
)

set(PCR_GUI_LIBS
    glfw
    OpenGL::GL
    Threads::Threads
)

# Helper to apply MSVC release flags consistently across both GUI targets.
function(pcr_apply_msvc_release_flags target)
    if(MSVC)
        set_property(TARGET ${target} PROPERTY
                     MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
        target_link_options(${target} PRIVATE
                            "/SUBSYSTEM:WINDOWS" "/ENTRY:mainCRTStartup")
    endif()
endfunction()

# Apple silences the OpenGL deprecation cascade. macOS 10.14 marked the
# legacy GL entry points (glGenTextures, glBindTexture, glTexImage2D, ...)
# deprecated, but ImGui's imgui_impl_opengl3 backend and Gui/main.cpp's
# live-preview texture upload still use them. We're not migrating those
# off GL on macOS - the path tracer uses Metal, the ImGui host UI uses
# the legacy GL on macOS like every other GLFW + ImGui app does.
function(pcr_silence_apple_gl_deprecation target)
    if(APPLE)
        target_compile_definitions(${target} PRIVATE GL_SILENCE_DEPRECATION)
    endif()
endfunction()

# --- CPU GUI: frank-based-rendering --------------------------------------
add_executable(frank-based-rendering
    Gui/main.cpp
    ${IMGUI_SOURCES}
    ${PCR_RENDERER_SOURCES}
)
target_include_directories(frank-based-rendering PRIVATE ${PCR_GUI_INCLUDES})
target_link_libraries(frank-based-rendering PRIVATE ${PCR_GUI_LIBS})
target_compile_definitions(frank-based-rendering PRIVATE
    PCR_BINARY_NAME="frank-based-rendering"
    PCR_USE_GPU=0
)
pcr_apply_msvc_release_flags(frank-based-rendering)
pcr_silence_apple_gl_deprecation(frank-based-rendering)
pcr_deploy_scenes(frank-based-rendering)
pcr_link_oidn(frank-based-rendering)

# --- GPU GUI: physically-cringe-rendering --------------------------------
# Two backends, one binary. CMake picks the source file based on
# platform:
#   Apple    -> Gpu/Metal/MetalRenderer.mm   (Metal compute, M1+)
#   else     -> Gpu/Opengl/OpenglRenderer.cpp (OpenGL 4.3 compute)
# Gui/main.cpp includes Gpu/GpuRenderer.h, which is a typedef shim
# that resolves to whichever class the chosen backend exposes.
if(APPLE)
    set(PCR_GPU_BACKEND_SRC Gpu/Metal/MetalRenderer.mm)
else()
    set(PCR_GPU_BACKEND_SRC Gpu/Opengl/OpenglRenderer.cpp)
endif()

add_executable(physically-cringe-rendering
    Gui/main.cpp
    ${PCR_GPU_BACKEND_SRC}
    ${IMGUI_SOURCES}
    ${PCR_RENDERER_SOURCES}
)
target_include_directories(physically-cringe-rendering PRIVATE ${PCR_GUI_INCLUDES})
target_link_libraries(physically-cringe-rendering PRIVATE ${PCR_GUI_LIBS})
target_compile_definitions(physically-cringe-rendering PRIVATE
    PCR_BINARY_NAME="physically-cringe-rendering"
    PCR_USE_GPU=1
)
if(APPLE)
    # ARC on the .mm file (the rest of the codebase is plain C++ and
    # doesn't need it, and -fobjc-arc on .cpp is rejected by clang).
    set_source_files_properties(Gpu/Metal/MetalRenderer.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
    target_link_libraries(physically-cringe-rendering PRIVATE
        "-framework Foundation"
        "-framework Metal"
    )
endif()
pcr_apply_msvc_release_flags(physically-cringe-rendering)
pcr_silence_apple_gl_deprecation(physically-cringe-rendering)
pcr_deploy_scenes(physically-cringe-rendering)
pcr_link_oidn(physically-cringe-rendering)

# --- Tests (opt-in, off by default) -------------------------------------
# Enable with -DPCR_BUILD_TESTS=ON. Pulls in the unit-test binaries under
# tests/unit and registers them with CTest. Render tests are shell-driven
# (tests/render/run.sh) and don't go through CMake; they invoke the
# already-built CLI binary.
option(PCR_BUILD_TESTS "Build unit tests under tests/unit" OFF)
if(PCR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../tests tests-build)
endif()
