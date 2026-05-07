cmake_minimum_required(VERSION 3.20)
project(pcr CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# --- Shared sources used by all binaries ---------------------------------
# These compile cleanly into every target even though some targets don't
# call all the symbols (linker drops dead code).
set(PCR_RENDERER_SOURCES
    Renderer.cpp
    NumGen.cpp
    lodepng.cpp
    Denoise.cpp
    Scenes/Cornell.cpp
)

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

# --- GPU GUI: physically-cringe-rendering --------------------------------
# Skipped on Apple platforms: macOS caps OpenGL at 4.1, but the compute
# shader requires 4.3. The right path forward there is a Metal or Vulkan
# backend; until then, the macOS GPU build is omitted from CI releases.
if(NOT APPLE)
    add_executable(physically-cringe-rendering
        Gui/main.cpp
        Gpu/GpuRenderer.cpp
        ${IMGUI_SOURCES}
        ${PCR_RENDERER_SOURCES}
    )
    target_include_directories(physically-cringe-rendering PRIVATE ${PCR_GUI_INCLUDES})
    target_link_libraries(physically-cringe-rendering PRIVATE ${PCR_GUI_LIBS})
    target_compile_definitions(physically-cringe-rendering PRIVATE
        PCR_BINARY_NAME="physically-cringe-rendering"
        PCR_USE_GPU=1
    )
    pcr_apply_msvc_release_flags(physically-cringe-rendering)
endif()
