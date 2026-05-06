cmake_minimum_required(VERSION 3.20)
project(pcr-cornell CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# --- Shared renderer sources used by both CLI and GUI ---------------------
set(PCR_RENDERER_SOURCES
    Renderer.cpp
    NumGen.cpp
    lodepng.cpp
    Scenes/Cornell.cpp
)

# --- CLI target -----------------------------------------------------------
add_executable(pcr-cornell
    Main.cpp
    ${PCR_RENDERER_SOURCES}
)
target_include_directories(pcr-cornell PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Includes
)
target_link_libraries(pcr-cornell PRIVATE Threads::Threads)

if(MSVC)
    target_compile_options(pcr-cornell PRIVATE /W4)
else()
    target_compile_options(pcr-cornell PRIVATE -Wall -Wextra)
endif()

# --- GUI target -----------------------------------------------------------
# Fetch GLFW for static linking. First build downloads + builds it (~30s).
include(FetchContent)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
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

add_executable(physically-cringe-renderer-v1.0.0
    Gui/main.cpp
    ${IMGUI_SOURCES}
    ${PCR_RENDERER_SOURCES}
)
target_include_directories(physically-cringe-renderer-v1.0.0 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Includes
    ${CMAKE_CURRENT_SOURCE_DIR}/Includes/imgui
)
target_link_libraries(physically-cringe-renderer-v1.0.0 PRIVATE
    glfw
    OpenGL::GL
    Threads::Threads
)

# Static MSVC runtime so the .exe ships without VC++ Redistributable.
# Use /SUBSYSTEM:WINDOWS so a console window doesn't pop up alongside the GUI.
if(MSVC)
    set_property(TARGET physically-cringe-renderer-v1.0.0 PROPERTY
                 MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_link_options(physically-cringe-renderer-v1.0.0 PRIVATE
                        "/SUBSYSTEM:WINDOWS" "/ENTRY:mainCRTStartup")
endif()
