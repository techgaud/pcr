cmake_minimum_required(VERSION 3.20)
project(pcr CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_executable(pcr
    Main.cpp
    Renderer.cpp
    NumGen.cpp
)

target_include_directories(pcr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pcr PRIVATE Threads::Threads)

if(MSVC)
    target_compile_options(pcr PRIVATE /W4)
else()
    target_compile_options(pcr PRIVATE -Wall -Wextra)
endif()
