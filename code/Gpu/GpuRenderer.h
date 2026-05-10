#pragma once

// Backend picker. The class GpuRenderer below resolves to whichever
// concrete renderer the host platform can actually drive:
//
//   Apple platforms  -> MetalRenderer  (Metal compute, M1+)
//   Everything else  -> OpenglRenderer (OpenGL 4.3 compute, Win/Linux)
//
// Both classes expose the same public interface (constructor, render(),
// the same set of public bool quality fields, etc.) so Gui/main.cpp can
// hold a `GpuRenderer` value and remain backend-agnostic. CMake selects
// which .cpp/.mm gets compiled, so only one backend is ever linked into
// a given binary; the other isn't even pulled in.

#if defined(__APPLE__)
    #include "Metal/MetalRenderer.h"
    using GpuRenderer = MetalRenderer;
#else
    #include "Opengl/OpenglRenderer.h"
    using GpuRenderer = OpenglRenderer;
#endif
