#pragma once

// Tell the Windows loader to also look in the binary's `lib/` subfolder
// when resolving DLLs. Called once at startup so OIDN's main DLL and its
// dlopen-ed device plugins both resolve out of `lib/`. No-op on Linux and
// macOS, where ELF/Mach-O RPATH (set in CMake) handles this at link time.

#ifdef _WIN32
#include <windows.h>
#include <cstring>
#include <string>
#endif

inline void pcrSetupLibSearch()
{
#ifdef _WIN32
    char exePath[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return;
    char *slash = std::strrchr(exePath, '\\');
    if (!slash) slash = std::strrchr(exePath, '/');
    if (!slash) return;
    *slash = '\0';
    std::string libDir = std::string(exePath) + "\\lib";
    SetDllDirectoryA(libDir.c_str());
#endif
}
