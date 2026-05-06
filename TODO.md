# TODO

Deferred work, with enough context to pick up cold later.

## Add a `cmake --install` flow for system-wide / user-local install

**Status:** not started. Today the binary lives at `Build/pcr-cornell` and you invoke it with the path. To run it as just `pcr-cornell` from any directory, add a CMake `install()` rule and `cmake --install`.

### Why deferred

For solo dev on this machine the explicit path (`Build/pcr-cornell`) is fine and avoids polluting `/usr/local/bin`. Worth doing when:

- You want to type `pcr-cornell ...` from any cwd without remembering the path
- You distribute releases (Homebrew formula, Debian package, GitHub Releases binaries)
- A second person wants to use it on their machine without learning the source layout

### CMake change

Add to `CMakeLists.txt` after the `add_executable` block:

```cmake
include(GNUInstallDirs)
install(TARGETS pcr-cornell
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
```

`GNUInstallDirs` provides `CMAKE_INSTALL_BINDIR`, which is `bin` on Unix. This is what GNU/Debian package conventions expect.

### Usage after that

After building, install with one of:

```bash
# System-wide (Linux/macOS), requires sudo
sudo cmake --install Build

# User-local (Linux/macOS), no sudo
cmake --install Build --prefix ~/.local

# Custom location
cmake --install Build --prefix /opt/pcr
```

Default install paths:

- Linux/macOS without `--prefix`: `/usr/local/bin/pcr-cornell`
- With `--prefix ~/.local`: `~/.local/bin/pcr-cornell`
- Windows without `--prefix`: `C:\Program Files\pcr-cornell\bin\pcr-cornell.exe`, requires admin

### PATH considerations

After installing, the install dir needs to be on `$PATH` for "just type `pcr-cornell`" to work.

- **Linux:** `/usr/local/bin` is on `$PATH` by default. `~/.local/bin` may not be. Add to `~/.bashrc` or `~/.zshrc` if needed:
  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  ```
- **macOS:** Same as Linux. `~/.local/bin` is not on `$PATH` by default on most setups.
- **Windows:** Default install prefix `Program Files\pcr-cornell\bin` is *not* on `PATH`. Either edit System Properties → Environment Variables, or install with `--prefix %LOCALAPPDATA%\pcr` and add that bin dir to user `PATH`.

### Uninstall

CMake writes an install manifest at install time:

```bash
xargs rm -v < Build/install_manifest.txt
```

Or just delete the binary by hand. CMake doesn't generate a `cmake --uninstall` by default.

### Testing the install rule

After adding the `install()` line:

```bash
cmake -S . -B Build
cmake --build Build
cmake --install Build --prefix /tmp/pcr-test
ls /tmp/pcr-test/bin/   # should contain pcr-cornell
/tmp/pcr-test/bin/pcr-cornell --help
```

If that works, real install with `cmake --install Build` (with appropriate sudo / admin / `--prefix`).

### Related future work

- A `cpack` config to produce `.deb`/`.rpm`/`.pkg` packages instead of just installing files. Useful if pcr ever gets distributed.
- A GitHub Actions workflow that runs `cmake --install` and uploads the binary to GitHub Releases. Useful if pcr ever gets multiple users on different platforms.
