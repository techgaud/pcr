# TODO

Deferred work, with enough context to pick up cold later.

## Surface Area Heuristic BVH

**Status:** designed, not implemented. Currently `Bvh::build` in `code/Bvh/Bvh.cpp` uses object-median splits — at each internal node, it sorts triangles along the longest AABB axis and splits at the median index. Simple, fast to build, and produces ~17 traversal-step paths through the bunny (70k triangles, log2(70k/4) ≈ 14 plus traversal overhead). SAH typically halves that.

### Why deferred

Object-median works fine for cornell-class scenes (zero or few triangles) and is reasonable for the bunny. SAH becomes a real win once you're rendering scenes with hundreds of thousands of triangles or doing many short renders against the same geometry, and we're not there yet.

### The algorithm

At each internal node, instead of splitting at the median, evaluate candidate splits and pick the one minimizing:

```
SAH(split) = C_trav + (P_left * N_left * C_isect) + (P_right * N_right * C_isect)
```

where:
- `C_trav` is the cost of traversing one node (~1 in MacDonald-Booth normalized units).
- `C_isect` is the cost of one ray-triangle intersection (~2 in the same units).
- `P_left` = surface area of left child's AABB / parent's AABB surface area.
- `P_right` = same for right child.
- `N_left`, `N_right` = triangle counts in each child.

You're trying to minimize *expected* traversal cost over a uniform-direction ray hitting the parent, given that hit probability scales with surface area.

### Sweep variant (simplest, what I'd implement)

For each axis (x, y, z):
1. Sort triangles by centroid along that axis.
2. Sweep left-to-right computing prefix AABB and prefix triangle count.
3. Sweep right-to-left computing suffix AABB and suffix triangle count.
4. For each split position k (1 to N-1), compute SAH cost using prefix[k] for left, suffix[k] for right.
5. Track the minimum cost split across all axes.

Compare best SAH cost against `N * C_isect` (cost of making this node a leaf instead). If leaf is cheaper, stop recursing.

### Bin variant (faster build, almost-identical quality)

Sweep is `O(N log N)` per node from the sort. Binning reduces to `O(N * B)` where B is bin count (typically 16):
1. For each axis, divide the AABB into B equal-width bins along that axis.
2. Walk triangles once, bucketing each into its bin (by centroid).
3. Compute SAH cost at each of the B-1 bin boundaries.
4. Pick the minimum.

Wald 2007 ("On Fast Construction of SAH-based Bounding Volume Hierarchies") is the canonical reference. Used in PBRT, Embree, every production tracer.

### Where to make the change

Single file: `code/Bvh/Bvh.cpp`. The recursive `buildRecursive` function picks the split. Object-median's split logic is ~20 lines; SAH-binning is ~80. The Node layout, `intersect` traversal, and everything else stays unchanged.

A `kUseSAH` constexpr at the top of Bvh.cpp would let you A/B object-median vs SAH builds during development without ripping out the existing code.

### What you'd see

- Build time goes up. Object-median is ~50ms for the bunny; SAH binning is ~150-300ms. One-time cost at scene load.
- Render time on triangle-heavy scenes drops 30-50%. The bunny especially benefits because object-median tends to produce skinny boxes for its concave geometry.
- Scenes with no triangles or few triangles see no change (the BVH is empty or trivial).

### Why this isn't urgent right now

- pcr's hottest scene is cornell-bunny at 70k triangles. Object-median is "fine" there — bunny renders complete in seconds, not minutes.
- Real win threshold is around 100k+ triangles or scenes where you're iterating renders against the same geometry many times.
- The work is well-bounded and can come back at any time without disturbing other systems. Not a blocker for anything else.

## Make the GPU binary portable (single-file Windows .exe)

**Status:** not started. Today the binary lives at `Build/frank-based-rendering-cli` and you invoke it with the path. To run it as just `frank-based-rendering-cli` from any directory, add a CMake `install()` rule and `cmake --install`.

### Why deferred

For solo dev on this machine the explicit path (`Build/frank-based-rendering-cli`) is fine and avoids polluting `/usr/local/bin`. Worth doing when:

- You want to type `frank-based-rendering-cli ...` from any cwd without remembering the path
- You distribute releases (Homebrew formula, Debian package, GitHub Releases binaries)
- A second person wants to use it on their machine without learning the source layout

### CMake change

Add to `CMakeLists.txt` after the `add_executable` block:

```cmake
include(GNUInstallDirs)
install(TARGETS frank-based-rendering-cli
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

- Linux/macOS without `--prefix`: `/usr/local/bin/frank-based-rendering-cli`
- With `--prefix ~/.local`: `~/.local/bin/frank-based-rendering-cli`
- Windows without `--prefix`: `C:\Program Files\frank-based-rendering-cli\bin\frank-based-rendering-cli.exe`, requires admin

### PATH considerations

After installing, the install dir needs to be on `$PATH` for "just type `frank-based-rendering-cli`" to work.

- **Linux:** `/usr/local/bin` is on `$PATH` by default. `~/.local/bin` may not be. Add to `~/.bashrc` or `~/.zshrc` if needed:
  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  ```
- **macOS:** Same as Linux. `~/.local/bin` is not on `$PATH` by default on most setups.
- **Windows:** Default install prefix `Program Files\frank-based-rendering-cli\bin` is *not* on `PATH`. Either edit System Properties → Environment Variables, or install with `--prefix %LOCALAPPDATA%\pcr` and add that bin dir to user `PATH`.

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
ls /tmp/pcr-test/bin/   # should contain frank-based-rendering-cli
/tmp/pcr-test/bin/frank-based-rendering-cli --help
```

If that works, real install with `cmake --install Build` (with appropriate sudo / admin / `--prefix`).

### Related future work

- A `cpack` config to produce `.deb`/`.rpm`/`.pkg` packages instead of just installing files. Useful if pcr ever gets distributed.
- A GitHub Actions workflow that runs `cmake --install` and uploads the binary to GitHub Releases. Useful if pcr ever gets multiple users on different platforms.
