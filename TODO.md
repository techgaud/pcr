# TODO

Deferred work, with enough context to pick up cold later.

## macOS signed + notarized .dmg distribution

**Status:** not started. The current macOS release flow ships a per-OS bundle ZIP. macOS Gatekeeper blocks any binary downloaded from the internet that isn't notarized; users have to `xattr -dr com.apple.quarantine` the extracted folder before launch. A signed and notarized .dmg removes that friction.

### Why deferred

Apple Developer Program membership is $99/year, which is not worth paying when the audience for pcr is one person who can also run `xattr` from a terminal. The first-launch friction is small and explained in the release body's install instructions.

### When to revisit

When pcr has an audience beyond Nate. Or when Nate wants the polished "double-click .dmg, drag to Applications" install flow.

### Implementation outline

1. Apple Developer Program membership.
2. Generate a Developer ID Application certificate, install it on the build machine.
3. CMake `BUNDLE` generator: build `physically-cringe-rendering.app` and `frank-based-rendering.app` as proper `.app` bundles (Info.plist, executable in Contents/MacOS, OIDN dylib in Contents/Frameworks). The CLI stays a plain command-line binary.
4. `codesign --deep --sign "Developer ID Application: <name>" --options runtime <app>` to sign the bundles + nested dylibs.
5. `xcrun notarytool submit <dmg> --apple-id <id> --team-id <id> --password <app-password> --wait` to notarize.
6. `xcrun stapler staple <dmg>` to embed the notarization ticket.
7. Probably easiest as a separate workflow step running on the macos-latest CI runner with the cert + app password stored as repo secrets.

Cost: ~1-2 hours of CMake + GitHub Actions YAML once the Apple Developer setup is done.

## MetalRT hardware ray tracing (M3+ only)

**Status:** not started. M1 Ultra (Nate's current Mac Studio) doesn't expose MetalRT, so this is gated by future hardware. M3 and later have hardware-accelerated ray tracing via `MTLAccelerationStructure` + `MTLIntersectionFunctionTable`.

### Why deferred

Wrong hardware. Even on M3+, this is a pure enhancement: pcr's BVH traversal works fine, MetalRT would just make it faster. Not a correctness fix.

### What it would replace

The MSL kernel's `intersectBvh()` function (stack-based BVH traversal in private memory, ~70 lines of MSL) gets replaced by a `MTLIntersectionQuery` against an `MTLAccelerationStructure` built from the scene's triangles at upload time. The host side builds the acceleration structure with a `MTLAccelerationStructureCommandEncoder`; the kernel does `query.intersect()` and reads back hit info.

### When to revisit

When Nate has an M3-or-later Mac. Until then, M1 Ultra at saturated multi-pass throughput is plenty fast for everything pcr does.

### Risk

MetalRT only accelerates triangles, not spheres or planes. We'd need to keep the existing intersection code for non-triangle primitives and fall back to MetalRT only for the BVH traversal. Mixed dispatch adds kernel complexity. Worth measuring before committing, for cornell-class scenes with few triangles the speedup may not justify the maintenance cost.
