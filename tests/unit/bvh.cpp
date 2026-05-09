// BVH unit tests.
//
// The BVH is the closest-hit accelerator for triangle scenes. Two
// invariants we need to nail down:
//   1. Build is robust on degenerate input (empty, coplanar, single tri).
//   2. Traversal returns the same closest triangle as a brute-force scan
//      on a known set of test rays.
//
// Render-diff tests in tests/render exercise the full BVH on cornell-bunny
// (~70k triangles); these unit tests focus on the small-mesh edge cases
// that golden-diff renders won't ever hit.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "Bvh/Bvh.h"
#include "Triangle.h"
#include "Ray.h"
#include "Vec3f.h"

namespace {

int g_failed = 0;

void check(bool ok, const char *label)
{
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", label);
    if (!ok) g_failed++;
}

bool nearly(float a, float b, float eps)
{
    return std::fabs(a - b) <= eps;
}

void test_empty()
{
    std::vector<Triangle> tris;
    auto nodes = Bvh::build(tris);
    check(nodes.empty(), "empty triangle list -> empty BVH");

    Vec3f hit, N;
    int matIdx = -1;
    float t = 0.f;
    Vec3f rayDir(0, 0, -1), rayOrig(0, 0, 0);
    Ray r(rayDir, rayOrig);
    bool got = Bvh::intersect(nodes, tris, r, hit, N, matIdx, t, 1e30f);
    check(!got, "intersect on empty BVH returns false");
}

void test_single_triangle()
{
    std::vector<Triangle> tris;
    tris.emplace_back(Vec3f(-1, -1, -5), Vec3f(1, -1, -5), Vec3f(0, 1, -5), /*matIdx=*/7);
    auto nodes = Bvh::build(tris);
    check(!nodes.empty(), "single triangle -> non-empty BVH");

    Vec3f hit, N;
    int matIdx = -1;
    float t = 0.f;
    Vec3f rayDir(0, 0, -1), rayOrig(0, 0, 0);
    Ray r(rayDir, rayOrig);
    bool got = Bvh::intersect(nodes, tris, r, hit, N, matIdx, t, 1e30f);
    check(got, "ray hits single triangle");
    check(matIdx == 7, "single-triangle hit returns correct matIdx");
    check(nearly(t, 5.f, 1e-3f), "single-triangle hit at t=5");
}

void test_coplanar_triangles_terminate()
{
    // 100 coplanar triangles. Object-median would split degenerately on
    // the long axes; the builder must terminate either way (SAH falls back
    // to leaf when no productive split exists).
    std::vector<Triangle> tris;
    for (int i = 0; i < 100; i++) {
        float x = float(i) * 0.01f;
        tris.emplace_back(Vec3f(x, 0, -5),
                          Vec3f(x + 0.005f, 0, -5),
                          Vec3f(x, 0.01f, -5),
                          /*matIdx=*/0);
    }
    auto nodes = Bvh::build(tris);
    check(!nodes.empty(), "100 coplanar triangles build without infinite recursion");

    auto stats = Bvh::statsOf(nodes);
    check(stats.depth < 60, "coplanar BVH depth bounded (< 60)");
}

// Build a "scene" of 200 random small triangles distributed in a unit
// cube around the origin. Pseudo-random with a fixed seed so the test
// is deterministic across runs and platforms.
std::vector<Triangle> randomSceneTriangles(int N)
{
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<Triangle> tris;
    tris.reserve(N);
    for (int i = 0; i < N; i++) {
        Vec3f a(d(rng), d(rng), d(rng) - 5.f);
        Vec3f b(a[0] + 0.05f * d(rng), a[1] + 0.05f * d(rng), a[2] + 0.05f * d(rng));
        Vec3f c(a[0] + 0.05f * d(rng), a[1] + 0.05f * d(rng), a[2] + 0.05f * d(rng));
        tris.emplace_back(a, b, c, i);
    }
    return tris;
}

void test_build_stats_synthetic()
{
    auto tris = randomSceneTriangles(200);
    auto nodes = Bvh::build(tris);

    auto stats = Bvh::statsOf(nodes);
    check(stats.leafCount > 0, "BVH has leaves");
    check(stats.depth < 30, "synthetic BVH depth < 30");
    if (stats.leafCount > 0) {
        float avg = float(stats.totalLeafTris) / float(stats.leafCount);
        check(avg <= 8.f, "synthetic BVH avg leaf size <= 8");
    }
}

// Brute-force closest-hit over the triangle list. Returns the matIdx
// of the closest triangle, or -1 if no hit.
int bruteForceClosest(const std::vector<Triangle> &tris, const Ray &ray, float &outT)
{
    int best = -1;
    float bestT = 1e30f;
    for (size_t i = 0; i < tris.size(); i++) {
        Vec3f hit, N;
        float t;
        if (tris[i].intersect(ray, hit, N, t, bestT)) {
            bestT = t;
            best = tris[i].matIdx;
        }
    }
    outT = bestT;
    return best;
}

void test_bvh_matches_brute_force()
{
    // Same triangle set, ten different rays. BVH and brute-force should
    // agree on closest matIdx and t for every ray. Even one disagreement
    // is a serious bug (closest-hit is the renderer's primary correctness
    // contract for BVH; a wrong-triangle answer means wrong colors).
    auto tris = randomSceneTriangles(200);
    auto trisCopy = tris; // build permutes; preserve original for brute force.
    auto nodes = Bvh::build(tris);

    std::mt19937 rng(0xBADD);
    std::uniform_real_distribution<float> d(-1.f, 1.f);

    int agreed = 0, disagreed = 0;
    for (int i = 0; i < 30; i++) {
        Vec3f origin(d(rng), d(rng), 0.f);
        Vec3f dir(d(rng) * 0.2f, d(rng) * 0.2f, -1.f);
        dir = dir.normalize();
        Ray r(dir, origin);

        Vec3f hit, N;
        int matIdx = -1;
        float t = 0.f;
        bool gotBvh = Bvh::intersect(nodes, tris, r, hit, N, matIdx, t, 1e30f);

        float tBrute;
        int matIdxBrute = bruteForceClosest(trisCopy, r, tBrute);

        bool match;
        if (matIdxBrute == -1) {
            match = !gotBvh;
        } else {
            match = gotBvh && matIdx == matIdxBrute && nearly(t, tBrute, 1e-3f);
        }
        if (match) agreed++;
        else       disagreed++;
    }
    check(disagreed == 0, "BVH closest-hit matches brute-force across 30 rays");
}

} // namespace

int main()
{
    test_empty();
    test_single_triangle();
    test_coplanar_triangles_terminate();
    test_build_stats_synthetic();
    test_bvh_matches_brute_force();

    if (g_failed) {
        std::printf("\n%d FAIL(s)\n", g_failed);
        return 1;
    }
    std::printf("\nAll bvh tests passed.\n");
    return 0;
}
