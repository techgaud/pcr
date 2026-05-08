#pragma once

#include <vector>

#include "../Includes/Ray.h"
#include "../Includes/Triangle.h"
#include "../Includes/Vec3f.h"

// Single-triangle scenes don't need this, but as soon as a mesh shows up
// (phase 3) the linear-scan sceneIntersect collapses. The BVH gives log(N)
// expected cost, which is what makes 70k-triangle bunny renders tractable.
//
// Default builder is Wald 2007 16-bin SAH: at each internal node, evaluate
// candidate splits along all three axes by surface-area-weighted child
// cost and pick the lowest. Falls back to leaf when SAH says the split
// costs more than just intersecting the range. Object-median splits are
// retained behind a kUseSAH constexpr toggle for debugging and as a
// faster-to-build fallback when render budgets are tiny.
//
// Layout: array of nodes, depth-first allocation order. Internal nodes
// store an explicit pair of child indices (leftChild, rightChild). Leaves
// store a triangle range (firstTri, count). count == 0 marks an internal
// node. The triangle vector is permuted in place during build so each
// leaf's triangles are contiguous, which is friendlier to the CPU
// prefetcher than indexing through an index list.
namespace Bvh
{
    struct Node
    {
        float boxMin[3];
        float boxMax[3];
        int leftOrFirst;   // internal: left-child idx; leaf: first triangle idx
        int rightChild;    // internal: right-child idx; leaf: unused
        int count;         // 0 = internal; > 0 = leaf with this many triangles
    };

    // Builds a BVH over `triangles`. Permutes the triangle vector so leaves
    // index contiguous ranges. Returns a flat node array, depth-first, with
    // node 0 as the root. Returns an empty vector when triangles is empty.
    std::vector<Node> build(std::vector<Triangle> &triangles);

    // Closest-hit traversal. Iterative with a fixed-size stack (64 covers
    // any practical BVH depth. even an adversarially-skewed object-median
    // tree over 2^64 triangles fits).
    //
    // Returns true if any triangle in [0, closest_t) is hit; sets hit/N/
    // matIdx to the closest hit's material registry index, and t_out to
    // its parametric distance.
    bool intersect(const std::vector<Node> &nodes,
                   const std::vector<Triangle> &triangles,
                   const Ray &ray,
                   Vec3f &hit, Vec3f &N, int &matIdx,
                   float &t_out, float closest_t);

    // Tree shape stats. Useful for development and for the optional
    // build-stats log a SceneLoader can print at scene-load time.
    struct BuildStats
    {
        int nodeCount     = 0;
        int leafCount     = 0;
        int totalLeafTris = 0;  // sum of count across leaves
        int maxLeafTris   = 0;
        int depth         = 0;  // longest root-to-leaf chain
    };
    BuildStats statsOf(const std::vector<Node> &nodes);
}
