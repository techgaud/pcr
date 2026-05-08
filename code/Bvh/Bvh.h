#pragma once

#include <vector>

#include "../Includes/Ray.h"
#include "../Includes/Triangle.h"
#include "../Includes/Vec3f.h"

// Single-triangle scenes don't need this, but as soon as a mesh shows up
// (phase 3) the linear-scan sceneIntersect collapses. The BVH gives log(N)
// expected cost, which is what makes 70k-triangle bunny renders tractable.
//
// This is an object-median-split builder: at each level, sort the triangle
// range by centroid on the longest axis of the parent box and split at the
// middle index. Simpler than SAH and adequate for the uniformly triangulated
// meshes typical of OBJ exports. Phase 3 may revisit with SAH if a non-
// uniform mesh hits traversal hot-spots.
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
}
