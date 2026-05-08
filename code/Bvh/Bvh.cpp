#include "Bvh.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <functional>

namespace Bvh
{
    namespace
    {
        // Build heuristic. Object-median splits at the median triangle on
        // the longest axis - cheap to build (O(N log N) per node from one
        // nth_element), produces "fine" trees on uniform geometry but
        // wastes traversal time on concave or non-uniform meshes (the
        // bunny is the canonical example: object-median spreads bunny
        // ears across the same node and intersects the box of either ear
        // for any ray near the head).
        //
        // SAH binning evaluates 16-bin candidate splits along all three
        // axes and picks the one minimizing expected ray-traversal cost
        // per Wald 2007. ~30-50% per-ray speedup on triangle-heavy scenes
        // for ~3-5x build time. Object-median stays around as the fast-
        // build path for development and is the fallback when SAH bins
        // fail to find a productive split.
        constexpr bool kUseSAH = true;

        struct AABB
        {
            float mn[3]{ FLT_MAX, FLT_MAX, FLT_MAX };
            float mx[3]{ -FLT_MAX, -FLT_MAX, -FLT_MAX };

            void includeVertex(const Vec3f &p)
            {
                for (int i = 0; i < 3; i++)
                {
                    if (p[i] < mn[i]) mn[i] = p[i];
                    if (p[i] > mx[i]) mx[i] = p[i];
                }
            }
            void includeTriangle(const Triangle &t)
            {
                includeVertex(t.v0);
                includeVertex(t.v1);
                includeVertex(t.v2);
            }
            void merge(const AABB &o)
            {
                for (int i = 0; i < 3; i++)
                {
                    if (o.mn[i] < mn[i]) mn[i] = o.mn[i];
                    if (o.mx[i] > mx[i]) mx[i] = o.mx[i];
                }
            }
            int longestAxis() const
            {
                float dx = mx[0] - mn[0];
                float dy = mx[1] - mn[1];
                float dz = mx[2] - mn[2];
                if (dx >= dy && dx >= dz) return 0;
                if (dy >= dz) return 1;
                return 2;
            }
            float surfaceArea() const
            {
                float dx = mx[0] - mn[0];
                float dy = mx[1] - mn[1];
                float dz = mx[2] - mn[2];
                if (dx < 0 || dy < 0 || dz < 0) return 0.f; // empty box
                return 2.f * (dx * dy + dy * dz + dz * dx);
            }
        };

        AABB aabbOfRange(const std::vector<Triangle> &tris, int begin, int end)
        {
            AABB box;
            for (int i = begin; i < end; i++)
                box.includeTriangle(tris[i]);
            return box;
        }

        float centroid(const Triangle &t, int axis)
        {
            return (t.v0[axis] + t.v1[axis] + t.v2[axis]) * (1.f / 3.f);
        }

        Vec3f centroidV(const Triangle &t)
        {
            return Vec3f(centroid(t, 0), centroid(t, 1), centroid(t, 2));
        }

        // Leaf threshold: triangle counts at or below this skip the split-
        // cost evaluation entirely. With SAH, leaves can also form at
        // larger counts when the heuristic decides splitting is more
        // expensive than just intersecting the range.
        constexpr int kLeafThreshold = 4;

        // Cost ratios for SAH in MacDonald-Booth normalized units.
        // C_isect dominates traversal cost so picking splits with fewer
        // triangles in each child is favored over balanced child boxes.
        constexpr float kCostTraversal = 1.f;
        constexpr float kCostIntersect = 2.f;

        // Bin count for SAH. Wald 2007's "On Fast Construction of SAH-
        // based BVHs" finds 16 is the sweet spot: more bins barely help
        // tree quality, fewer bins start to lose to full sweep SAH.
        constexpr int kSahBins = 16;

        // Result of evaluating SAH bins for one node. axis = -1 means no
        // productive split was found (e.g. all centroids coincide).
        struct SahSplit
        {
            int axis = -1;
            int bin = -1;
            float cost = FLT_MAX;
        };

        // Bin triangles in tris[begin..end) along all three axes and
        // return the lowest-cost candidate split. Caller compares cost
        // against the leaf cost to decide whether to split or stop.
        SahSplit findSahSplit(const std::vector<Triangle> &tris,
                              int begin, int end,
                              const AABB &nodeBox)
        {
            // SAH bins triangles by their CENTROID (not their bbox), so
            // a triangle's full extent reaches into bins on either side
            // of its centroid bin via the box's actual extent inclusion
            // below. Without using the centroid bbox the binning collapses
            // for nodes whose centroids cluster tightly inside a much
            // wider geometric bbox.
            AABB centroidBox;
            for (int i = begin; i < end; i++)
                centroidBox.includeVertex(centroidV(tris[i]));

            SahSplit best;
            float parentSA = nodeBox.surfaceArea();
            if (parentSA <= 0.f) return best;

            for (int axis = 0; axis < 3; axis++)
            {
                float lo = centroidBox.mn[axis];
                float hi = centroidBox.mx[axis];
                float extent = hi - lo;
                if (extent < 1e-6f) continue;
                float scale = (float)kSahBins / extent;

                struct Bin { AABB box; int count = 0; };
                std::array<Bin, kSahBins> bins{};

                for (int i = begin; i < end; i++)
                {
                    float c = centroid(tris[i], axis);
                    int idx = std::clamp((int)((c - lo) * scale), 0, kSahBins - 1);
                    bins[idx].box.includeTriangle(tris[i]);
                    bins[idx].count++;
                }

                // Sweep: prefix accumulates left[k] over bins [0..k]; the
                // suffix accumulates right[k] over bins [k+1..kSahBins-1].
                std::array<float, kSahBins - 1> leftSA{};
                std::array<int,   kSahBins - 1> leftCount{};
                std::array<float, kSahBins - 1> rightSA{};
                std::array<int,   kSahBins - 1> rightCount{};

                AABB acc;
                int accCount = 0;
                for (int i = 0; i < kSahBins - 1; i++)
                {
                    acc.merge(bins[i].box);
                    accCount += bins[i].count;
                    leftSA[i] = acc.surfaceArea();
                    leftCount[i] = accCount;
                }

                acc = AABB{};
                accCount = 0;
                for (int i = kSahBins - 1; i > 0; i--)
                {
                    acc.merge(bins[i].box);
                    accCount += bins[i].count;
                    rightSA[i - 1] = acc.surfaceArea();
                    rightCount[i - 1] = accCount;
                }

                for (int s = 0; s < kSahBins - 1; s++)
                {
                    if (leftCount[s] == 0 || rightCount[s] == 0) continue;
                    float cost = kCostTraversal
                               + kCostIntersect *
                                 (leftSA[s] * leftCount[s]
                                + rightSA[s] * rightCount[s]) / parentSA;
                    if (cost < best.cost)
                    {
                        best.cost = cost;
                        best.axis = axis;
                        best.bin = s;
                    }
                }
            }
            return best;
        }

        // Partition triangles in [begin, end) into two halves matching
        // the SAH bin split. Returns the partition midpoint. Mirrors the
        // exact bin assignment findSahSplit used so left/right counts
        // match the SAH evaluation.
        int partitionForSah(std::vector<Triangle> &tris, int begin, int end,
                            int axis, int bin)
        {
            AABB centroidBox;
            for (int i = begin; i < end; i++)
                centroidBox.includeVertex(centroidV(tris[i]));
            float lo = centroidBox.mn[axis];
            float hi = centroidBox.mx[axis];
            float scale = (float)kSahBins / (hi - lo);

            auto it = std::partition(
                tris.begin() + begin, tris.begin() + end,
                [&](const Triangle &t) {
                    float c = centroid(t, axis);
                    int idx = std::clamp((int)((c - lo) * scale), 0, kSahBins - 1);
                    return idx <= bin;
                });
            return (int)(it - tris.begin());
        }

        // Recursive build. Returns the index of the root of the subtree
        // covering tris[begin..end). Appends nodes to `nodes` in depth-
        // first order (parent first, then left subtree, then right subtree).
        int buildRecursive(std::vector<Triangle> &tris,
                           std::vector<Node> &nodes,
                           int begin, int end)
        {
            int nodeIdx = (int)nodes.size();
            nodes.emplace_back();

            AABB box = aabbOfRange(tris, begin, end);
            for (int i = 0; i < 3; i++)
            {
                nodes[nodeIdx].boxMin[i] = box.mn[i];
                nodes[nodeIdx].boxMax[i] = box.mx[i];
            }

            int count = end - begin;
            auto makeLeaf = [&]() {
                nodes[nodeIdx].leftOrFirst = begin;
                nodes[nodeIdx].rightChild = 0; // unused for leaves
                nodes[nodeIdx].count = count;
                return nodeIdx;
            };
            if (count <= kLeafThreshold) return makeLeaf();

            int mid;
            if constexpr (kUseSAH)
            {
                // Try SAH split. If the heuristic concludes splitting is
                // more expensive than intersecting the range as a leaf,
                // accept that and stop recursing.
                SahSplit split = findSahSplit(tris, begin, end, box);
                float leafCost = (float)count * kCostIntersect;
                if (split.axis < 0 || split.cost >= leafCost) return makeLeaf();

                mid = partitionForSah(tris, begin, end, split.axis, split.bin);
                if (mid == begin || mid == end) return makeLeaf();
            }
            else
            {
                // Object-median split on the longest axis.
                int axis = box.longestAxis();
                mid = begin + count / 2;
                std::nth_element(tris.begin() + begin,
                                 tris.begin() + mid,
                                 tris.begin() + end,
                                 [axis](const Triangle &a, const Triangle &b) {
                                     return centroid(a, axis) < centroid(b, axis);
                                 });
                // Degenerate split (everything has the same centroid
                // component on this axis). Bail to a leaf.
                if (mid == begin || mid == end) return makeLeaf();
            }

            int leftIdx = buildRecursive(tris, nodes, begin, mid);
            int rightIdx = buildRecursive(tris, nodes, mid, end);

            nodes[nodeIdx].leftOrFirst = leftIdx;
            nodes[nodeIdx].rightChild = rightIdx;
            nodes[nodeIdx].count = 0;
            return nodeIdx;
        }

        // Slab-method ray-AABB. Returns true if the ray segment (0, segMax)
        // intersects the box; tNear is the entry distance.
        bool rayAabb(const Ray &ray, const float *mn, const float *mx,
                     float segMax, float &tNear)
        {
            float tmin = 0.f;
            float tmax = segMax;
            for (int i = 0; i < 3; i++)
            {
                float invD = 1.f / ray.dir[i];
                float t1 = (mn[i] - ray.origin[i]) * invD;
                float t2 = (mx[i] - ray.origin[i]) * invD;
                if (t1 > t2) std::swap(t1, t2);
                tmin = std::max(tmin, t1);
                tmax = std::min(tmax, t2);
                if (tmin > tmax) return false;
            }
            tNear = tmin;
            return true;
        }
    } // namespace

    std::vector<Node> build(std::vector<Triangle> &triangles)
    {
        std::vector<Node> nodes;
        if (triangles.empty()) return nodes;

        // Worst case: 2N - 1 nodes for N leaves of size 1; with leaf size 4
        // it's smaller. Reserve the worst case so emplace_back never
        // reallocates and invalidates references mid-recursion.
        nodes.reserve(2 * triangles.size());
        buildRecursive(triangles, nodes, 0, (int)triangles.size());
        return nodes;
    }

    BuildStats statsOf(const std::vector<Node> &nodes)
    {
        BuildStats s;
        s.nodeCount = (int)nodes.size();
        for (const auto &n : nodes)
        {
            if (n.count > 0)
            {
                s.leafCount++;
                s.totalLeafTris += n.count;
                if (n.count > s.maxLeafTris) s.maxLeafTris = n.count;
            }
        }
        // Depth = max stack push depth from a recursive walk.
        if (!nodes.empty())
        {
            std::function<int(int)> depthOf = [&](int idx) {
                const Node &n = nodes[idx];
                if (n.count > 0) return 1;
                return 1 + std::max(depthOf(n.leftOrFirst), depthOf(n.rightChild));
            };
            s.depth = depthOf(0);
        }
        return s;
    }

    bool intersect(const std::vector<Node> &nodes,
                   const std::vector<Triangle> &triangles,
                   const Ray &ray,
                   Vec3f &hit, Vec3f &N, int &matIdx,
                   float &t_out, float closest_t)
    {
        if (nodes.empty()) return false;

        constexpr int kStackMax = 64;
        // Each stack entry caches the AABB tNear from the time it was
        // pushed. At pop we compare against the current `closest`; if a
        // closer hit was found in another subtree since this entry was
        // pushed, we drop it without re-testing the AABB. This is the
        // payoff of ordered traversal: closer hits found via the near
        // child cull entries that point at farther subtrees.
        struct StackEntry { int idx; float tNear; };
        StackEntry stack[kStackMax];
        int top = 0;
        {
            float tRoot;
            if (!rayAabb(ray, nodes[0].boxMin, nodes[0].boxMax, closest_t, tRoot))
                return false;
            stack[top++] = {0, tRoot};
        }

        bool anyHit = false;
        float closest = closest_t;

        while (top > 0)
        {
            StackEntry e = stack[--top];
            if (e.tNear > closest) continue;
            const Node &n = nodes[e.idx];

            if (n.count > 0)
            {
                // Leaf: linear test over the (small) triangle range.
                Vec3f triHit, triN;
                float tt;
                for (int i = 0; i < n.count; i++)
                {
                    int triIdx = n.leftOrFirst + i;
                    if (!triangles[triIdx].intersect(ray, triHit, triN, tt, closest))
                        continue;
                    closest = tt;
                    hit = triHit;
                    N = triN;
                    matIdx = triangles[triIdx].matIdx;
                    anyHit = true;
                }
                continue;
            }

            // Internal: test both children's AABBs and push them in
            // far-first order so the near child is popped (and traversed)
            // first. The near child often finds a hit close enough to
            // cull the far child entirely on its eventual pop.
            const Node &cl = nodes[n.leftOrFirst];
            const Node &cr = nodes[n.rightChild];
            float tL, tR;
            bool hitL = rayAabb(ray, cl.boxMin, cl.boxMax, closest, tL);
            bool hitR = rayAabb(ray, cr.boxMin, cr.boxMax, closest, tR);

            if (hitL && hitR)
            {
                if (top + 2 > kStackMax) continue;
                if (tL <= tR)
                {
                    stack[top++] = {n.rightChild,  tR};
                    stack[top++] = {n.leftOrFirst, tL};
                }
                else
                {
                    stack[top++] = {n.leftOrFirst, tL};
                    stack[top++] = {n.rightChild,  tR};
                }
            }
            else if (hitL)
            {
                if (top + 1 > kStackMax) continue;
                stack[top++] = {n.leftOrFirst, tL};
            }
            else if (hitR)
            {
                if (top + 1 > kStackMax) continue;
                stack[top++] = {n.rightChild, tR};
            }
        }

        if (anyHit) t_out = closest;
        return anyHit;
    }
}
