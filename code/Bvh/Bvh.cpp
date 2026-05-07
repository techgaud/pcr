#include "Bvh.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace Bvh
{
    namespace
    {
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
            int longestAxis() const
            {
                float dx = mx[0] - mn[0];
                float dy = mx[1] - mn[1];
                float dz = mx[2] - mn[2];
                if (dx >= dy && dx >= dz) return 0;
                if (dy >= dz) return 1;
                return 2;
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

        // Leaf threshold: traversal cost vs triangle-test cost trade-off.
        // 4 is a common sweet spot for similarly-sized triangles; phase 3
        // may tune for the bunny.
        constexpr int kLeafThreshold = 4;

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
            if (count <= kLeafThreshold)
            {
                nodes[nodeIdx].leftOrFirst = begin;
                nodes[nodeIdx].rightChild = 0; // unused for leaves
                nodes[nodeIdx].count = count;
                return nodeIdx;
            }

            // Object-median split on the longest axis.
            int axis = box.longestAxis();
            int mid = begin + count / 2;
            std::nth_element(tris.begin() + begin,
                             tris.begin() + mid,
                             tris.begin() + end,
                             [axis](const Triangle &a, const Triangle &b) {
                                 return centroid(a, axis) < centroid(b, axis);
                             });

            // Degenerate split (everything has the same centroid component
            // on this axis). Bail to a leaf — the traversal cost is fine
            // because rayAabb will still cull the range when possible.
            if (mid == begin || mid == end)
            {
                nodes[nodeIdx].leftOrFirst = begin;
                nodes[nodeIdx].rightChild = 0;
                nodes[nodeIdx].count = count;
                return nodeIdx;
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

    bool intersect(const std::vector<Node> &nodes,
                   const std::vector<Triangle> &triangles,
                   const Ray &ray,
                   Vec3f &hit, Vec3f &N, Material &material,
                   float &t_out, float closest_t)
    {
        if (nodes.empty()) return false;

        constexpr int kStackMax = 64;
        int stack[kStackMax];
        int top = 0;
        stack[top++] = 0; // root

        bool anyHit = false;
        float closest = closest_t;

        while (top > 0)
        {
            int idx = stack[--top];
            const Node &n = nodes[idx];

            float tNear;
            if (!rayAabb(ray, n.boxMin, n.boxMax, closest, tNear))
                continue;
            if (tNear > closest) continue;

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
                    material = triangles[triIdx].material;
                    anyHit = true;
                }
            }
            else
            {
                // Internal: push both children. Ordered traversal (push the
                // farther child first so the nearer is tested first) is a
                // small win on coherent rays; skipping it keeps the code
                // simpler and the scene sizes phase 2 ships with don't
                // need it.
                if (top + 2 > kStackMax) continue; // belt-and-suspenders
                stack[top++] = n.leftOrFirst;
                stack[top++] = n.rightChild;
            }
        }

        if (anyHit) t_out = closest;
        return anyHit;
    }
}
