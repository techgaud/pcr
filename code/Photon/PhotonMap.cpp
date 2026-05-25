#include "PhotonMap.h"

#include <cmath>

namespace Photon
{
    Map::Map(float radius)
        : _radius{radius}, _invCellSize{1.0f / radius}
    {
        // Reserve modestly. The grid will resize as photons land; if
        // the caller knows the expected photon count up front, the
        // _photons vector reserve is more impactful (done implicitly
        // via insert's vector growth).
        _grid.reserve(1024);
    }

    void Map::cellOf(const Vec3f &p, int &cx, int &cy, int &cz) const
    {
        // floorf, not (int) cast. Negative-axis scenes happen (camera
        // looking at the +Z side of a Cornell box puts a chunk of
        // geometry at negative X / Y), and (int)(-0.1f) is 0 but
        // floorf(-0.1f) is -1.
        cx = (int)std::floor(p[0] * _invCellSize);
        cy = (int)std::floor(p[1] * _invCellSize);
        cz = (int)std::floor(p[2] * _invCellSize);
    }

    uint64_t Map::cellHash(int cx, int cy, int cz) const
    {
        // Teschner 2003 large-prime spatial hash. The three primes are
        // the canonical choice from the original paper.
        uint64_t h = (uint64_t)(uint32_t)cx * 73856093ULL
                   ^ (uint64_t)(uint32_t)cy * 19349663ULL
                   ^ (uint64_t)(uint32_t)cz * 83492791ULL;
        return h;
    }

    void Map::insert(const Record &r)
    {
        uint32_t idx = (uint32_t)_photons.size();
        _photons.push_back(r);

        int cx, cy, cz;
        cellOf(r.position, cx, cy, cz);
        _grid[cellHash(cx, cy, cz)].push_back(idx);
    }

    void Map::build()
    {
        // No-op for the hash-map backend. Reserved entry point so
        // the call site already exists when we swap in a flat-array
        // sorted-by-cell layout for the GPU port.
    }
}
