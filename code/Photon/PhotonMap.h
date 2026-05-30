#pragma once

#include <cstdint>
#include <numbers>
#include <unordered_map>
#include <vector>

#include "../Includes/Vec3f.h"

// Caustic photon mapping (Jensen 1996) data structures. The renderer
// (CPU / Metal / OpenGL alike) builds one of these at render start
// when --photon-map is on, then queries it on every diffuse hit during
// path tracing to add an estimated caustic-radiance term to the
// per-pixel integral.
//
// "Caustic" here means: photons that have bounced off at least one
// specular surface (mirror or glass) before hitting the diffuse
// surface where they get stored. Eye-side path tracing already
// converges fast on diffuse-only light transport, so storing photons
// at diffuse-direct hits would double-count what NEE already does.
// Limiting to caustic photons keeps the map small and targets exactly
// the variance source path tracing struggles with.

namespace Photon
{
    // A single photon record. Stored in a flat vector inside Map;
    // referenced from the hash grid by index. 36 bytes; deliberately
    // POD-layout so the eventual GPU port can memcpy into an MTLBuffer
    // without serialization.
    //
    // power is RGB linear power (energy per unit time). Each photon
    // carries 1/N of its source light's total emitted flux at shoot
    // time, then gets multiplied by the per-bounce albedo of any
    // mirror/glass surface its path traverses before landing here.
    //
    // wi is the unit direction the photon was TRAVELING at the
    // moment of deposition (i.e., points from prior bounce toward
    // this surface). The density estimator uses wi.N to filter out
    // photons that arrived from below the surface (back-side hits),
    // and a future non-Lambertian BSDF would use wi in the BRDF eval.
    struct Record
    {
        Vec3f position;
        Vec3f wi;
        Vec3f power;
        // Spectral-mode power: per-photon power at the map's 4 hero
        // wavelengths (Map::lambdas()). Only populated when shooting in
        // --spectral mode; RGB renders leave it zero and use `power`.
        // A photon that dispersed at glass carries power in a single
        // channel; a non-dispersed (mirror-only) caustic photon can
        // carry all four. The GPU flatten ignores this field (the GPU
        // GpuRecord stays 36-byte RGB until the spectral GPU port).
        float specPower[4] = {0.f, 0.f, 0.f, 0.f};
    };
    static_assert(sizeof(Record) == 52,
                  "Photon::Record is 52 bytes (RGB power + spectral hero-4); "
                  "GpuFlatten copies only the RGB power into the 36-byte GpuRecord");

    // Spatial index over a photon array. Hash grid with cell size
    // equal to the kernel radius, so a radius-r query needs to check
    // the 3x3x3 = 27 cells centered on the query point's cell.
    //
    // std::unordered_map<hash, vector<idx>> is used for the bucket
    // store, which is allocator-heavy but trivially correct and fine
    // at the photon counts we target (~1M-10M). A flat sorted-by-cell
    // layout with prefix-summed offsets would be faster and is the
    // shape the GPU port needs; we'll switch to that when the Metal
    // port lands (session 2).
    //
    // Build sequence (host-side): construct with the target radius;
    // call insert() for each photon shot; call build() once after
    // all inserts to seal the grid. Queries are then read-only and
    // thread-safe.
    class Map
    {
    public:
        // radius is the kernel radius in scene units. It also doubles
        // as the cell size, so query() touches at most 27 cells.
        explicit Map(float radius);

        // Append one photon record. Updates the hash grid in place.
        void insert(const Record &r);

        // No-op for the current hash-map backend; reserved so callers
        // can express "I'm done inserting" without coupling to the
        // backend choice. The eventual flat-array port will use this
        // to do its prefix-sum pass.
        void build();

        // Invoke visitor(record, distanceSquared) for every photon
        // within `radius()` of point `p`. Templated on F so the visitor
        // inlines; the hot loop ends up branch-light. The visitor
        // returns void; there's no early-exit.
        template <typename F>
        void query(const Vec3f &p, F &&visitor) const;

        size_t size() const { return _photons.size(); }
        float  radius() const { return _radius; }
        const std::vector<Record> &records() const { return _photons; }

        // Spectral mode: the 4 hero wavelengths these photons' specPower
        // channels correspond to. Set once before/after shooting a
        // spectral pass; the spectral density estimate reads them so the
        // eye path (which uses the same per-pass wavelengths) can combine
        // photon power with surface reflectance per wavelength.
        bool isSpectral() const { return _spectral; }
        const float *lambdas() const { return _lambdas; }
        void setSpectralLambdas(const float lam[4])
        {
            _spectral = true;
            for (int k = 0; k < 4; k++) _lambdas[k] = lam[k];
        }

    private:
        // 3D integer cell coordinates from a world position. The
        // floor + cast pattern handles negative coords correctly
        // (where (int)(-0.1f / r) would round toward zero, not floor).
        void cellOf(const Vec3f &p, int &cx, int &cy, int &cz) const;

        // Teschner 2003 spatial hash. Coprime large primes per axis
        // keep neighboring cells distributed across buckets. xor mix
        // is cheaper than the alternative add-and-multiply chain.
        uint64_t cellHash(int cx, int cy, int cz) const;

        float                                      _radius;
        float                                      _invCellSize;
        std::vector<Record>                        _photons;
        std::unordered_map<uint64_t, std::vector<uint32_t>> _grid;
        bool                                       _spectral = false;
        float                                      _lambdas[4] = {0.f, 0.f, 0.f, 0.f};
    };

    // Inline definition. Lives in the header so the visitor template
    // can be instantiated at the call site without an explicit
    // declaration list in the .cpp.
    template <typename F>
    void Map::query(const Vec3f &p, F &&visitor) const
    {
        if (_photons.empty()) return;

        int cx, cy, cz;
        cellOf(p, cx, cy, cz);

        const float r2 = _radius * _radius;
        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            uint64_t h = cellHash(cx + dx, cy + dy, cz + dz);
            auto it = _grid.find(h);
            if (it == _grid.end()) continue;
            for (uint32_t idx : it->second)
            {
                const Record &rec = _photons[idx];
                Vec3f d{rec.position[0] - p[0],
                        rec.position[1] - p[1],
                        rec.position[2] - p[2]};
                float dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if (dist2 > r2) continue;
                visitor(rec, dist2);
            }
        }
    }
}
