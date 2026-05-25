#pragma once

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "PhotonMap.h"

// GPU-flattening of a Photon::Map. The CPU side stores photons in a
// per-cell std::unordered_map for cheap insertion + neighborhood
// queries during eye-path tracing. The GPU side needs the same logical
// structure in a layout that survives an MTLBuffer / SSBO upload and
// supports per-thread lookups without a heap-allocated map.
//
// Layout:
//   - GpuRecord: 36-byte POD matching the MSL packed_float3 trio
//     and GLSL Photon struct. position + wi + power, all vec3.
//   - GpuCell: 12-byte open-addressed hash-table slot. cellHash =
//     kEmptyCell marks unused. cellHash holds the 32-bit Teschner
//     hash of (cx, cy, cz); offset + count delimit a contiguous run
//     of photons in the records array that fall in this cell.
//
// buildGpuTable() walks the source Map's flat record array,
// re-buckets by 32-bit cell hash, sorts photons by cell, and writes
// the cell table with linear-probe open addressing. Table size is
// the next power of two above 2x the bucket count (load factor
// <=0.5; the kernel-side 8-step probe limit then has comfortable
// headroom for the rare case where collisions stack).
//
// Both Metal and OpenGL backends use this exact layout; the MSL +
// GLSL kernel-side density estimates also use the same 32-bit
// Teschner hash so host bucketing and GPU lookup agree byte-for-byte.

namespace Photon
{
    struct GpuRecord
    {
        float position[3];
        float wi[3];
        float power[3];
    };
    static_assert(sizeof(GpuRecord) == 36,
                  "Photon::GpuRecord must be 36 bytes to match MSL packed_float3 trio "
                  "and GLSL std430 vec3 trio layouts");

    struct GpuCell
    {
        uint32_t cellHash;  // kEmptyCell (0xFFFFFFFF) = unused slot
        uint32_t offset;    // index into records
        uint32_t count;     // photons in this cell
    };
    static_assert(sizeof(GpuCell) == 12,
                  "Photon::GpuCell must be 12 bytes to match MSL PCRPhotonCell "
                  "and GLSL std430 PhotonCell layouts");

    inline constexpr uint32_t kEmptyCell = 0xFFFFFFFFu;

    // Teschner 2003 spatial hash. The MSL photonCellHash and the GLSL
    // photonCellHash MUST compute identical hashes for identical (cx,
    // cy, cz) inputs - otherwise the kernel's lookup wouldn't find the
    // cells the host placed. Three coprime large primes one per axis;
    // xor mix is cheaper than the alternative add-and-multiply chain.
    inline uint32_t cellHash32(int cx, int cy, int cz)
    {
        return uint32_t(cx) * 73856093u
             ^ uint32_t(cy) * 19349663u
             ^ uint32_t(cz) * 83492791u;
    }

    struct GpuTable
    {
        std::vector<GpuRecord> records;
        std::vector<GpuCell>   cells;
        uint32_t               tableMask = 0;  // tableSize - 1
    };

    // Flatten a CPU Map into the GPU layout. Returns false (and clears
    // outputs) when the input map is empty; callers should treat that
    // as "photon mapping is effectively off for this render" and bind
    // dummy 1-element buffers in its place so kernel SSBO declarations
    // still resolve.
    inline bool buildGpuTable(const Map &map, GpuTable &out)
    {
        out.records.clear();
        out.cells.clear();
        out.tableMask = 0;

        const auto &records = map.records();
        if (records.empty()) return false;

        const float invR = 1.0f / map.radius();

        // First pass: re-bucket by 32-bit cell hash. The host Map keys
        // its private grid by 64-bit hash (it lives in an
        // std::unordered_map); the GPU side uses 32-bit (matching the
        // MSL + GLSL hash). The two may disagree on bucketing (two
        // distinct 64-bit hashes can fold to the same 32-bit hash),
        // which is fine - merging buckets only inflates per-cell
        // counts, not correctness.
        std::unordered_map<uint32_t, std::vector<uint32_t>> by32;
        by32.reserve(records.size() / 4);
        for (size_t i = 0; i < records.size(); i++)
        {
            const auto &r = records[i];
            int cx = (int)std::floor(r.position[0] * invR);
            int cy = (int)std::floor(r.position[1] * invR);
            int cz = (int)std::floor(r.position[2] * invR);
            uint32_t h = cellHash32(cx, cy, cz);
            by32[h].push_back((uint32_t)i);
        }

        // Round table size up to next power of two with load factor 0.5.
        uint32_t targetSlots = (uint32_t)by32.size() * 2u;
        uint32_t tableSize = 16u;
        while (tableSize < targetSlots) tableSize <<= 1;
        out.tableMask = tableSize - 1u;

        out.cells.assign(tableSize, GpuCell{kEmptyCell, 0u, 0u});
        out.records.reserve(records.size());

        // Insert each bucket: copy photons into the flat sorted-by-cell
        // records array; record (cellHash, offset, count) in the table
        // via linear probing.
        for (auto &kv : by32)
        {
            uint32_t h = kv.first;
            uint32_t slot = h & out.tableMask;
            // Open addressing. With load factor <=0.5 the expected
            // probe is ~1; bounded loop count guards against insert
            // pathology that shouldn't happen but would otherwise
            // blow the build sky-high.
            for (uint32_t step = 0; step < tableSize; step++)
            {
                if (out.cells[slot].cellHash == kEmptyCell) break;
                slot = (slot + 1u) & out.tableMask;
            }
            uint32_t startOffset = (uint32_t)out.records.size();
            for (uint32_t idx : kv.second)
            {
                const auto &r = records[idx];
                GpuRecord rec;
                rec.position[0] = r.position[0]; rec.position[1] = r.position[1]; rec.position[2] = r.position[2];
                rec.wi[0]       = r.wi[0];       rec.wi[1]       = r.wi[1];       rec.wi[2]       = r.wi[2];
                rec.power[0]    = r.power[0];    rec.power[1]    = r.power[1];    rec.power[2]    = r.power[2];
                out.records.push_back(rec);
            }
            out.cells[slot] = GpuCell{ h, startOffset, (uint32_t)kv.second.size() };
        }
        return true;
    }
}
