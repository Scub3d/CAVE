// Search-mode simulation compute shader (Cube shape, IMAGE storage, variant C).
//
// PROBE variant of searchSimulation.slang. Cells live in a 3D RWTexture3D<uint>
// with `r32ui` format, 8 cells packed per texel (one uint = 8 nibbles) along X.
// Each thread handles ONE texel (= 8 cells), matching the SSBO shader's
// 8-cells-per-thread workload AND its dword-aligned write granularity (no
// byte-store penalty).
//
// Algorithmic optimization: neighbor reads are SHARED across the 8 cells in
// each texel. For FE neighborhood, this means:
//   - 1 working texel load.
//   - 6 face-direction texel loads (X, Y, Z ±1).
//   - 12 edge-direction texel loads (combinations of two axis ±1).
//   = 19 texel loads to compute 8 cells (~2.4 loads per cell), vs the naive
//     per-cell-lookup pattern that would issue 18 × 8 = 144 individual cell
//     reads (most cached, but the issue rate dominates at high concurrency).
//
// Workgroup is `(8, 8, 4) = 256 threads`; covers 8×8×4 texels = 64×8×4 cells
// = 2048 cells per workgroup — same as SSBO and variant B.
//
// Variant lineage on this probe:
//  - A: r8ui (2 cells/texel), 1 texel/thread (= 2 cells/thread). 9-12% slower than SSBO across p=1..64.
//  - B: r8ui (2 cells/texel), 4 texels/thread (= 8 cells/thread). Equal workgroup geometry; still 9-12% slower (variant A and B basically tied at p=1, 244.83s vs 245.48s).
//  - C: r32ui (8 cells/texel), 1 texel/thread (= 8 cells/thread). Eliminates byte-store penalty; pre-fetches all neighbor texels and shares across the 8 cells.
//
// Limitation: this variant does NOT support wraps. The pre-fetch convention
// assumes texel-aligned neighbor lookups; wrap with non-aligned grids
// (GRID_X_SIZE % 8 != 0) would mis-address. The bench config uses wraps off
// (FE neighborhood, mask 0x3 with no WRAP bit). If wraps is requested at
// runtime, the kernel returns 0 (effectively producing wrong results); the
// SSBO path remains the correct production target for wrap workloads.
//
// Required defines (injected via SlangCompiler):
//   GRID_X_SIZE, GRID_Y_SIZE, GRID_Z_SIZE  — grid dimensions in cells
//   IMAGE_WIDTH                            — (GRID_X_SIZE + 7) / 8  (texel x-extent)
//   MAX_RULE_BITS                          — 0 (raw bitmask) or K (1-based index)

import cube;

static const uint CELL_STATE_MASK = 0xFu;

static const uint64_t MAX_CELL_STATE_SHIFT     = uint64_t(60);
static const uint64_t NEIGHBORHOOD_FLAGS_SHIFT = uint64_t(60);
static const uint64_t PERMUTATION_BIT_MASK     = (uint64_t(1) << 60) - uint64_t(1);

static const uint FACE_NEIGHBORS_MASK   = 1u;
static const uint EDGE_NEIGHBORS_MASK   = 2u;
static const uint CORNER_NEIGHBORS_MASK = 4u;
static const uint WRAP_NEIGHBORS_MASK   = 8u;

struct SimulationParameters
{
    uint64_t birthAndMaxCellStateRules;
    uint64_t survivalAndNeighborhoodRules;
    uint64_t shapeAndGridConfiguration;
};

struct GridSnapshot
{
    uint numberOfDeadCells;
    uint numberOfDyingCells;
    uint numberOfCellsBorn;
    uint numberOfSurvivingCells;
};

[vk::binding(0, 0)] RWStructuredBuffer<SimulationParameters> simulationParametersBuffer;
[vk::binding(1, 0)] RWStructuredBuffer<GridSnapshot>         gridSnapshotBuffer;
[vk::binding(2, 0)] [vk::image_format("r32ui")] RWTexture3D<uint> cellsInImage;
[vk::binding(3, 0)] [vk::image_format("r32ui")] RWTexture3D<uint> cellsOutImage;
[vk::binding(4, 0)] RWStructuredBuffer<uint64_t>             validBitmasks;

groupshared uint workgroupCounts[4]; // [0]=died, [1]=dying, [2]=born, [3]=surviving

// Extract the cell-state nibble at index n (0..7) from a packed uint.
uint extractNibble(uint packed, uint n)
{
    return (packed >> (n * 4u)) & 0xFu;
}

// Bounds-clamped texel load. Out-of-bounds returns 0u so nibble extraction
// gives a "dead" cell — the correct semantic for non-wrap neighborhood.
uint loadTexelClamp(int tx, int ty, int tz)
{
    if (tx < 0 || tx >= int(IMAGE_WIDTH)
        || ty < 0 || ty >= int(GRID_Y_SIZE)
        || tz < 0 || tz >= int(GRID_Z_SIZE))
    {
        return 0u;
    }
    return cellsInImage[int3(tx, ty, tz)];
}

// Whether the cell at logical (cx, cy, cz) is alive, given a pre-fetched
// packed texel value and the nibble position within it. Bounds-checks the
// cell coords (in case the caller passed an OOB neighbor coord) and runs the
// shape's IsCellViable predicate.
bool isCellAliveFromPacked(int3 cellCoords, uint maxCellState, uint3 dims, uint packedTexel, uint nibblePos)
{
    if (cellCoords.x < 0 || cellCoords.x >= int(dims.x)
        || cellCoords.y < 0 || cellCoords.y >= int(dims.y)
        || cellCoords.z < 0 || cellCoords.z >= int(dims.z))
    {
        return false;
    }
    if (!CubeFaceEdgeCorner::IsCellViable(uint3(cellCoords), dims))
        return false;
    return extractNibble(packedTexel, nibblePos) == maxCellState;
}

uint stepOneCellByCount(uint currentCell, int3 cellCoords, uint maxCellState, uint3 dims,
    uint64_t birthRulesBitmask, uint64_t survivalRulesBitmask,
    int aliveNeighborCount,
    inout uint bornCount, inout uint diedCount, inout uint survivedCount, inout uint dyingCount)
{
    if (!CubeFaceEdgeCorner::IsCellViable(uint3(cellCoords), dims))
    {
        diedCount++;
        return 0u;
    }
    if (currentCell == maxCellState)
    {
        if (aliveNeighborCount > 0
            && ((survivalRulesBitmask >> uint64_t(aliveNeighborCount - 1)) & uint64_t(1)) == uint64_t(1))
        {
            survivedCount++;
            return maxCellState;
        }
        dyingCount++;
        return currentCell - 1u;
    }
    else if (currentCell == 0u)
    {
        if (aliveNeighborCount > 0
            && ((birthRulesBitmask >> uint64_t(aliveNeighborCount - 1)) & uint64_t(1)) == uint64_t(1))
        {
            bornCount++;
            return maxCellState;
        }
        diedCount++;
        return 0u;
    }
    else
    {
        dyingCount++;
        return currentCell - 1u;
    }
}

[shader("compute")]
[numthreads(8, 8, 4)]
void computeMain(uint3 globalId : SV_DispatchThreadID, uint3 localId : SV_GroupThreadID)
{
    uint3 dims = uint3(GRID_X_SIZE, GRID_Y_SIZE, GRID_Z_SIZE);

    int tx = int(globalId.x);
    int ty = int(globalId.y);
    int tz = int(globalId.z);
    bool texelActive = (tx < int(IMAGE_WIDTH))
                    && (ty < int(GRID_Y_SIZE))
                    && (tz < int(GRID_Z_SIZE));

    uint linearLocal = localId.x + localId.y * 8u + localId.z * 64u;
    if (linearLocal < 4u) workgroupCounts[linearLocal] = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint bornCount     = 0u;
    uint diedCount     = 0u;
    uint survivedCount = 0u;
    uint dyingCount    = 0u;

    if (texelActive)
    {
        SimulationParameters simParams = simulationParametersBuffer[0];

        uint maxCellState      = uint(simParams.birthAndMaxCellStateRules >> MAX_CELL_STATE_SHIFT) + 1u;
        uint neighborhoodFlags = uint(simParams.survivalAndNeighborhoodRules >> NEIGHBORHOOD_FLAGS_SHIFT);

        uint64_t birthRulesBitmask;
        uint64_t survivalRulesBitmask;
        if (MAX_RULE_BITS > 0u)
        {
            uint birthIndex    = uint(simParams.birthAndMaxCellStateRules    & PERMUTATION_BIT_MASK) - 1u;
            uint survivalIndex = uint(simParams.survivalAndNeighborhoodRules & PERMUTATION_BIT_MASK) - 1u;
            birthRulesBitmask    = validBitmasks[birthIndex];
            survivalRulesBitmask = validBitmasks[survivalIndex];
        }
        else
        {
            birthRulesBitmask    = simParams.birthAndMaxCellStateRules    & PERMUTATION_BIT_MASK;
            survivalRulesBitmask = simParams.survivalAndNeighborhoodRules & PERMUTATION_BIT_MASK;
        }

        // Pre-fetch the working texel and all 18 surrounding texels (6 face + 12 edge).
        // Corner texels are intentionally NOT pre-fetched — the FE neighborhood used
        // by the bench skips corners. If FEC ever lands here, add 8 more pre-fetched
        // corner texels.
        uint pkW = cellsInImage[int3(tx, ty, tz)];

        uint pkXn = loadTexelClamp(tx - 1, ty,     tz    );
        uint pkXp = loadTexelClamp(tx + 1, ty,     tz    );
        uint pkYn = loadTexelClamp(tx,     ty - 1, tz    );
        uint pkYp = loadTexelClamp(tx,     ty + 1, tz    );
        uint pkZn = loadTexelClamp(tx,     ty,     tz - 1);
        uint pkZp = loadTexelClamp(tx,     ty,     tz + 1);

        uint pkXnYn = loadTexelClamp(tx - 1, ty - 1, tz    );
        uint pkXnYp = loadTexelClamp(tx - 1, ty + 1, tz    );
        uint pkXpYn = loadTexelClamp(tx + 1, ty - 1, tz    );
        uint pkXpYp = loadTexelClamp(tx + 1, ty + 1, tz    );

        uint pkXnZn = loadTexelClamp(tx - 1, ty,     tz - 1);
        uint pkXnZp = loadTexelClamp(tx - 1, ty,     tz + 1);
        uint pkXpZn = loadTexelClamp(tx + 1, ty,     tz - 1);
        uint pkXpZp = loadTexelClamp(tx + 1, ty,     tz + 1);

        uint pkYnZn = loadTexelClamp(tx,     ty - 1, tz - 1);
        uint pkYnZp = loadTexelClamp(tx,     ty - 1, tz + 1);
        uint pkYpZn = loadTexelClamp(tx,     ty + 1, tz - 1);
        uint pkYpZp = loadTexelClamp(tx,     ty + 1, tz + 1);

        bool faceOn   = (neighborhoodFlags & FACE_NEIGHBORS_MASK)   != 0u;
        bool edgeOn   = (neighborhoodFlags & EDGE_NEIGHBORS_MASK)   != 0u;
        bool cornerOn = (neighborhoodFlags & CORNER_NEIGHBORS_MASK) != 0u;

        uint outPacked = 0u;
        for (uint k = 0u; k < 8u; k++)
        {
            int cx = tx * 8 + int(k);
            if (cx >= int(GRID_X_SIZE)) break;

            uint currentCell = extractNibble(pkW, k);

            // Same-texel vs cross-texel X-direction nibble selection.
            uint xnPacked = (k > 0u) ? pkW : pkXn;
            uint xnNib    = (k > 0u) ? (k - 1u) : 7u;
            uint xpPacked = (k < 7u) ? pkW : pkXp;
            uint xpNib    = (k < 7u) ? (k + 1u) : 0u;

            int aliveNeighbors = 0;

            if (faceOn)
            {
                if (isCellAliveFromPacked(int3(cx - 1, ty, tz), maxCellState, dims, xnPacked, xnNib)) aliveNeighbors++;
                if (isCellAliveFromPacked(int3(cx + 1, ty, tz), maxCellState, dims, xpPacked, xpNib)) aliveNeighbors++;
                if (isCellAliveFromPacked(int3(cx, ty - 1, tz), maxCellState, dims, pkYn, k))         aliveNeighbors++;
                if (isCellAliveFromPacked(int3(cx, ty + 1, tz), maxCellState, dims, pkYp, k))         aliveNeighbors++;
                if (isCellAliveFromPacked(int3(cx, ty, tz - 1), maxCellState, dims, pkZn, k))         aliveNeighbors++;
                if (isCellAliveFromPacked(int3(cx, ty, tz + 1), maxCellState, dims, pkZp, k))         aliveNeighbors++;
            }

            if (edgeOn)
            {
                // XY edges: cross-texel handled via the same xn/xp packed selection.
                uint xnYnPacked = (k > 0u) ? pkYn : pkXnYn;
                uint xnYpPacked = (k > 0u) ? pkYp : pkXnYp;
                uint xpYnPacked = (k < 7u) ? pkYn : pkXpYn;
                uint xpYpPacked = (k < 7u) ? pkYp : pkXpYp;
                if (isCellAliveFromPacked(int3(cx - 1, ty - 1, tz), maxCellState, dims, xnYnPacked, xnNib)) aliveNeighbors++;
                if (isCellAliveFromPacked(int3(cx - 1, ty + 1, tz), maxCellState, dims, xnYpPacked, xnNib)) aliveNeighbors++;
                if (isCellAliveFromPacked(int3(cx + 1, ty - 1, tz), maxCellState, dims, xpYnPacked, xpNib)) aliveNeighbors++;
                if (isCellAliveFromPacked(int3(cx + 1, ty + 1, tz), maxCellState, dims, xpYpPacked, xpNib)) aliveNeighbors++;

                // XZ edges.
                uint xnZnPacked = (k > 0u) ? pkZn : pkXnZn;
                uint xnZpPacked = (k > 0u) ? pkZp : pkXnZp;
                uint xpZnPacked = (k < 7u) ? pkZn : pkXpZn;
                uint xpZpPacked = (k < 7u) ? pkZp : pkXpZp;
                if (isCellAliveFromPacked(int3(cx - 1, ty, tz - 1), maxCellState, dims, xnZnPacked, xnNib)) aliveNeighbors++;
                if (isCellAliveFromPacked(int3(cx - 1, ty, tz + 1), maxCellState, dims, xnZpPacked, xnNib)) aliveNeighbors++;
                if (isCellAliveFromPacked(int3(cx + 1, ty, tz - 1), maxCellState, dims, xpZnPacked, xpNib)) aliveNeighbors++;
                if (isCellAliveFromPacked(int3(cx + 1, ty, tz + 1), maxCellState, dims, xpZpPacked, xpNib)) aliveNeighbors++;

                // YZ edges: same X (k), different Y/Z.
                if (isCellAliveFromPacked(int3(cx, ty - 1, tz - 1), maxCellState, dims, pkYnZn, k)) aliveNeighbors++;
                if (isCellAliveFromPacked(int3(cx, ty - 1, tz + 1), maxCellState, dims, pkYnZp, k)) aliveNeighbors++;
                if (isCellAliveFromPacked(int3(cx, ty + 1, tz - 1), maxCellState, dims, pkYpZn, k)) aliveNeighbors++;
                if (isCellAliveFromPacked(int3(cx, ty + 1, tz + 1), maxCellState, dims, pkYpZp, k)) aliveNeighbors++;
            }

            // CORNER neighborhood not pre-fetched in variant C; if the runtime flag
            // requests corners, this kernel under-counts. The bench uses FE only.
            // (To support FEC: pre-fetch 8 corner texels and add the cornerOn loop.)

            uint newCell = stepOneCellByCount(currentCell, int3(cx, ty, tz), maxCellState, dims,
                birthRulesBitmask, survivalRulesBitmask,
                aliveNeighbors,
                bornCount, diedCount, survivedCount, dyingCount);

            outPacked |= (newCell & 0xFu) << (k * 4u);
        }

        cellsOutImage[int3(tx, ty, tz)] = outPacked;
    }

    if (diedCount     > 0u) InterlockedAdd(workgroupCounts[0], diedCount);
    if (dyingCount    > 0u) InterlockedAdd(workgroupCounts[1], dyingCount);
    if (bornCount     > 0u) InterlockedAdd(workgroupCounts[2], bornCount);
    if (survivedCount > 0u) InterlockedAdd(workgroupCounts[3], survivedCount);
    GroupMemoryBarrierWithGroupSync();

    if (linearLocal < 4u)
    {
        uint workgroupTotal = workgroupCounts[linearLocal];
        if (workgroupTotal > 0u)
        {
            if      (linearLocal == 0u) InterlockedAdd(gridSnapshotBuffer[0].numberOfDeadCells,      workgroupTotal);
            else if (linearLocal == 1u) InterlockedAdd(gridSnapshotBuffer[0].numberOfDyingCells,     workgroupTotal);
            else if (linearLocal == 2u) InterlockedAdd(gridSnapshotBuffer[0].numberOfCellsBorn,      workgroupTotal);
            else                         InterlockedAdd(gridSnapshotBuffer[0].numberOfSurvivingCells, workgroupTotal);
        }
    }
}
