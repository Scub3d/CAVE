// Search-mode simulation compute shader (Cube shape only).
//
// Production shader: 3D dispatch (8, 8, 4) = 256 threads, each thread owning
// ONE uint = 8 packed cells along X. Workgroup covers 8×8×4 uints = 64×8×4
// cells = 2048 cells. Stats reduction uses subgroup wave intrinsics.
//
// Optimizations vs the prior 1D-dispatch shader (split-tested 2026-05-02):
//
// 1. 3D DISPATCH. Each workgroup is now a spatially-coherent 3D block of
//    uints (rather than a long 1D run), giving better L1/L2 hit rates on
//    Y/Z neighbor reads. Speedup: -25 to -37% across p=4..256 vs 1D dispatch
//    on cube 49^3 FE K=2 GPU0-only.
//
// 2. WAVE-INTRINSIC stats reduction. Per-thread bornCount/diedCount/etc are
//    summed first via WaveActiveSum within each 32-lane warp; only the
//    first lane of each warp commits one InterlockedAdd-to-groupshared per
//    counter. With 8 warps × 4 counters = 32 atomicAdds-to-groupshared per
//    workgroup (vs 256 × 4 = 1024 in the per-thread variant). Speedup over
//    pure 3D dispatch: -10 to -30% additional across p=4..128.
//
// Variants tried and rejected (see project_optimization_split_test memory note):
//   - Cooperative groupshared TILE LOADING: helped at p=8-32 (-12% vs 3D)
//     but regressed at p=64+ (+11 to +57%). When combined with wave intrinsics
//     it became a net loss everywhere (+5-9% vs wave-only). Removed.
//
// LIMITATION: this variant does NOT support wraps. The dispatch math assumes
// uint-aligned access; wrap with non-aligned grids (gridX % 8 != 0) would
// require careful nibble re-mapping. Production search workloads use
// wraps=off — recover the prior 1D-dispatch shader from git history if needed.
//
// Required defines (injected via SlangCompiler):
//   GRID_X_SIZE, GRID_Y_SIZE, GRID_Z_SIZE   — grid dimensions in cells
//   MAX_RULE_BITS                           — 0 (raw bitmask) or K (1-based index into validBitmasks)
//
// IMPORTANT — Slang 64-bit literal gotcha: `0xF000000000000000UL` truncates to
// uint32 (=0). Always build uint64 constants from bit shifts or explicit casts.

import cube;

static const uint CELLS_PER_UINT = 8u;
static const uint CELL_STATE_MASK = 0xFu;

static const uint64_t MAX_CELL_STATE_SHIFT     = uint64_t(60);
static const uint64_t NEIGHBORHOOD_FLAGS_SHIFT = uint64_t(60);
static const uint64_t PERMUTATION_BIT_MASK     = (uint64_t(1) << 60) - uint64_t(1);

static const uint FACE_NEIGHBORS_MASK   = 1u;
static const uint EDGE_NEIGHBORS_MASK   = 2u;
static const uint CORNER_NEIGHBORS_MASK = 4u;
static const uint WRAP_NEIGHBORS_MASK   = 8u;

static const uint UINTS_PER_ROW = (GRID_X_SIZE + CELLS_PER_UINT - 1u) / CELLS_PER_UINT;
static const uint UINTS_PER_SLICE = UINTS_PER_ROW * GRID_Y_SIZE;

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
[vk::binding(2, 0)] RWStructuredBuffer<uint>                 cellsIn;
[vk::binding(3, 0)] RWStructuredBuffer<uint>                 cellsOut;
[vk::binding(4, 0)] RWStructuredBuffer<uint64_t>             validBitmasks;

groupshared uint workgroupCounts[4]; // [0]=died, [1]=dying, [2]=born, [3]=surviving

bool isAliveNeighbor(int3 c, uint maxCellState, uint3 dims)
{
    if (c.x < 0 || c.x >= int(dims.x) || c.y < 0 || c.y >= int(dims.y) || c.z < 0 || c.z >= int(dims.z))
        return false;
    if (!CubeFaceEdgeCorner::IsCellViable(uint3(c), dims)) return false;
    uint linearIndex = uint(c.x) + uint(c.y) * dims.x + uint(c.z) * dims.x * dims.y;
    uint uintIndex = linearIndex / CELLS_PER_UINT;
    uint nibble    = linearIndex % CELLS_PER_UINT;
    uint state     = (cellsIn[uintIndex] >> (nibble * 4u)) & CELL_STATE_MASK;
    return state == maxCellState;
}

int countAliveNeighbors(int3 cellCoords, uint maxCellState, uint3 dims, uint neighborhoodFlags)
{
    int count = 0;
    if ((neighborhoodFlags & FACE_NEIGHBORS_MASK) != 0u)
    {
        for (int i = 0; i < 6; i++)
            if (isAliveNeighbor(cellCoords + kFaceDeltas[i], maxCellState, dims)) count++;
    }
    if ((neighborhoodFlags & EDGE_NEIGHBORS_MASK) != 0u)
    {
        for (int i = 0; i < 12; i++)
            if (isAliveNeighbor(cellCoords + kEdgeDeltas[i], maxCellState, dims)) count++;
    }
    if ((neighborhoodFlags & CORNER_NEIGHBORS_MASK) != 0u)
    {
        for (int i = 0; i < 8; i++)
            if (isAliveNeighbor(cellCoords + kCornerDeltas[i], maxCellState, dims)) count++;
    }
    return count;
}

[shader("compute")]
[numthreads(8, 8, 4)]
void computeMain(uint3 globalId : SV_DispatchThreadID, uint3 localId : SV_GroupThreadID)
{
    uint3 dims = uint3(GRID_X_SIZE, GRID_Y_SIZE, GRID_Z_SIZE);

    int uintX = int(globalId.x);
    int uintY = int(globalId.y);
    int uintZ = int(globalId.z);
    bool uintActive = (uintX < int(UINTS_PER_ROW)) && (uintY < int(GRID_Y_SIZE)) && (uintZ < int(GRID_Z_SIZE));

    uint linearLocal = localId.x + localId.y * 8u + localId.z * 64u;
    if (linearLocal < 4u) workgroupCounts[linearLocal] = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint bornCount     = 0u;
    uint diedCount     = 0u;
    uint survivedCount = 0u;
    uint dyingCount    = 0u;

    if (uintActive)
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

        uint uintIdx = uint(uintX) + uint(uintY) * UINTS_PER_ROW + uint(uintZ) * UINTS_PER_SLICE;
        uint inPacked = cellsIn[uintIdx];
        uint outPacked = 0u;

        for (uint nibble = 0u; nibble < CELLS_PER_UINT; nibble++)
        {
            int cx = uintX * int(CELLS_PER_UINT) + int(nibble);
            if (cx >= int(dims.x)) break;

            if (!CubeFaceEdgeCorner::IsCellViable(uint3(uint(cx), uint(uintY), uint(uintZ)), dims))
            {
                diedCount++;
                continue;
            }

            uint currentCell = (inPacked >> (nibble * 4u)) & CELL_STATE_MASK;
            int aliveNeighborCount = countAliveNeighbors(
                int3(cx, uintY, uintZ), maxCellState, dims, neighborhoodFlags);

            uint newState = 0u;
            if (currentCell == maxCellState)
            {
                if (aliveNeighborCount > 0
                    && ((survivalRulesBitmask >> uint64_t(aliveNeighborCount - 1)) & uint64_t(1)) == uint64_t(1))
                {
                    newState = maxCellState;
                    survivedCount++;
                }
                else
                {
                    newState = currentCell - 1u;
                    dyingCount++;
                }
            }
            else if (currentCell == 0u)
            {
                if (aliveNeighborCount > 0
                    && ((birthRulesBitmask >> uint64_t(aliveNeighborCount - 1)) & uint64_t(1)) == uint64_t(1))
                {
                    newState = maxCellState;
                    bornCount++;
                }
                else
                {
                    newState = 0u;
                    diedCount++;
                }
            }
            else
            {
                newState = currentCell - 1u;
                dyingCount++;
            }

            outPacked |= (newState & CELL_STATE_MASK) << (nibble * 4u);
        }

        cellsOut[uintIdx] = outPacked;
    }

    // Wave-intrinsic stats reduction (32-thread warps): each warp sums its
    // counts via WaveActiveSum, then the first lane of each warp commits one
    // InterlockedAdd-to-groupshared per counter. 8 warps × 4 counters = 32
    // atomicAdds-to-groupshared per workgroup (vs 256 × 4 = 1024 in the prior
    // per-thread variant).
    uint warpDied     = WaveActiveSum(diedCount);
    uint warpDying    = WaveActiveSum(dyingCount);
    uint warpBorn     = WaveActiveSum(bornCount);
    uint warpSurvived = WaveActiveSum(survivedCount);
    if (WaveIsFirstLane())
    {
        if (warpDied     > 0u) InterlockedAdd(workgroupCounts[0], warpDied);
        if (warpDying    > 0u) InterlockedAdd(workgroupCounts[1], warpDying);
        if (warpBorn     > 0u) InterlockedAdd(workgroupCounts[2], warpBorn);
        if (warpSurvived > 0u) InterlockedAdd(workgroupCounts[3], warpSurvived);
    }
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
