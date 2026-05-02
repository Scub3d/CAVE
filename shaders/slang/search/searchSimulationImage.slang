// Search-mode simulation compute shader (Cube shape, IMAGE storage).
//
// PROBE variant of searchSimulation.slang. Cells live in a 3D RWTexture3D<uint>
// with `r8ui` format, 2 cells packed per texel (low/high nibble) along X, instead
// of the 1D RWStructuredBuffer<uint> with 8 cells per uint used by the production
// SSBO path. The intent is to exploit the GPU's hardware 3D texture cache + tiled
// memory layout for FE/FEC neighborhood lookups, where Y/Z neighbors are far in
// memory under linear addressing but should be cache-warm under tiled addressing.
//
// Each thread handles ONE texel (= 2 cells). Workgroup is 3D — `(8, 8, 4) = 256
// threads` — so a workgroup covers an 8×8×4 texel block (~16×8×4 cells), giving
// the texture cache a tight spatial working set.
//
// Per-tick stats: same workgroup-shared-memory atomic reduction as the SSBO
// shader, just with leader thread picking via `localId.x == 0 && localId.y == 0
// && localId.z < 4` instead of `localId.x < 4`.
//
// Required defines (injected via SlangCompiler):
//   GRID_X_SIZE, GRID_Y_SIZE, GRID_Z_SIZE  — grid dimensions in cells
//   IMAGE_WIDTH                            — (GRID_X_SIZE + 1) / 2  (texel x-extent)
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
[vk::binding(2, 0)] [vk::image_format("r8ui")] RWTexture3D<uint> cellsInImage;
[vk::binding(3, 0)] [vk::image_format("r8ui")] RWTexture3D<uint> cellsOutImage;
[vk::binding(4, 0)] RWStructuredBuffer<uint64_t>             validBitmasks;

groupshared uint workgroupCounts[4]; // [0]=died, [1]=dying, [2]=born, [3]=surviving

// Read a single cell's 4-bit state at logical cell coords. Same packing as
// rendering's cellStateUtils.slang::readCellState — texel x = cellX/2, low
// nibble for even X, high nibble for odd X.
uint readCellAt(int3 cellCoord)
{
    int3 texelCoord = int3(cellCoord.x >> 1, cellCoord.y, cellCoord.z);
    uint packed = cellsInImage[texelCoord];
    return ((cellCoord.x & 1) == 0) ? (packed & 0xFu) : ((packed >> 4) & 0xFu);
}

bool isAliveNeighbor(int3 neighborCoords, uint maxCellState, uint3 dims, bool wraps)
{
    if (wraps)
    {
        neighborCoords.x = (neighborCoords.x + int(dims.x)) % int(dims.x);
        neighborCoords.y = (neighborCoords.y + int(dims.y)) % int(dims.y);
        neighborCoords.z = (neighborCoords.z + int(dims.z)) % int(dims.z);
    }
    else
    {
        if (neighborCoords.x < 0 || neighborCoords.x >= int(dims.x)) return false;
        if (neighborCoords.y < 0 || neighborCoords.y >= int(dims.y)) return false;
        if (neighborCoords.z < 0 || neighborCoords.z >= int(dims.z)) return false;
    }

    if (!CubeFaceEdgeCorner::IsCellViable(uint3(neighborCoords), dims)) return false;

    return readCellAt(neighborCoords) == maxCellState;
}

int countAliveNeighbors(int3 cellCoords, uint maxCellState, uint3 dims, bool wraps, uint neighborhoodFlags)
{
    int count = 0;
    if ((neighborhoodFlags & FACE_NEIGHBORS_MASK) != 0u)
    {
        for (int i = 0; i < 6; i++)
            if (isAliveNeighbor(cellCoords + kFaceDeltas[i], maxCellState, dims, wraps)) count++;
    }
    if ((neighborhoodFlags & EDGE_NEIGHBORS_MASK) != 0u)
    {
        for (int i = 0; i < 12; i++)
            if (isAliveNeighbor(cellCoords + kEdgeDeltas[i], maxCellState, dims, wraps)) count++;
    }
    if ((neighborhoodFlags & CORNER_NEIGHBORS_MASK) != 0u)
    {
        for (int i = 0; i < 8; i++)
            if (isAliveNeighbor(cellCoords + kCornerDeltas[i], maxCellState, dims, wraps)) count++;
    }
    return count;
}

uint stepOneCell(uint currentCell, int3 cellCoords, uint maxCellState, uint3 dims, bool wraps,
    uint neighborhoodFlags, uint64_t birthRulesBitmask, uint64_t survivalRulesBitmask,
    inout uint bornCount, inout uint diedCount, inout uint survivedCount, inout uint dyingCount)
{
    if (!CubeFaceEdgeCorner::IsCellViable(uint3(cellCoords), dims))
    {
        diedCount++;
        return 0u;
    }

    int aliveNeighborCount = countAliveNeighbors(cellCoords, maxCellState, dims, wraps, neighborhoodFlags);

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

    int3 texelCoord = int3(globalId);
    bool texelActive = (texelCoord.x < int(IMAGE_WIDTH))
                    && (texelCoord.y < int(GRID_Y_SIZE))
                    && (texelCoord.z < int(GRID_Z_SIZE));

    // ALL threads must reach the barriers — including those off the dispatch grid edge.
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
        bool wraps             = (neighborhoodFlags & WRAP_NEIGHBORS_MASK) != 0u;

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

        uint inPacked = cellsInImage[texelCoord];
        uint cellEvenX = inPacked & 0xFu;
        uint cellOddX  = (inPacked >> 4) & 0xFu;

        // Two cells in this texel: even X and odd X.
        int evenXCellX = texelCoord.x * 2;
        int oddXCellX  = evenXCellX + 1;

        uint newEvenX = stepOneCell(cellEvenX, int3(evenXCellX, texelCoord.y, texelCoord.z),
            maxCellState, dims, wraps, neighborhoodFlags, birthRulesBitmask, survivalRulesBitmask,
            bornCount, diedCount, survivedCount, dyingCount);

        // Edge case: at odd grid widths, the high nibble of the last X texel is unused.
        uint newOddX = 0u;
        if (oddXCellX < int(dims.x))
        {
            newOddX = stepOneCell(cellOddX, int3(oddXCellX, texelCoord.y, texelCoord.z),
                maxCellState, dims, wraps, neighborhoodFlags, birthRulesBitmask, survivalRulesBitmask,
                bornCount, diedCount, survivedCount, dyingCount);
        }

        uint outPacked = (newEvenX & 0xFu) | ((newOddX & 0xFu) << 4);
        cellsOutImage[texelCoord] = outPacked;
    }

    // Reduce per-thread counts into workgroup shared accumulators.
    if (diedCount     > 0u) InterlockedAdd(workgroupCounts[0], diedCount);
    if (dyingCount    > 0u) InterlockedAdd(workgroupCounts[1], dyingCount);
    if (bornCount     > 0u) InterlockedAdd(workgroupCounts[2], bornCount);
    if (survivedCount > 0u) InterlockedAdd(workgroupCounts[3], survivedCount);
    GroupMemoryBarrierWithGroupSync();

    // First 4 threads (linear local index 0..3) commit one workgroup total each.
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
