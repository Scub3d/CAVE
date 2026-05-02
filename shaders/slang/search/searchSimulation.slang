// Search-mode simulation compute shader (Cube shape only).
//
// Each thread processes 8 cells packed as 4-bit nibbles in one uint32. Reads
// the rule encoded in `simulationParameters`, evaluates birth/survival/death
// for each cell against the rule, writes packed output.
//
// Per-tick stats are accumulated via workgroup-shared-memory atomic reduction
// (4 conditional atomicAdds per workgroup → global gridSnapshot, vs the naive
// per-thread global atomic which is 256× more global atomic traffic).
//
// Required defines (injected via SlangCompiler):
//   GRID_X_SIZE, GRID_Y_SIZE, GRID_Z_SIZE   — grid dimensions baked in
//   MAX_RULE_BITS                           — 0 (raw bitmask) or K (1-based index into validBitmasks)
//   WORKGROUP_SIZE                          — threads per workgroup (32, 64, 128, 256, 512)
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

    uint linearIndex = uint(neighborCoords.x)
                     + uint(neighborCoords.y) * dims.x
                     + uint(neighborCoords.z) * dims.x * dims.y;
    uint uintIndex = linearIndex / CELLS_PER_UINT;
    uint nibble    = linearIndex % CELLS_PER_UINT;
    uint state     = (cellsIn[uintIndex] >> (nibble * 4u)) & CELL_STATE_MASK;
    return state == maxCellState;
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

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void computeMain(uint3 globalId : SV_DispatchThreadID, uint3 localId : SV_GroupThreadID)
{
    uint3 dims      = uint3(GRID_X_SIZE, GRID_Y_SIZE, GRID_Z_SIZE);
    uint totalCells = dims.x * dims.y * dims.z;
    uint totalUints = (totalCells + CELLS_PER_UINT - 1u) / CELLS_PER_UINT;
    uint uintIndex  = globalId.x;
    bool cellsActive = (uintIndex < totalUints);

    // ALL threads must reach the barriers, including those past totalUints.
    if (localId.x < 4u) workgroupCounts[localId.x] = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint bornCount     = 0u;
    uint diedCount     = 0u;
    uint survivedCount = 0u;
    uint dyingCount    = 0u;

    if (cellsActive)
    {
        SimulationParameters simParams = simulationParametersBuffer[0];

        // maxCellState uses shift-by-1 encoding: actualValue = storedBits + 1.
        uint maxCellState      = uint(simParams.birthAndMaxCellStateRules >> MAX_CELL_STATE_SHIFT) + 1u;
        uint neighborhoodFlags = uint(simParams.survivalAndNeighborhoodRules >> NEIGHBORHOOD_FLAGS_SHIFT);
        bool wraps             = (neighborhoodFlags & WRAP_NEIGHBORS_MASK) != 0u;

        // Lower 60 bits: raw bitmask (K=0) or 1-based index into validBitmasks (K>0).
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

        uint inPacked  = cellsIn[uintIndex];
        uint outPacked = 0u;

        // Process up to 8 cells that share this uint. Each cell is independent —
        // no thread-vs-thread races since each thread owns its own uint end-to-end.
        for (uint nibble = 0u; nibble < CELLS_PER_UINT; nibble++)
        {
            uint cellIndex = uintIndex * CELLS_PER_UINT + nibble;
            if (cellIndex >= totalCells) break;

            uint xCoord = cellIndex % dims.x;
            uint yCoord = (cellIndex / dims.x) % dims.y;
            uint zCoord = (cellIndex / (dims.x * dims.y)) % dims.z;

            if (!CubeFaceEdgeCorner::IsCellViable(uint3(xCoord, yCoord, zCoord), dims))
            {
                diedCount++;
                continue;
            }

            uint currentCell = (inPacked >> (nibble * 4u)) & CELL_STATE_MASK;
            int aliveNeighborCount = countAliveNeighbors(
                int3(xCoord, yCoord, zCoord), maxCellState, dims, wraps, neighborhoodFlags);

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

        cellsOut[uintIndex] = outPacked;
    }

    // Reduce per-thread counts into workgroup shared accumulators.
    if (diedCount     > 0u) InterlockedAdd(workgroupCounts[0], diedCount);
    if (dyingCount    > 0u) InterlockedAdd(workgroupCounts[1], dyingCount);
    if (bornCount     > 0u) InterlockedAdd(workgroupCounts[2], bornCount);
    if (survivedCount > 0u) InterlockedAdd(workgroupCounts[3], survivedCount);
    GroupMemoryBarrierWithGroupSync();

    // First 4 threads each commit one workgroup total to the global snapshot.
    if (localId.x < 4u)
    {
        uint workgroupTotal = workgroupCounts[localId.x];
        if (workgroupTotal > 0u)
        {
            if      (localId.x == 0u) InterlockedAdd(gridSnapshotBuffer[0].numberOfDeadCells,      workgroupTotal);
            else if (localId.x == 1u) InterlockedAdd(gridSnapshotBuffer[0].numberOfDyingCells,     workgroupTotal);
            else if (localId.x == 2u) InterlockedAdd(gridSnapshotBuffer[0].numberOfCellsBorn,      workgroupTotal);
            else                       InterlockedAdd(gridSnapshotBuffer[0].numberOfSurvivingCells, workgroupTotal);
        }
    }
}
