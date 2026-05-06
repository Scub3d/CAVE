// Search-mode simulation compute shader (Elongated Rhombic Dodecahedron shape).
//
// Each thread processes 8 cells packed as 4-bit nibbles in one uint32. Reads
// the rule encoded in `simulationParameters`, evaluates birth/survival/death
// for each cell against the rule, writes packed output.
//
// ERD specifics versus Cube:
//   - Two neighbor configurations indexed by cellCoordinates.y % 2.
//   - Face deltas: kErdFaceDeltas[configIndex][0..11]  (12 deltas per config, 2 configs).
//   - Edge deltas: kErdEdgeDeltas[0..3]                (4 deltas, configuration-agnostic).
//   - No corner deltas — the corner block is omitted entirely.
//   - IsCellViable returns true.
//
// Required defines (injected via SlangCompiler):
//   GRID_X_SIZE, GRID_Y_SIZE, GRID_Z_SIZE
//   MAX_RULE_BITS, WORKGROUP_SIZE

import elongatedRhombicDodecahedron;

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

groupshared uint workgroupCounts[4];

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

    if (!ElongatedRhombicDodecahedronFace::IsCellViable(uint3(neighborCoords), dims)) return false;

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
    int configIndex = cellCoords.y & 1;

    if ((neighborhoodFlags & FACE_NEIGHBORS_MASK) != 0u)
    {
        for (int i = 0; i < 12; i++)
            if (isAliveNeighbor(cellCoords + kErdFaceDeltas[configIndex][i], maxCellState, dims, wraps)) count++;
    }
    if ((neighborhoodFlags & EDGE_NEIGHBORS_MASK) != 0u)
    {
        for (int i = 0; i < 4; i++)
            if (isAliveNeighbor(cellCoords + kErdEdgeDeltas[i], maxCellState, dims, wraps)) count++;
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

    if (localId.x < 4u) workgroupCounts[localId.x] = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint bornCount     = 0u;
    uint diedCount     = 0u;
    uint survivedCount = 0u;
    uint dyingCount    = 0u;

    if (cellsActive)
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

        uint inPacked  = cellsIn[uintIndex];
        uint outPacked = 0u;

        for (uint nibble = 0u; nibble < CELLS_PER_UINT; nibble++)
        {
            uint cellIndex = uintIndex * CELLS_PER_UINT + nibble;
            if (cellIndex >= totalCells) break;

            uint xCoord = cellIndex % dims.x;
            uint yCoord = (cellIndex / dims.x) % dims.y;
            uint zCoord = (cellIndex / (dims.x * dims.y)) % dims.z;

            if (!ElongatedRhombicDodecahedronFace::IsCellViable(uint3(xCoord, yCoord, zCoord), dims))
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

    if (diedCount     > 0u) InterlockedAdd(workgroupCounts[0], diedCount);
    if (dyingCount    > 0u) InterlockedAdd(workgroupCounts[1], dyingCount);
    if (bornCount     > 0u) InterlockedAdd(workgroupCounts[2], bornCount);
    if (survivedCount > 0u) InterlockedAdd(workgroupCounts[3], survivedCount);
    GroupMemoryBarrierWithGroupSync();

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
