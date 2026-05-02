// =============================================================================
// SEARCH-MODE SIMULATION SHADER TEMPLATE — copy this file when adding a new shape
// =============================================================================
//
// This file is a starting-point template, NOT loaded by the runtime. The C++
// dispatch in `searchShaderGenerator.cpp::GenerateSimulationShader` matches on
// `shape.GetName()` and compiles a per-shape file (e.g. `searchSimulation.slang`
// for Cube, `searchSimulationERD.slang` for Elongated Rhombic Dodecahedron).
//
// As of this template, the live shaders are:
//   - shaders/slang/searchSimulation.slang     (Cube)
//   - shaders/slang/searchSimulationERD.slang  (Elongated Rhombic Dodecahedron)
//
// To add a third shape (call it `Foo`):
//
// 1. Implement the shape's neighbor topology in `shaders/slang/foo.slang`,
//    following `cube.slang` / `elongatedDodecahedron.slang` as reference.
//    Define module-scope `static const int3 kFaceDeltas[...]`,
//    `static const int3 kEdgeDeltas[...]`, `static const int3 kCornerDeltas[...]`
//    plus a `struct FooFace : IShape` (and FooFaceEdge / FooFaceEdgeCorner if
//    the shape supports those subsets) with `IsCellViable`.
//
// 2. Copy this file to `shaders/slang/searchSimulationFoo.slang`.
//
// 3. Replace `import cube;` with `import foo;`.
//
// 4. Edit the SHAPE-SPECIFIC sections (search the file for "SHAPE-SPECIFIC"):
//      a. CONFIGURATION COUNT — Cube has 1 configuration (face deltas don't vary
//         by cell coordinate). ERD has 2 (alternating Y rows). Some hypothetical
//         shape might have more.
//      b. CONFIGURATION INDEX — the expression that picks which configuration
//         applies to the current cell. Cube returns 0 unconditionally; ERD
//         returns `cellCoords.y & 1`. Should match what your shape's
//         `IShape::GetNeighborConfigurationIndex` returns.
//      c. DELTA COUNTS — face count (6 for Cube, 12 for ERD); edge count
//         (12 for Cube, 4 for ERD); corner count (8 for Cube, 0 for ERD).
//      d. DELTA ARRAY ACCESS — `kFaceDeltas[i]` for shapes with one config,
//         `kFaceDeltas[configIndex][i]` for shapes with multiple. Same pattern
//         for edges and corners.
//      e. IS-VIABLE CALL SITE — replace `CubeFaceEdgeCorner::IsCellViable(...)`
//         with `FooFaceEdgeCorner::IsCellViable(...)` (or whichever IShape
//         struct your shape exposes — pick one that returns the broadest
//         viability check; Cube's is `return true`).
//      f. EMPTY NEIGHBORHOOD BLOCKS — if your shape has zero deltas in some
//         category (e.g. ERD has no corner deltas), DELETE that block entirely
//         rather than looping zero times.
//
// 5. In `searchShaderGenerator.cpp::GenerateSimulationShader`, add a dispatch
//    branch:
//        if (shape.GetName() == "Foo") {
//            _shaderModule = _slangCompiler.Compile(_deviceContext,
//                "shaders/slang", "searchSimulationFoo", "computeMain", defines);
//            return;
//        }
//
// 6. Add a regression test pair (one with neighborhood F, one with FE) at small
//    grid + K=2. Capture the GLSL-baseline JSONL aggregates from a build that
//    still has the legacy fallback BEFORE you flip the dispatch over to Slang
//    (or, if no GLSL baseline exists for the new shape, hand-verify a few rules
//    by inspection). After flipping to Slang, the JSONLs must diff to zero.
//
// 7. Common gotchas (learned the hard way 2026-05-01 during the Cube and ERD
//    migrations):
//      - Slang's `UL` literal suffix means uint32, NOT uint64. The expression
//        `0xF000000000000000UL` silently truncates to 0. Always build 64-bit
//        constants from explicit casts: `uint64_t(N) << SHIFT` or `uint64_t(0xN)`.
//      - GLSL's `uvec3` does not exist in Slang — use `uint3`. (Same for
//        ivec3 → int3, vec3 → float3, etc.)
//      - The `simulationParameters` push-constant-equivalent is bound at
//        binding 0 of set 0; do not change its layout — the C++ side writes
//        a fixed 24-byte SimulationParameters struct.
//      - The shape struct used for `IsCellViable` must match the runtime mask:
//        viability is a *shape* property and shouldn't change with the
//        runtime-selected F/E/C subset, so picking any of FooFace /
//        FooFaceEdge / FooFaceEdgeCorner gives the same answer.
//      - One descriptor set is bound per slot — the dispatch loop in C++
//        `searchSystem.cpp::RecordCommandBuffer` issues N separate dispatches
//        (one per slot, with the slot's per-grid descriptor set bound). The
//        shader is plain single-buffer access (`cellsIn[i]`, no `[slotIndex]`).
//        We tried collapsing into a 2D dispatch with descriptor arrays — it
//        regressed perf 11-35% across configs because dynamic descriptor-array
//        indexing inserts an extra indirect load per access. Don't redo it
//        without a way to hoist the indirection out of the inner loop.
//
// =============================================================================

// === SHAPE-SPECIFIC: import the shape's own .slang module here ===
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

    // === SHAPE-SPECIFIC: IsCellViable — replace `CubeFaceEdgeCorner` with your
    // shape's broadest IShape struct. Most shapes return `true` from IsCellViable;
    // shapes carved from a sphere or other irregular volume override it.
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

    // === SHAPE-SPECIFIC: configuration index — for shapes with multiple neighbor
    // configurations (e.g. ERD: alternating Y rows), compute the index here.
    // For Cube (single config), this is unused; leave the line for parity.
    // Examples:
    //   Cube : int configIndex = 0;
    //   ERD  : int configIndex = cellCoords.y & 1;  // y%2 for non-negative y
    // int configIndex = 0;

    // === SHAPE-SPECIFIC: face neighbor block ===
    // Cube has 1 config × 6 deltas; access is kFaceDeltas[i].
    // ERD  has 2 configs × 12 deltas; access is kFaceDeltas[configIndex][i].
    if ((neighborhoodFlags & FACE_NEIGHBORS_MASK) != 0u)
    {
        for (int i = 0; i < 6; i++)
            if (isAliveNeighbor(cellCoords + kFaceDeltas[i], maxCellState, dims, wraps)) count++;
    }

    // === SHAPE-SPECIFIC: edge neighbor block ===
    // Cube : 12 deltas, configuration-agnostic
    // ERD  :  4 deltas, configuration-agnostic
    // Delete this block if your shape has zero edge deltas.
    if ((neighborhoodFlags & EDGE_NEIGHBORS_MASK) != 0u)
    {
        for (int i = 0; i < 12; i++)
            if (isAliveNeighbor(cellCoords + kEdgeDeltas[i], maxCellState, dims, wraps)) count++;
    }

    // === SHAPE-SPECIFIC: corner neighbor block ===
    // Cube : 8 deltas, configuration-agnostic
    // ERD  : NO corner deltas — entire block deleted for ERD
    // Delete this block if your shape has zero corner deltas.
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

            // === SHAPE-SPECIFIC: IsCellViable call site — same as in isAliveNeighbor above ===
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
