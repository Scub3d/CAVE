// Ray march compute shader — simulates 3D cellular automata with packed 4-bit cell states.
// Each thread processes 2 cells packed into 1 byte along the X axis.
// Shape is selected at compile time via SHAPE_TYPE define.
// Cube: 0=Face, 1=FaceEdge, 2=FaceEdgeCorner
// Elongated Dodecahedron: 3=Face, 4=FaceEdge

import shape;
import cube;
import elongatedDodecahedron;
import cellStateUtils;

// Select shape type via preprocessor define
#ifndef SHAPE_TYPE
#define SHAPE_TYPE 0
#endif

#if SHAPE_TYPE == 0
typedef CubeFace SelectedShape;
#elif SHAPE_TYPE == 1
typedef CubeFaceEdge SelectedShape;
#elif SHAPE_TYPE == 2
typedef CubeFaceEdgeCorner SelectedShape;
#elif SHAPE_TYPE == 3
typedef ElongatedDodecahedronFace SelectedShape;
#elif SHAPE_TYPE == 4
typedef ElongatedDodecahedronFaceEdge SelectedShape;
#endif

#ifdef USE_IMAGE
// Storage images (ping-pong cell state + skip grids)
[vk::binding(0, 0)] [vk::image_format("r8ui")] RWTexture3D<uint> cellStateImageIn;
[vk::binding(1, 0)] [vk::image_format("r8ui")] RWTexture3D<uint> cellStateImageOut;
[vk::binding(2, 0)] [vk::image_format("r32ui")] RWTexture3D<uint> skipGridIn;
[vk::binding(3, 0)] [vk::image_format("r32ui")] RWTexture3D<uint> skipGridOut;
#else
// Storage buffers (ping-pong cell state + skip grids)
[vk::binding(0, 0)] RWStructuredBuffer<uint> cellStateBufferIn;
[vk::binding(1, 0)] RWStructuredBuffer<uint> cellStateBufferOut;
[vk::binding(2, 0)] RWStructuredBuffer<uint> skipGridBufferIn;
[vk::binding(3, 0)] RWStructuredBuffer<uint> skipGridBufferOut;
#endif

// Push constants: simulation parameters + compute skip toggle
struct SimulationPushConstants
{
    uint64_t birthAndMaxCellStateRules;
    uint64_t survivalAndNeighborhoodRules;
    uint64_t shapeAndGridConfiguration;
    uint enableComputeSkip;
    uint padding0;
};
[[vk::push_constant]] SimulationPushConstants pushConstants;

// Grid and image dimensions — injected as preprocessor defines at compile time
// These MUST be defined by the compiler: GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z,
// IMAGE_WIDTH, CHUNK_SIZE, SKIP_GRID_DIM_X/Y/Z, WRAP_AT_BOUNDARY
static const int BIT_SHIFT = 60;
static const uint64_t EXISTENCE_PERMUTATION_BIT_MASK = 0x0FFFFFFFFFFFFFFF;

#ifndef USE_IMAGE
// Tiled block counts — computed from compile-time defines (buffer mode only)
static const int3 numCellBlocks = int3(
    (IMAGE_WIDTH + 7) / 8,
    (GRID_DIM_Y + 7) / 8,
    (GRID_DIM_Z + 7) / 8);

static const int3 numSkipBlocks = int3(
    (SKIP_GRID_DIM_X + 7) / 8,
    (SKIP_GRID_DIM_Y + 7) / 8,
    (SKIP_GRID_DIM_Z + 7) / 8);
#endif

// Check if a neighbor cell is alive (state == maxCellState)
bool isAliveNeighbor(int3 neighborCoordinates, uint maxCellState)
{
    int3 gridDimensions = int3(GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z);

    // Handle X/Y boundary (wrap or clamp)
#if WRAP_AT_BOUNDARY
    neighborCoordinates.x = ((neighborCoordinates.x % gridDimensions.x) + gridDimensions.x) % gridDimensions.x;
    neighborCoordinates.y = ((neighborCoordinates.y % gridDimensions.y) + gridDimensions.y) % gridDimensions.y;
#else
    if (neighborCoordinates.x < 0 || neighborCoordinates.x >= gridDimensions.x
        || neighborCoordinates.y < 0 || neighborCoordinates.y >= gridDimensions.y)
        return false;
#endif

    // Handle Z boundary — when domain decomposed, ghost exchange populates boundary
    // data in the ghost slices, so the shader never wraps Z (just bounds-checks the
    // local grid which includes ghost slices on both sides).
#if WRAP_AT_BOUNDARY && !defined(DOMAIN_DECOMPOSITION)
    neighborCoordinates.z = ((neighborCoordinates.z % gridDimensions.z) + gridDimensions.z) % gridDimensions.z;
#else
    if (neighborCoordinates.z < 0 || neighborCoordinates.z >= gridDimensions.z)
        return false;
#endif

#ifdef USE_IMAGE
    return readCellState(cellStateImageIn, neighborCoordinates) == maxCellState;
#else
    return readCellState(cellStateBufferIn, neighborCoordinates, numCellBlocks) == maxCellState;
#endif
}

// Count alive neighbors for a cell using the shape's topology
int countAliveNeighbors(int3 cellCoordinates, uint maxCellState)
{
    int neighborConfigurationIndex = SelectedShape.GetNeighborConfigurationIndex(cellCoordinates);
    int aliveCount = 0;
    for (int i = 0; i < SelectedShape.NeighborCount; i++)
    {
        if (isAliveNeighbor(cellCoordinates + SelectedShape.GetNeighborDelta(neighborConfigurationIndex, i), maxCellState))
            aliveCount++;
    }
    return aliveCount;
}

// Simulate a single cell: apply birth/survival rules, return new state
uint simulateCell(int3 cellCoordinates, uint currentState,
                               uint maxCellState, uint64_t survivalRules, uint64_t birthRules)
{
    int3 gridDimensions = int3(GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z);

    if (!SelectedShape.IsCellViable(uint3(cellCoordinates), uint3(gridDimensions)))
        return 0;

    uint aliveNeighborCount = uint(countAliveNeighbors(cellCoordinates, maxCellState));

    if (currentState == maxCellState)
    {
        if (aliveNeighborCount > 0u && ((survivalRules >> (aliveNeighborCount - 1u)) & 1) == 1UL)
            return maxCellState;
        else
            return currentState - 1;
    }
    else if (currentState == 0)
    {
        if (aliveNeighborCount > 0u && ((birthRules >> (aliveNeighborCount - 1u)) & 1) == 1UL)
            return maxCellState;
        else
            return 0;
    }
    else
    {
        return currentState - 1;
    }
}

// Scatter-write to skip grid: mark this cell's chunk and affected neighbor chunks
void scatterWriteSkipGrid(int3 cellCoordinates)
{
    int3 maxNeighborReach = int3(MAX_NEIGHBOR_REACH_X, MAX_NEIGHBOR_REACH_Y, MAX_NEIGHBOR_REACH_Z);
    int3 skipGridDimensions = int3(SKIP_GRID_DIM_X, SKIP_GRID_DIM_Y, SKIP_GRID_DIM_Z);

    int3 minAffectedChunk = max(
        int3(floor(float3(cellCoordinates - maxNeighborReach) / float(CHUNK_SIZE))),
        int3(0));
    int3 maxAffectedChunk = min(
        int3(floor(float3(cellCoordinates + maxNeighborReach) / float(CHUNK_SIZE))),
        skipGridDimensions - 1);

    for (int z = minAffectedChunk.z; z <= maxAffectedChunk.z; z++)
        for (int y = minAffectedChunk.y; y <= maxAffectedChunk.y; y++)
            for (int x = minAffectedChunk.x; x <= maxAffectedChunk.x; x++)
            {
#ifdef USE_IMAGE
                skipGridOut[int3(x, y, z)] = 1;
#else
                skipGridBufferOut[tiledBufferIndex(int3(x, y, z), numSkipBlocks)] = 1;
#endif
            }
}

// Main entry point — generic over shape type.
// Each thread processes 2 cells packed into 1 byte along X.
// When domain decomposed, dispatch covers only owned Z slices.
// Thread Z is offset by Z_OWNED_START to map into the local buffer
// (which has layout: [ghost_lo, owned, ghost_hi]).
[shader("compute")]
[numthreads(8, 8, 8)]
void computeMain(uint3 globalID : SV_DispatchThreadID)
{
    int3 texelCoordinates = int3(globalID);

#ifdef DOMAIN_DECOMPOSITION
    // Offset thread Z past ghost_lo into the owned range
    texelCoordinates.z += Z_OWNED_START;
    if (texelCoordinates.x >= IMAGE_WIDTH || texelCoordinates.y >= GRID_DIM_Y || texelCoordinates.z >= Z_OWNED_END)
        return;
#else
    if (texelCoordinates.x >= IMAGE_WIDTH || texelCoordinates.y >= GRID_DIM_Y || texelCoordinates.z >= GRID_DIM_Z)
        return;
#endif

    int3 cellA = int3(texelCoordinates.x * 2, texelCoordinates.y, texelCoordinates.z);
    int3 cellB = int3(texelCoordinates.x * 2 + 1, texelCoordinates.y, texelCoordinates.z);
    bool cellBValid = cellB.x < GRID_DIM_X;

    // Compute skip: both cells share the same chunk (even/odd pairs never straddle 8-cell boundaries)
    if (pushConstants.enableComputeSkip > 0)
    {
        int3 chunkCoordinates = cellA / int(CHUNK_SIZE);
#ifdef USE_IMAGE
        if (skipGridIn[chunkCoordinates] == 0)
        {
            cellStateImageOut[texelCoordinates] = 0;
            return;
        }
#else
        if (skipGridBufferIn[tiledBufferIndex(chunkCoordinates, numSkipBlocks)] == 0)
        {
            cellStateBufferOut[tiledBufferIndex(texelCoordinates, numCellBlocks)] = 0;
            return;
        }
#endif
    }

    // maxCellState uses shift-by-1 encoding: actualValue = storedBits + 1 (range 1..16).
    uint maxCellState = uint(pushConstants.birthAndMaxCellStateRules >> BIT_SHIFT) + 1u;
    uint64_t survivalRules = pushConstants.survivalAndNeighborhoodRules & EXISTENCE_PERMUTATION_BIT_MASK;
    uint64_t birthRules = pushConstants.birthAndMaxCellStateRules & EXISTENCE_PERMUTATION_BIT_MASK;

    // Read current packed byte
#ifdef USE_IMAGE
    uint packed = cellStateImageIn[texelCoordinates];
#else
    uint packed = cellStateBufferIn[tiledBufferIndex(texelCoordinates, numCellBlocks)];
#endif
    uint currentStateA = packed & 0xF;
    uint currentStateB = (packed >> 4) & 0xF;

    // Simulate both cells
    uint newStateA = simulateCell(cellA, currentStateA, maxCellState, survivalRules, birthRules);
    uint newStateB = cellBValid ? simulateCell(cellB, currentStateB, maxCellState, survivalRules, birthRules) : 0;

    // Pack and write
#ifdef USE_IMAGE
    cellStateImageOut[texelCoordinates] = packCellStates(newStateA, newStateB);
#else
    cellStateBufferOut[tiledBufferIndex(texelCoordinates, numCellBlocks)] = packCellStates(newStateA, newStateB);
#endif

    // Skip grid scatter-write for each cell
    if (newStateA > 0) scatterWriteSkipGrid(cellA);
    if (cellBValid && newStateB > 0) scatterWriteSkipGrid(cellB);
}
