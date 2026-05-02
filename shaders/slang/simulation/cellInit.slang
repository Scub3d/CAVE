// Cell initialization compute shader — populates spawn data into packed 4-bit cell state.
// Each thread writes 1 packed byte (2 cells along X). For each alive cell it also
// scatter-writes the skip grid to mark active chunks; without this the skip grid stays
// at its post-clear all-zero state and the FIRST frame's ray march has no acceleration,
// which TDRs the GPU on grids ≳1300³ even at 1080p.
// Requires defines: IMAGE_WIDTH, GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z,
//                   SPAWN_MIN_X/Y/Z, SPAWN_MAX_X/Y/Z, SPAWN_MODE (0=random, 1=filled),
//                   SPAWN_RANDOM_DENSITY (float in [0,1], only used for random mode),
//                   CHUNK_SIZE, SKIP_GRID_DIM_X/Y/Z, MAX_NEIGHBOR_REACH_X/Y/Z

import cellStateUtils;

#ifdef USE_IMAGE
[vk::binding(0, 0)] [vk::image_format("r8ui")] RWTexture3D<uint> cellStateImage;
[vk::binding(1, 0)] [vk::image_format("r32ui")] RWTexture3D<uint> skipGridImage;
#else
[vk::binding(0, 0)] RWStructuredBuffer<uint> cellStateBuffer;
[vk::binding(1, 0)] RWStructuredBuffer<uint> skipGridBuffer;
#endif

struct InitPushConstants
{
    uint64_t birthAndMaxCellStateRules;
    uint64_t survivalAndNeighborhoodRules;
    uint64_t shapeAndGridConfiguration;
    uint enableComputeSkip;
    uint padding0;
};
[[vk::push_constant]] InitPushConstants pushConstants;

static const int BIT_SHIFT = 60;

// Wang hash — well-known single-seed integer hash that produces uniformly distributed uint32s.
uint wangHash(uint seed)
{
    seed = (seed ^ 61u) ^ (seed >> 16);
    seed *= 9u;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15);
    return seed;
}

uint hashCell(int3 cell)
{
    uint h = wangHash(uint(cell.x) * 73856093u);
    h = wangHash(h ^ uint(cell.y) * 19349663u);
    h = wangHash(h ^ uint(cell.z) * 83492791u);
    return h;
}

bool cellFillsByRandom(int3 cell)
{
    // Compare hash directly to a uint threshold derived from density.
    // Avoids any float-conversion precision concerns. threshold = floor(density * 2^32).
    // density = 1.0 → threshold overflows to 0; handle as always-true.
    if (SPAWN_RANDOM_DENSITY >= 1.0) return true;
    uint threshold = uint(SPAWN_RANDOM_DENSITY * 4294967296.0);
    return hashCell(cell) < threshold;
}

#ifndef USE_IMAGE
static const int3 numCellBlocks = int3(
    (IMAGE_WIDTH + 7) / 8,
    (GRID_DIM_Y + 7) / 8,
    (GRID_DIM_Z + 7) / 8);

static const int3 numSkipBlocks = int3(
    (SKIP_GRID_DIM_X + 7) / 8,
    (SKIP_GRID_DIM_Y + 7) / 8,
    (SKIP_GRID_DIM_Z + 7) / 8);
#endif

// Match rayMarchCompute.slang's scatter-write pattern so the post-init skip grid
// matches what a simulation tick would have produced. Marks the cell's chunk plus
// any neighbor chunk within MAX_NEIGHBOR_REACH (which the simulation needs to
// inspect when computing the next tick).
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
                skipGridImage[int3(x, y, z)] = 1;
#else
                skipGridBuffer[tiledBufferIndex(int3(x, y, z), numSkipBlocks)] = 1;
#endif
            }
}

[shader("compute")]
[numthreads(8, 8, 4)]
void initMain(uint3 globalID : SV_DispatchThreadID)
{
    int3 texelCoord = int3(globalID);
    if (texelCoord.x >= IMAGE_WIDTH || texelCoord.y >= GRID_DIM_Y || texelCoord.z >= GRID_DIM_Z)
        return;

    // maxCellState uses shift-by-1 encoding: actualValue = storedBits + 1 (range 1..16).
    uint maxCellState = uint(pushConstants.birthAndMaxCellStateRules >> BIT_SHIFT) + 1u;

    int cellAx = texelCoord.x * 2;
    int cellBx = cellAx + 1;
    int3 cellA = int3(cellAx, texelCoord.y, texelCoord.z);
    int3 cellB = int3(cellBx, texelCoord.y, texelCoord.z);

    int3 spawnMin = int3(SPAWN_MIN_X, SPAWN_MIN_Y, SPAWN_MIN_Z);
    int3 spawnMax = int3(SPAWN_MAX_X, SPAWN_MAX_Y, SPAWN_MAX_Z);

    bool aInSpawn = all(cellA >= spawnMin) && all(cellA < spawnMax);
    bool bInSpawn = cellBx < GRID_DIM_X && all(cellB >= spawnMin) && all(cellB < spawnMax);

#if SPAWN_MODE == 0
    bool aAlive = aInSpawn && cellFillsByRandom(cellA);
    bool bAlive = bInSpawn && cellFillsByRandom(cellB);
#else
    bool aAlive = aInSpawn;
    bool bAlive = bInSpawn;
#endif

    uint stateA = aAlive ? maxCellState : 0;
    uint stateB = bAlive ? maxCellState : 0;
#ifdef USE_IMAGE
    cellStateImage[texelCoord] = packCellStates(stateA, stateB);
#else
    cellStateBuffer[tiledBufferIndex(texelCoord, numCellBlocks)] = packCellStates(stateA, stateB);
#endif

    // Mark the skip-grid chunks containing/adjacent-to alive cells so the first
    // frame's ray march has correct acceleration (skip grid is otherwise zero-init).
    if (aAlive) scatterWriteSkipGrid(cellA);
    if (bAlive) scatterWriteSkipGrid(cellB);
}
