// Ghost layer Z-slice copy — extracts/inserts Z-slices between tiled Morton buffers
// and linear staging buffers for cross-GPU ghost exchange.
//
// Two entry points compiled separately via SlangCompiler:
//   extractMain — tiled cell buffer → linear staging buffer
//   insertMain  — linear staging buffer → tiled cell buffer
//
// Requires defines: IMAGE_WIDTH, GRID_DIM_Y

import cellStateUtils;

[vk::binding(0, 0)] RWStructuredBuffer<uint> tiledBuffer;
[vk::binding(1, 0)] RWStructuredBuffer<uint> linearBuffer;

struct GhostCopyPushConstants
{
    int zStart;       // global Z of first slice to extract/insert
    int imageWidth;   // packed texel width = ceil(gridX / 2)
    int gridDimY;     // Y dimension
    int numBlocksX;   // tiled block counts
    int numBlocksY;
    int numBlocksZ;
};
[[vk::push_constant]] GhostCopyPushConstants pushConstants;

// Extract: read from tiled buffer at Z-slices [zStart, zStart+dispatchZ), write to linear staging
[shader("compute")]
[numthreads(8, 8, 1)]
void extractMain(uint3 globalID : SV_DispatchThreadID)
{
    int x = int(globalID.x);
    int y = int(globalID.y);
    int sliceIndex = int(globalID.z);

    if (x >= pushConstants.imageWidth || y >= pushConstants.gridDimY)
        return;

    int z = pushConstants.zStart + sliceIndex;
    int3 numBlocks = int3(pushConstants.numBlocksX, pushConstants.numBlocksY, pushConstants.numBlocksZ);

    uint value = tiledBuffer[tiledBufferIndex(int3(x, y, z), numBlocks)];

    int linearIndex = sliceIndex * pushConstants.imageWidth * pushConstants.gridDimY
                    + y * pushConstants.imageWidth + x;
    linearBuffer[linearIndex] = value;
}

// Insert: read from linear staging, write into tiled buffer at Z-slices [zStart, zStart+dispatchZ)
[shader("compute")]
[numthreads(8, 8, 1)]
void insertMain(uint3 globalID : SV_DispatchThreadID)
{
    int x = int(globalID.x);
    int y = int(globalID.y);
    int sliceIndex = int(globalID.z);

    if (x >= pushConstants.imageWidth || y >= pushConstants.gridDimY)
        return;

    int z = pushConstants.zStart + sliceIndex;
    int3 numBlocks = int3(pushConstants.numBlocksX, pushConstants.numBlocksY, pushConstants.numBlocksZ);

    int linearIndex = sliceIndex * pushConstants.imageWidth * pushConstants.gridDimY
                    + y * pushConstants.imageWidth + x;
    uint value = linearBuffer[linearIndex];

    tiledBuffer[tiledBufferIndex(int3(x, y, z), numBlocks)] = value;
}
