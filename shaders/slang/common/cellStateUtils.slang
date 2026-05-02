// Shared cell state utilities — packed 4-bit format helpers.
// Two cells packed per byte along X axis: low nibble = even X, high nibble = odd X.
// Buffer mode: 8x8x8 tiled blocks with Morton order within each block.
// Image mode: direct 3D texture indexing via imageLoad/imageStore.

#ifdef USE_IMAGE

// Read a single cell's 4-bit state from a packed 3D image
uint readCellState(RWTexture3D<uint> cellStateImage, int3 cellCoordinates)
{
    int3 texelCoordinates = int3(cellCoordinates.x >> 1, cellCoordinates.y, cellCoordinates.z);
    uint packed = cellStateImage[texelCoordinates];
    return ((cellCoordinates.x & 1) == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
}

#else // Buffer mode

// Interleave 3 bits of a single axis coordinate into Morton position
uint expandBits3(uint v)
{
    v &= 7u;
    v = (v | (v << 8u)) & 0x100Fu;
    v = (v | (v << 4u)) & 0x10C30C3u;
    v = (v | (v << 2u)) & 0x1249249u;
    return v;
}

// 3D Morton encode for coordinates within an 8x8x8 block (each axis 0-7)
uint mortonEncode3D(int3 localCoord)
{
    return expandBits3(uint(localCoord.x))
         | (expandBits3(uint(localCoord.y)) << 1u)
         | (expandBits3(uint(localCoord.z)) << 2u);
}

// Compute tiled buffer index from 3D texel coordinates
// numBlocks = ceil(texelDimension / 8) per axis
uint tiledBufferIndex(int3 texelCoord, int3 numBlocks)
{
    int3 blockCoord = texelCoord / 8;
    int3 localCoord = texelCoord - blockCoord * 8;
    uint blockIndex = uint(blockCoord.z) * uint(numBlocks.x) * uint(numBlocks.y)
                    + uint(blockCoord.y) * uint(numBlocks.x)
                    + uint(blockCoord.x);
    return blockIndex * 512u + mortonEncode3D(localCoord);
}

// Read a single cell's 4-bit state from a packed buffer
uint readCellState(RWStructuredBuffer<uint> cellStateBuffer, int3 cellCoordinates, int3 numCellBlocks)
{
    int3 texelCoordinates = int3(cellCoordinates.x >> 1, cellCoordinates.y, cellCoordinates.z);
    uint packed = cellStateBuffer[tiledBufferIndex(texelCoordinates, numCellBlocks)];
    return ((cellCoordinates.x & 1) == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
}

#endif // USE_IMAGE

// Pack two 4-bit cell states into a single byte
uint packCellStates(uint stateA, uint stateB)
{
    return (stateB << 4) | (stateA & 0xF);
}

// Bounds check for cell coordinates against grid dimensions
bool isInBounds(int3 cellCoordinates, int3 gridDimensions)
{
    return cellCoordinates.x >= 0 && cellCoordinates.x < gridDimensions.x
        && cellCoordinates.y >= 0 && cellCoordinates.y < gridDimensions.y
        && cellCoordinates.z >= 0 && cellCoordinates.z < gridDimensions.z;
}

// Wrap coordinates for boundary wrapping mode
int3 wrapCoordinates(int3 cellCoordinates, int3 gridDimensions)
{
    return (cellCoordinates + gridDimensions) % gridDimensions;
}
