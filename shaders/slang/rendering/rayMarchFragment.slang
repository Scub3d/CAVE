// Ray march fragment shader — DDA ray marching through 3D cellular automata volume.
// Two-level DDA: coarse (skip grid chunks) + fine (individual cells).
// Reads packed 4-bit cell states via readCellState().

import cellStateUtils;
import cube;
import elongatedRhombicDodecahedron;

// Camera UBO — must match C++ Cave::CameraData layout exactly
struct CameraUBO
{
    float4x4 model;
    float4x4 view;
    float4x4 projection;
    float4 lightDirection;      // xyz = direction, w = enableLighting (0.0 or 1.0)
    float4 lightingParameters;  // x = ambient, y = diffuse, z = specular, w = shininess
    float4 cameraPosition;      // xyz = camera world position, w = unused
    float4 cullingPlanes[4];
    uint4 cullingControl;
};

// Push constants — must match C++ Cave::RayMarchPushConstants layout exactly
struct RayMarchConstants
{
    uint gridDimensionX;
    uint gridDimensionY;
    uint gridDimensionZ;
    uint maxCellState;
    float volumeMinX;
    float volumeMinY;
    float volumeMinZ;
    float volumeMaxX;
    float volumeMaxY;
    float volumeMaxZ;
    float aliveColorR;
    float aliveColorG;
    float aliveColorB;
    float deadColorR;
    float deadColorG;
    float deadColorB;
    float occlusionStrength;
    float maxNeighborCount;
    uint chunkSize;
    uint computeNeighborCount;
    uint debugRandomCellColors;
    float meshScaleRatio;
    float centeringOffsetX;
    float centeringOffsetY;
    float centeringOffsetZ;
    uint ownedZMin;
    uint ownedZMax;
};

// Bindings — selected by USE_IMAGE define
[vk::binding(0, 0)] ConstantBuffer<CameraUBO> cameraData;

#ifdef USE_IMAGE
[vk::binding(1, 0)] [vk::image_format("r8ui")] RWTexture3D<uint> cellStateImage;
[vk::binding(2, 0)] [vk::image_format("r32ui")] RWTexture3D<uint> skipGrid;
#else
[vk::binding(1, 0)] RWStructuredBuffer<uint> cellStateBuffer;
[vk::binding(2, 0)] RWStructuredBuffer<uint> skipGridBuffer;
#endif

[[vk::push_constant]] RayMarchConstants rayMarch;

// Shape selection via preprocessor define
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
typedef ElongatedRhombicDodecahedronFace SelectedShape;
#elif SHAPE_TYPE == 4
typedef ElongatedRhombicDodecahedronFaceEdge SelectedShape;
#endif

// Spatial reach defaults — C++ injects actual values based on neighbor deltas.
// Cells in (-reach..reach)^3 are checked for overlapping geometry at each fine-DDA step.
#ifndef SPATIAL_REACH_X
#define SPATIAL_REACH_X 0
#endif
#ifndef SPATIAL_REACH_Y
#define SPATIAL_REACH_Y 0
#endif
#ifndef SPATIAL_REACH_Z
#define SPATIAL_REACH_Z 0
#endif

// Returns true if the given world-space point is on the culled side of any active plane.
bool isCulledByPlanes(float3 worldPos)
{
    uint mask = cameraData.cullingControl.x;
    if (mask == 0) return false;
    for (int p = 0; p < 4; p++)
    {
        if ((mask & (1u << uint(p))) == 0) continue;
        float4 plane = cameraData.cullingPlanes[p];
        if (dot(plane.xyz, worldPos) > plane.w) return true;
    }
    return false;
}

// Cell state read — abstracts over image vs buffer storage
uint readCellStateGeneric(int3 cellCoord, int3 numCellBlocks)
{
#ifdef USE_IMAGE
    return readCellState(cellStateImage, cellCoord);
#else
    return readCellState(cellStateBuffer, cellCoord, numCellBlocks);
#endif
}

// Skip grid read — abstracts over image vs buffer storage
uint readSkipGridGeneric(int3 chunkCoord, int3 numSkipBlocks)
{
#ifdef USE_IMAGE
    return skipGrid[chunkCoord];
#else
    return skipGridBuffer[tiledBufferIndex(chunkCoord, numSkipBlocks)];
#endif
}

// Hash for debug random cell colors — same constants as the GLSL fragment so renders match.
float3 debugCellColorHash(int3 cellCoord)
{
    uint h = uint(cellCoord.x * 73856093) ^ uint(cellCoord.y * 19349663) ^ uint(cellCoord.z * 83492791);
    float r = float((h >>  0u) & 0xFFu) / 255.0;
    float g = float((h >>  8u) & 0xFFu) / 255.0;
    float b = float((h >> 16u) & 0xFFu) / 255.0;
    return lerp(float3(0.2, 0.2, 0.2), float3(r, g, b), 0.8);
}

// Shading with optional Phong lighting, neighbor-count AO, debug colors + wireframe
float4 ShadeCell(uint cellState, uint neighborCount, float3 hitPoint, float3 hitNormal,
                 int3 cellCoord, float3 cellCenter)
{
    float stateRatio = float(cellState) / float(rayMarch.maxCellState);
    float3 aliveColor = float3(rayMarch.aliveColorR, rayMarch.aliveColorG, rayMarch.aliveColorB);
    float3 deadColor = float3(rayMarch.deadColorR, rayMarch.deadColorG, rayMarch.deadColorB);
    float3 baseColor = (rayMarch.debugRandomCellColors > 0)
        ? debugCellColorHash(cellCoord)
        : lerp(deadColor, aliveColor, stateRatio);

    if (rayMarch.debugRandomCellColors > 0)
    {
        float3 localHitPoint = (hitPoint - cellCenter) / rayMarch.meshScaleRatio;
        if (SelectedShape.IsNearCellEdge(localHitPoint, 0.15))
            baseColor = float3(0.0, 0.0, 0.0);
    }

    float occlusion = 1.0 - float(neighborCount) / rayMarch.maxNeighborCount * rayMarch.occlusionStrength;

    if (cameraData.lightDirection.w > 0.5)
    {
        float3 lightDir = normalize(-cameraData.lightDirection.xyz);
        float ambient = cameraData.lightingParameters.x * occlusion;
        float diffuse = max(dot(hitNormal, lightDir), 0.0) * cameraData.lightingParameters.y;
        float3 viewDir = normalize(cameraData.cameraPosition.xyz - hitPoint);
        float3 halfDir = normalize(lightDir + viewDir);
        float specular = pow(max(dot(hitNormal, halfDir), 0.0), cameraData.lightingParameters.w)
                         * cameraData.lightingParameters.z;
        return float4(baseColor * (ambient + diffuse) + float3(specular, specular, specular), 1.0);
    }
    else
    {
        return float4(baseColor * occlusion, 1.0);
    }
}

// Fragment shader input/output
struct FragmentInput
{
    [[vk::location(0)]] float3 worldPosition : FragmentWorldPosition;
};

struct FragmentOutput
{
    float4 color : SV_Target0;
    float depth : SV_Depth;
};

[shader("fragment")]
FragmentOutput fragmentMain(FragmentInput input)
{
    FragmentOutput output;

    float3 volumeMin = float3(rayMarch.volumeMinX, rayMarch.volumeMinY, rayMarch.volumeMinZ);
    float3 volumeMax = float3(rayMarch.volumeMaxX, rayMarch.volumeMaxY, rayMarch.volumeMaxZ);
    int3 gridDimensions = int3(rayMarch.gridDimensionX, rayMarch.gridDimensionY, rayMarch.gridDimensionZ);
    float3 cellSize = (volumeMax - volumeMin) / float3(gridDimensions);
    int chunkSize = int(rayMarch.chunkSize);
    float3 chunkWorldSize = cellSize * float(chunkSize);
    int3 skipGridDimensions = (gridDimensions + chunkSize - 1) / chunkSize;

    // Buffer-mode tile counts — only used when USE_IMAGE is undefined.
#ifndef USE_IMAGE
    int3 numCellBlocks = (int3(IMAGE_WIDTH, gridDimensions.y, gridDimensions.z) + 7) / 8;
    int3 numSkipBlocks = (skipGridDimensions + 7) / 8;
#else
    int3 numCellBlocks = int3(0);
    int3 numSkipBlocks = int3(0);
#endif

    float3 rayOrigin = cameraData.cameraPosition.xyz;
    float3 rayDirection = normalize(input.worldPosition - rayOrigin);

    // Ray-volume intersection
    float3 invDir = 1.0 / rayDirection;
    float3 t0 = (volumeMin - rayOrigin) * invDir;
    float3 t1 = (volumeMax - rayOrigin) * invDir;
    float3 tMinV = min(t0, t1);
    float3 tMaxV = max(t0, t1);
    float tEnter = max(max(tMinV.x, tMinV.y), tMinV.z);
    float tExit = min(min(tMaxV.x, tMaxV.y), tMaxV.z);
    if (tExit < max(tEnter, 0.0))
        discard;

    float tStart = max(tEnter, 0.0001);
    float3 startPosition = rayOrigin + rayDirection * tStart;
    float3 chunkPosition = (startPosition - volumeMin) / chunkWorldSize;

    int3 currentChunk = int3(floor(chunkPosition));
    currentChunk = clamp(currentChunk, int3(0), skipGridDimensions - 1);

    int3 stepDir = int3(sign(rayDirection));
    float3 coarseTDelta = abs(chunkWorldSize / rayDirection);

    // Initialize coarse DDA tMax
    float3 coarseTMax;
    for (int i = 0; i < 3; i++)
    {
        if (rayDirection[i] > 0.0)
            coarseTMax[i] = tStart + (float(currentChunk[i] + 1) * chunkWorldSize[i] + volumeMin[i] - startPosition[i]) / rayDirection[i];
        else if (rayDirection[i] < 0.0)
            coarseTMax[i] = tStart + (float(currentChunk[i]) * chunkWorldSize[i] + volumeMin[i] - startPosition[i]) / rayDirection[i];
        else
            coarseTMax[i] = 1e30;
    }

    float3 centeringOffset = float3(rayMarch.centeringOffsetX, rayMarch.centeringOffsetY, rayMarch.centeringOffsetZ);

    // Best-hit tracking across the entire ray traversal — needed because shifted cell geometry
    // can extend across cell boundaries, so the first IntersectCellShape success isn't necessarily
    // the closest hit.
    float bestHitDistance = 1e30;
    float3 bestHitNormal = float3(0.0, 0.0, 0.0);
    int3 bestHitCell = int3(0, 0, 0);
    float3 bestHitCenter = float3(0.0, 0.0, 0.0);
    uint bestHitState = 0;

    // Two-level DDA traversal
    float chunkEntryT = tStart;
    uint maxCoarseSteps = uint(skipGridDimensions.x + skipGridDimensions.y + skipGridDimensions.z);

    for (uint coarseStep = 0; coarseStep < maxCoarseSteps; coarseStep++)
    {
        if (any(currentChunk < int3(0)) || any(currentChunk >= skipGridDimensions))
            break;

        // Once the coarse DDA has stepped past the best hit, no closer hit is possible
        if (bestHitState > 0 && chunkEntryT > bestHitDistance)
            break;

        uint occupied = readSkipGridGeneric(currentChunk, numSkipBlocks);
        if (occupied > 0)
        {
            // Fine DDA within this chunk
            int3 chunkCellMin = currentChunk * chunkSize;
            int3 chunkCellMax = min(chunkCellMin + chunkSize, gridDimensions);

            float3 chunkEntryPos = rayOrigin + rayDirection * (chunkEntryT + 0.0001);
            float3 fineGridPos = (chunkEntryPos - volumeMin) / cellSize;

            int3 fineCell = int3(floor(fineGridPos));
            fineCell = clamp(fineCell, chunkCellMin, chunkCellMax - 1);

            float3 fineTDelta = abs(cellSize / rayDirection);
            float3 fineTMax;
            for (int i = 0; i < 3; i++)
            {
                if (rayDirection[i] > 0.0)
                    fineTMax[i] = chunkEntryT + (float(fineCell[i] + 1) * cellSize[i] + volumeMin[i] - chunkEntryPos[i]) / rayDirection[i];
                else if (rayDirection[i] < 0.0)
                    fineTMax[i] = chunkEntryT + (float(fineCell[i]) * cellSize[i] + volumeMin[i] - chunkEntryPos[i]) / rayDirection[i];
                else
                    fineTMax[i] = 1e30;
            }

            for (int fineStep = 0; fineStep < chunkSize * chunkSize * chunkSize; fineStep++)
            {
                if (any(fineCell < chunkCellMin) || any(fineCell >= chunkCellMax))
                    break;

                float currentT = min(fineTMax.x, min(fineTMax.y, fineTMax.z));
                if (bestHitState > 0 && currentT > bestHitDistance)
                    break;

                // Spatial-reach overlap loop — checks neighboring cells whose geometry could
                // intersect the ray at this fine-DDA position. For cell shapes that don't extend
                // outside their own cell box (cube), reach is 0 and this is a single iteration.
                for (int dy = -SPATIAL_REACH_Y; dy <= SPATIAL_REACH_Y; dy++) {
                for (int dx = -SPATIAL_REACH_X; dx <= SPATIAL_REACH_X; dx++) {
                for (int dz = -SPATIAL_REACH_Z; dz <= SPATIAL_REACH_Z; dz++) {
                    int3 checkCell = fineCell + int3(dx, dy, dz);
                    if (any(checkCell < int3(0)) || any(checkCell >= gridDimensions))
                        continue;
                    // Skip ghost cells in domain-decomposed rendering. Each GPU only renders
                    // its owned Z range to avoid double-rendering the boundary.
                    if (checkCell.z < int(rayMarch.ownedZMin) || checkCell.z >= int(rayMarch.ownedZMax))
                        continue;
                    uint state = readCellStateGeneric(checkCell, numCellBlocks);
                    if (state == 0) continue;

                    float3 cellCenter = SelectedShape.ComputeWorldPosition(checkCell, rayMarch.meshScaleRatio, centeringOffset);
                    if (isCulledByPlanes(cellCenter))
                        continue;

                    float hitDistance;
                    float3 hitNormal;
                    if (SelectedShape.IntersectCellShape(rayOrigin, rayDirection, cellCenter, rayMarch.meshScaleRatio, hitDistance, hitNormal))
                    {
                        if (hitDistance < bestHitDistance)
                        {
                            bestHitDistance = hitDistance;
                            bestHitNormal = hitNormal;
                            bestHitCell = checkCell;
                            bestHitState = state;
                            bestHitCenter = cellCenter;
                        }
                    }
                }}}

                // Fine DDA step
                if (fineTMax.x < fineTMax.y)
                {
                    if (fineTMax.x < fineTMax.z) { fineCell.x += stepDir.x; fineTMax.x += fineTDelta.x; }
                    else                          { fineCell.z += stepDir.z; fineTMax.z += fineTDelta.z; }
                }
                else
                {
                    if (fineTMax.y < fineTMax.z) { fineCell.y += stepDir.y; fineTMax.y += fineTDelta.y; }
                    else                          { fineCell.z += stepDir.z; fineTMax.z += fineTDelta.z; }
                }
            }
        }

        // Coarse DDA step
        if (coarseTMax.x < coarseTMax.y)
        {
            if (coarseTMax.x < coarseTMax.z) { chunkEntryT = coarseTMax.x; currentChunk.x += stepDir.x; coarseTMax.x += coarseTDelta.x; }
            else                              { chunkEntryT = coarseTMax.z; currentChunk.z += stepDir.z; coarseTMax.z += coarseTDelta.z; }
        }
        else
        {
            if (coarseTMax.y < coarseTMax.z) { chunkEntryT = coarseTMax.y; currentChunk.y += stepDir.y; coarseTMax.y += coarseTDelta.y; }
            else                              { chunkEntryT = coarseTMax.z; currentChunk.z += stepDir.z; coarseTMax.z += coarseTDelta.z; }
        }
    }

    // Emit the closest hit found across the entire traversal
    if (bestHitState > 0)
    {
        uint neighborCount = 0;
        if (rayMarch.computeNeighborCount > 0)
        {
            int neighborConfigurationIndex = SelectedShape.GetNeighborConfigurationIndex(bestHitCell);
            for (int n = 0; n < SelectedShape.NeighborCount; n++)
            {
                int3 nc = bestHitCell + SelectedShape.GetNeighborDelta(neighborConfigurationIndex, n);
                if (all(nc >= int3(0)) && all(nc < gridDimensions))
                {
                    if (readCellStateGeneric(nc, numCellBlocks) == rayMarch.maxCellState)
                        neighborCount++;
                }
            }
        }

        float3 hitPoint = rayOrigin + rayDirection * bestHitDistance;
        float4 shadedColor = ShadeCell(bestHitState, neighborCount, hitPoint, bestHitNormal, bestHitCell, bestHitCenter);
        float4 clipPosition = mul(cameraData.projection, mul(cameraData.view, float4(hitPoint, 1.0)));
        // Remap [-1,1] clip depth to [0,1] for Vulkan's depth range. (glm::perspective without
        // GLM_FORCE_DEPTH_ZERO_TO_ONE produces [-1,1].)
        output.depth = clipPosition.z / clipPosition.w * 0.5 + 0.5;
#ifdef ENCODE_DEPTH_IN_ALPHA
        // Dual-GPU compositing needs LINEAR distance from camera in alpha (not NDC depth)
        // for stable per-pixel front/back picking. bestHitDistance is the ray parameter,
        // which equals world distance because rayDirection is normalized.
        output.color = float4(shadedColor.rgb, bestHitDistance);
#else
        output.color = shadedColor;
#endif
        return output;
    }

    discard;
    // Unreachable but Slang requires a return.
    output.color = float4(0, 0, 0, 0);
    output.depth = 1.0;
    return output;
}
