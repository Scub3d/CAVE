// Cube shape implementations — Face, FaceEdge, and FaceEdgeCorner neighborhoods.
import shape;

// Module-level constant arrays for neighbor deltas
static const int3 kFaceDeltas[6] = {
    int3( 1, 0, 0), int3( 0, 1, 0), int3( 0, 0, 1),
    int3(-1, 0, 0), int3( 0,-1, 0), int3( 0, 0,-1)
};

static const int3 kEdgeDeltas[12] = {
    int3( 1, 1, 0), int3( 1,-1, 0), int3(-1, 1, 0), int3(-1,-1, 0),
    int3( 1, 0, 1), int3( 1, 0,-1), int3(-1, 0, 1), int3(-1, 0,-1),
    int3( 0, 1, 1), int3( 0, 1,-1), int3( 0,-1, 1), int3( 0,-1,-1)
};

static const int3 kCornerDeltas[8] = {
    int3( 1, 1, 1), int3(-1, 1, 1), int3( 1,-1, 1), int3(-1,-1, 1),
    int3( 1, 1,-1), int3(-1, 1,-1), int3( 1,-1,-1), int3(-1,-1,-1)
};

// Shape geometry helpers — shared by all three Cube neighborhood variants since the
// underlying cell shape is identical. Each cell occupies an AABB centered at its world
// position with half-extent equal to meshScaleRatio.

float3 cubeComputeWorldPosition(int3 cellCoordinates, float meshScaleRatio, float3 centeringOffset)
{
    float spacing = 2.0 * meshScaleRatio;
    return float3(cellCoordinates) * spacing - centeringOffset;
}

bool cubeIntersect(float3 rayOrigin, float3 rayDirection, float3 cellCenter, float meshScaleRatio,
                   out float hitDistance, out float3 hitNormal)
{
    float3 cellMin = cellCenter - float3(meshScaleRatio, meshScaleRatio, meshScaleRatio);
    float3 cellMax = cellCenter + float3(meshScaleRatio, meshScaleRatio, meshScaleRatio);
    float3 inverseDirection = 1.0 / rayDirection;
    float3 t0 = (cellMin - rayOrigin) * inverseDirection;
    float3 t1 = (cellMax - rayOrigin) * inverseDirection;
    float3 tMinVec = min(t0, t1);
    float3 tMaxVec = max(t0, t1);
    float tEnter = max(max(tMinVec.x, tMinVec.y), tMinVec.z);
    float tExit = min(min(tMaxVec.x, tMaxVec.y), tMaxVec.z);
    hitDistance = 0.0;
    hitNormal = float3(0.0, 0.0, 0.0);
    if (tExit < max(tEnter, 0.0)) return false;
    hitDistance = max(tEnter, 0.0);
    if (tMinVec.x > tMinVec.y && tMinVec.x > tMinVec.z)
        hitNormal = float3(-sign(rayDirection.x), 0.0, 0.0);
    else if (tMinVec.y > tMinVec.z)
        hitNormal = float3(0.0, -sign(rayDirection.y), 0.0);
    else
        hitNormal = float3(0.0, 0.0, -sign(rayDirection.z));
    return true;
}

bool cubeIsNearCellEdge(float3 localHitPoint, float edgeWidth)
{
    float3 d = float3(1.0, 1.0, 1.0) - abs(localHitPoint);
    int nearCount = 0;
    if (d.x < edgeWidth) nearCount++;
    if (d.y < edgeWidth) nearCount++;
    if (d.z < edgeWidth) nearCount++;
    return nearCount >= 2;
}

// Face neighbors only (6 neighbors, reach 1)
struct CubeFace : IShape
{
    static const int NeighborCount = 6;
    static const int NeighborConfigurationCount = 1;
    static const int MaxReachX = 1;
    static const int MaxReachY = 1;
    static const int MaxReachZ = 1;

    static int GetNeighborConfigurationIndex(int3 cellCoordinates) { return 0; }

    static int3 GetNeighborDelta(int neighborConfigurationIndex, int neighborIndex)
    {
        return kFaceDeltas[neighborIndex];
    }

    static bool IsCellViable(uint3 cellCoordinates, uint3 gridDimensions)
    {
        return true;
    }

    static float3 ComputeWorldPosition(int3 cellCoord, float meshScaleRatio, float3 centeringOffset)
    {
        return cubeComputeWorldPosition(cellCoord, meshScaleRatio, centeringOffset);
    }

    static bool IntersectCellShape(float3 rayOrigin, float3 rayDirection, float3 cellCenter,
                                   float meshScaleRatio, out float hitDistance, out float3 hitNormal)
    {
        return cubeIntersect(rayOrigin, rayDirection, cellCenter, meshScaleRatio, hitDistance, hitNormal);
    }

    static bool IsNearCellEdge(float3 localHitPoint, float edgeWidth)
    {
        return cubeIsNearCellEdge(localHitPoint, edgeWidth);
    }
}

// Face + Edge neighbors (18 neighbors, reach 1)
struct CubeFaceEdge : IShape
{
    static const int NeighborCount = 18;
    static const int NeighborConfigurationCount = 1;
    static const int MaxReachX = 1;
    static const int MaxReachY = 1;
    static const int MaxReachZ = 1;

    static int GetNeighborConfigurationIndex(int3 cellCoordinates) { return 0; }

    static int3 GetNeighborDelta(int neighborConfigurationIndex, int neighborIndex)
    {
        if (neighborIndex < 6) return kFaceDeltas[neighborIndex];
        return kEdgeDeltas[neighborIndex - 6];
    }

    static bool IsCellViable(uint3 cellCoordinates, uint3 gridDimensions)
    {
        return true;
    }

    static float3 ComputeWorldPosition(int3 cellCoord, float meshScaleRatio, float3 centeringOffset)
    {
        return cubeComputeWorldPosition(cellCoord, meshScaleRatio, centeringOffset);
    }

    static bool IntersectCellShape(float3 rayOrigin, float3 rayDirection, float3 cellCenter,
                                   float meshScaleRatio, out float hitDistance, out float3 hitNormal)
    {
        return cubeIntersect(rayOrigin, rayDirection, cellCenter, meshScaleRatio, hitDistance, hitNormal);
    }

    static bool IsNearCellEdge(float3 localHitPoint, float edgeWidth)
    {
        return cubeIsNearCellEdge(localHitPoint, edgeWidth);
    }
}

// Face + Edge + Corner neighbors (26 neighbors, reach 1)
struct CubeFaceEdgeCorner : IShape
{
    static const int NeighborCount = 26;
    static const int NeighborConfigurationCount = 1;
    static const int MaxReachX = 1;
    static const int MaxReachY = 1;
    static const int MaxReachZ = 1;

    static int GetNeighborConfigurationIndex(int3 cellCoordinates) { return 0; }

    static int3 GetNeighborDelta(int neighborConfigurationIndex, int neighborIndex)
    {
        if (neighborIndex < 6) return kFaceDeltas[neighborIndex];
        if (neighborIndex < 18) return kEdgeDeltas[neighborIndex - 6];
        return kCornerDeltas[neighborIndex - 18];
    }

    static bool IsCellViable(uint3 cellCoordinates, uint3 gridDimensions)
    {
        return true;
    }

    static float3 ComputeWorldPosition(int3 cellCoord, float meshScaleRatio, float3 centeringOffset)
    {
        return cubeComputeWorldPosition(cellCoord, meshScaleRatio, centeringOffset);
    }

    static bool IntersectCellShape(float3 rayOrigin, float3 rayDirection, float3 cellCenter,
                                   float meshScaleRatio, out float hitDistance, out float3 hitNormal)
    {
        return cubeIntersect(rayOrigin, rayDirection, cellCenter, meshScaleRatio, hitDistance, hitNormal);
    }

    static bool IsNearCellEdge(float3 localHitPoint, float edgeWidth)
    {
        return cubeIsNearCellEdge(localHitPoint, edgeWidth);
    }
}
