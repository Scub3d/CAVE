// Elongated Rhombic Dodecahedron shape — Face and FaceEdge neighborhoods.
// 2 neighbor configurations: even-y rows (config 0) and odd-y rows (config 1).
import shape;

// Face neighbor deltas: 2 configs x 12 deltas
static const int3 kFaceDeltas[2][12] = {
    // Config 0: Even row (y % 2 == 0)
    {
        int3( 1,  0,  0), int3( 0,  1,  0), int3( 0,  0,  1),
        int3(-1,  0,  0), int3( 0, -1,  0), int3( 0,  0, -1),
        int3(-1, -1,  0), int3(-1, -1, -1), int3( 0, -1, -1),
        int3(-1,  1,  0), int3(-1,  1, -1), int3( 0,  1, -1),
    },
    // Config 1: Odd row (y % 2 == 1)
    {
        int3( 1,  0,  0), int3( 0,  1,  0), int3( 0,  0,  1),
        int3(-1,  0,  0), int3( 0, -1,  0), int3( 0,  0, -1),
        int3( 1, -1,  0), int3( 1, -1,  1), int3( 0, -1,  1),
        int3( 1,  1,  0), int3( 1,  1,  1), int3( 0,  1,  1),
    },
};

// Edge neighbor deltas: same for both configs
static const int3 kEdgeDeltas[4] = {
    int3( 1, 0,  1), int3( 1, 0, -1),
    int3(-1, 0,  1), int3(-1, 0, -1),
};

// Shape geometry — 12-face elongated rhombic dodecahedron with alternating-row Y offset.
// Helpers shared by both ED neighborhood variants since the underlying cell shape is identical.

static const float4 kEDFacePlanes[12] = {
    float4( 0.447214,  0.774597,  0.447214, 1.161895),
    float4( 0.447214,  0.774597, -0.447214, 1.161895),
    float4( 0.000000,  0.000000, -1.000000, 0.866025),
    float4( 1.000000,  0.000000,  0.000000, 0.866025),
    float4( 0.447214, -0.774597, -0.447214, 1.161895),
    float4( 0.447214, -0.774597,  0.447214, 1.161895),
    float4(-0.447214, -0.774597,  0.447214, 1.161895),
    float4(-0.447214, -0.774597, -0.447214, 1.161895),
    float4(-0.447214,  0.774597,  0.447214, 1.161895),
    float4(-0.447214,  0.774597, -0.447214, 1.161895),
    float4(-1.000000,  0.000000,  0.000000, 0.866025),
    float4( 0.000000,  0.000000,  1.000000, 0.866025),
};

float3 edComputeWorldPosition(int3 cellCoordinates, float meshScaleRatio, float3 centeringOffset)
{
    float xSpacing = 1.73204 * meshScaleRatio;
    float ySpacing = 2.0 * meshScaleRatio;
    float zSpacing = 1.73204 * meshScaleRatio;
    float halfOffset = 0.86602 * meshScaleRatio;
    float oddRowShift = (uint(cellCoordinates.y) % 2u == 1u) ? halfOffset : 0.0;
    float3 rawPosition = float3(
        float(cellCoordinates.x) * xSpacing + oddRowShift,
        float(cellCoordinates.y) * ySpacing,
        float(cellCoordinates.z) * zSpacing + oddRowShift
    );
    return rawPosition - centeringOffset;
}

bool edIntersect(float3 rayOrigin, float3 rayDirection, float3 cellCenter, float meshScaleRatio,
                 out float hitDistance, out float3 hitNormal)
{
    float3 relOrigin = rayOrigin - cellCenter;
    float tEnter = -1e30;
    float tExit = 1e30;
    float3 enterNormal = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < 12; i++)
    {
        float3 n = kEDFacePlanes[i].xyz;
        float d = kEDFacePlanes[i].w * meshScaleRatio;
        float denom = dot(n, rayDirection);
        float dist = d - dot(n, relOrigin);
        if (abs(denom) < 1e-8)
        {
            if (dist < 0.0) { hitDistance = 0.0; hitNormal = float3(0.0, 0.0, 0.0); return false; }
        }
        else
        {
            float t = dist / denom;
            if (denom < 0.0)
            {
                if (t > tEnter) { tEnter = t; enterNormal = n; }
            }
            else
            {
                if (t < tExit) tExit = t;
            }
        }
    }
    if (tEnter > tExit || tExit < 0.0) { hitDistance = 0.0; hitNormal = float3(0.0, 0.0, 0.0); return false; }
    hitDistance = max(tEnter, 0.0);
    hitNormal = enterNormal;
    return true;
}

bool edIsNearCellEdge(float3 localHitPoint, float edgeWidth)
{
    int nearCount = 0;
    for (int i = 0; i < 12; i++)
    {
        float dist = kEDFacePlanes[i].w - dot(kEDFacePlanes[i].xyz, localHitPoint);
        if (dist < edgeWidth) nearCount++;
    }
    return nearCount >= 2;
}

// Face neighbors only (12 neighbors per config)
struct ElongatedDodecahedronFace : IShape
{
    static const int NeighborCount = 12;
    static const int NeighborConfigurationCount = 2;
    static const int MaxReachX = 1;
    static const int MaxReachY = 1;
    static const int MaxReachZ = 1;

    static int GetNeighborConfigurationIndex(int3 cellCoordinates)
    {
        return cellCoordinates.y % 2;
    }

    static int3 GetNeighborDelta(int neighborConfigurationIndex, int neighborIndex)
    {
        return kFaceDeltas[neighborConfigurationIndex][neighborIndex];
    }

    static bool IsCellViable(uint3 cellCoordinates, uint3 gridDimensions)
    {
        return true;
    }

    static float3 ComputeWorldPosition(int3 cellCoord, float meshScaleRatio, float3 centeringOffset)
    {
        return edComputeWorldPosition(cellCoord, meshScaleRatio, centeringOffset);
    }

    static bool IntersectCellShape(float3 rayOrigin, float3 rayDirection, float3 cellCenter,
                                   float meshScaleRatio, out float hitDistance, out float3 hitNormal)
    {
        return edIntersect(rayOrigin, rayDirection, cellCenter, meshScaleRatio, hitDistance, hitNormal);
    }

    static bool IsNearCellEdge(float3 localHitPoint, float edgeWidth)
    {
        return edIsNearCellEdge(localHitPoint, edgeWidth);
    }
}

// Face + Edge neighbors (16 neighbors per config)
struct ElongatedDodecahedronFaceEdge : IShape
{
    static const int NeighborCount = 16;
    static const int NeighborConfigurationCount = 2;
    static const int MaxReachX = 1;
    static const int MaxReachY = 1;
    static const int MaxReachZ = 1;

    static int GetNeighborConfigurationIndex(int3 cellCoordinates)
    {
        return cellCoordinates.y % 2;
    }

    static int3 GetNeighborDelta(int neighborConfigurationIndex, int neighborIndex)
    {
        if (neighborIndex < 12) return kFaceDeltas[neighborConfigurationIndex][neighborIndex];
        return kEdgeDeltas[neighborIndex - 12];
    }

    static bool IsCellViable(uint3 cellCoordinates, uint3 gridDimensions)
    {
        return true;
    }

    static float3 ComputeWorldPosition(int3 cellCoord, float meshScaleRatio, float3 centeringOffset)
    {
        return edComputeWorldPosition(cellCoord, meshScaleRatio, centeringOffset);
    }

    static bool IntersectCellShape(float3 rayOrigin, float3 rayDirection, float3 cellCenter,
                                   float meshScaleRatio, out float hitDistance, out float3 hitNormal)
    {
        return edIntersect(rayOrigin, rayDirection, cellCenter, meshScaleRatio, hitDistance, hitNormal);
    }

    static bool IsNearCellEdge(float3 localHitPoint, float edgeWidth)
    {
        return edIsNearCellEdge(localHitPoint, edgeWidth);
    }
}
