// Shape interface — defines the neighborhood topology for a cellular automata shape.
// Each shape provides its neighbor deltas and a cell viability function.
// The simulation and rendering shaders are generic over this interface.

interface IShape
{
    // Number of neighbor cells to check for birth/survival rules
    static const int NeighborCount;

    // Number of distinct neighbor configurations (e.g., 1 for Cube, 2 for ERD)
    static const int NeighborConfigurationCount;

    // Maximum absolute reach per axis (for skip grid scatter-write radius)
    static const int MaxReachX;
    static const int MaxReachY;
    static const int MaxReachZ;

    // Get the neighbor configuration index for a cell based on its grid position.
    // Cube returns 0 for all cells. Shapes with alternating rows return e.g. cellCoordinates.y % 2.
    static int GetNeighborConfigurationIndex(int3 cellCoordinates);

    // Get the delta offset to the i-th neighbor for a given configuration
    static int3 GetNeighborDelta(int neighborConfigurationIndex, int neighborIndex);

    // Returns true if the cell at the given coordinates is viable for this shape.
    // Non-viable cells are always dead (e.g., cells outside a sphere in a spherical shape).
    static bool IsCellViable(uint3 cellCoordinates, uint3 gridDimensions);

    // Compute the world-space center of a cell, accounting for shape-specific spacing
    // (e.g. ED rows are offset on alternating Y) and the global mesh scale + centering offset.
    static float3 ComputeWorldPosition(int3 cellCoordinates, float meshScaleRatio, float3 centeringOffset);

    // Ray vs cell-shape intersection. Returns true on hit, writing entry distance and surface normal.
    // hitDistance is the ray parameter (== world-space distance since rays are normalized).
    static bool IntersectCellShape(float3 rayOrigin, float3 rayDirection, float3 cellCenter,
                                   float meshScaleRatio, out float hitDistance, out float3 hitNormal);

    // True if the local hit point (relative to cell center, in mesh-scale units) is near a
    // wireframe edge. Used by the debug-color overlay.
    static bool IsNearCellEdge(float3 localHitPoint, float edgeWidth);
}
