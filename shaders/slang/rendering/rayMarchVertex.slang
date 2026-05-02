// Ray march vertex shader — transforms bounding box vertices to world space.
// The bounding box encloses the simulation volume; the fragment shader ray marches through it.

struct CameraUBO
{
    float4x4 model;
    float4x4 view;
    float4x4 projection;
    float4 lightDirection;
    float4 lightingParameters;
    float4 cameraPosition;
    float4 cullingPlanes[4];
    uint4 cullingControl;
};

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

[vk::binding(0, 0)] ConstantBuffer<CameraUBO> cameraData;
[[vk::push_constant]] RayMarchConstants rayMarch;

struct VertexInput
{
    float4 position : POSITION;
};

struct VertexOutput
{
    [[vk::location(0)]] float3 worldPosition : FragmentWorldPosition;
    float4 clipPosition : SV_Position;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;

    float3 volumeMin = float3(rayMarch.volumeMinX, rayMarch.volumeMinY, rayMarch.volumeMinZ);
    float3 volumeMax = float3(rayMarch.volumeMaxX, rayMarch.volumeMaxY, rayMarch.volumeMaxZ);
    float3 volumeCenter = (volumeMin + volumeMax) * 0.5;
    float3 volumeHalfExtent = (volumeMax - volumeMin) * 0.5;
    float3 worldPosition = input.position.xyz * volumeHalfExtent + volumeCenter;

    output.worldPosition = worldPosition;
    output.clipPosition = mul(cameraData.projection, mul(cameraData.view, mul(cameraData.model, float4(worldPosition, 1.0))));

    return output;
}
