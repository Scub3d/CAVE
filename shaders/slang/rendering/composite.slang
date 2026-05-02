// Dual-GPU compositing shader — combines two rendered images by depth comparison.
// Each input image uses alpha channel for clip-space depth (0 = near, 1 = far).
// Alpha = 0 means no hit (background). Closest fragment wins per pixel.

[vk::binding(0, 0)] [vk::image_format("rgba16f")] RWTexture2D<float4> gpu0Color;
[vk::binding(1, 0)] [vk::image_format("rgba16f")] RWTexture2D<float4> gpu1Color;
[vk::binding(2, 0)] [vk::image_format("rgba16f")] RWTexture2D<float4> compositeOutput;

struct CompositePushConstants
{
    uint width;
    uint height;
};
[[vk::push_constant]] CompositePushConstants pushConstants;

[shader("compute")]
[numthreads(8, 8, 1)]
void compositeMain(uint3 globalID : SV_DispatchThreadID)
{
    if (globalID.x >= pushConstants.width || globalID.y >= pushConstants.height)
        return;

    int2 pixel = int2(globalID.xy);
    float4 color0 = gpu0Color[pixel];
    float4 color1 = gpu1Color[pixel];

    bool hit0 = color0.a > 0.0;
    bool hit1 = color1.a > 0.0;

    if (hit0 && hit1)
    {
        // Both GPUs have hits — pick the closer one (smaller depth = closer)
        compositeOutput[pixel] = (color0.a <= color1.a)
            ? float4(color0.rgb, 1.0)
            : float4(color1.rgb, 1.0);
    }
    else if (hit0)
    {
        compositeOutput[pixel] = float4(color0.rgb, 1.0);
    }
    else if (hit1)
    {
        compositeOutput[pixel] = float4(color1.rgb, 1.0);
    }
    else
    {
        compositeOutput[pixel] = float4(0.0, 0.0, 0.0, 1.0);
    }
}
