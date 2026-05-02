#pragma once

#include <memory>
#include <cstdint>
#include <vulkan/vulkan.hpp>

#include "deviceContext.h"
#include "buffer.h"
#include "image.h"
#include "slangCompiler.h"
#include "descriptors.h"
#include "shader.h"
#include "pipelines/computePipeline.h"

namespace Cave
{

	struct CompositePushConstants
	{
		uint32_t width;
		uint32_t height;
	};

	// Composites two GPU-rendered images by depth comparison for dual-GPU rendering.
	// Transfers GPU1's color image to GPU0 via host staging, then runs a compute
	// shader on GPU0 that picks the closer fragment per pixel (depth encoded in alpha).
	class DualGpuCompositor
	{
	public:
		DualGpuCompositor(DeviceContext& gpu0, DeviceContext& gpu1);
		~DualGpuCompositor();

		DualGpuCompositor(const DualGpuCompositor&) = delete;
		DualGpuCompositor& operator=(const DualGpuCompositor&) = delete;

		void Initialize(vk::Extent2D renderExtent, vk::Format colorFormat, uint32_t framesInFlight);

		// Binds the GPU0 color attachments (from RayMarchRenderSystem) and builds the
		// composite descriptor + pipeline. Must be called once after Initialize and
		// after both render systems are created.
		void SetupDescriptors(const std::vector<std::shared_ptr<Image>>& gpu0ColorAttachments);

		// Composites gpu0 and gpu1 rendered images. gpu1's image is transferred to GPU0
		// via host staging, then a compute shader picks the closer fragment per pixel.
		// Returns the composited output image for the given frame.
		vk::Image Composite(uint32_t frameIndex, std::shared_ptr<Image> gpu1ColorAttachment);

		vk::Image GetOutputImage(uint32_t frameIndex) const;

	private:
		DeviceContext& _gpu0;
		DeviceContext& _gpu1;

		vk::Extent2D _renderExtent;
		vk::Format _colorFormat;
		uint32_t _framesInFlight = 0;

		// Staging buffers for GPU1 → GPU0 image transfer
		std::shared_ptr<Buffer> _gpu1ReadbackStaging;
		std::shared_ptr<Buffer> _gpu0UploadStaging;
		uint64_t _stagingBufferSize = 0;

		// GPU0-local image holding GPU1's transferred render result (per frame)
		std::vector<std::shared_ptr<Image>> _gpu1ColorOnGpu0;

		// Composite output image (per frame)
		std::vector<std::shared_ptr<Image>> _compositeOutputImages;

		// GPU0 color attachments (stored for image transitions in RunCompositePass)
		std::vector<std::shared_ptr<Image>> _gpu0ColorAttachments;

		// Composite compute pipeline
		SlangCompiler _slangCompiler;
		vk::ShaderModule _compositeShaderModule;
		std::unique_ptr<ComputePipeline> _compositePipeline;
		std::shared_ptr<Descriptor> _compositeDescriptor;

		// Command resources (pre-allocated command buffers, reset + re-record each tick)
		vk::CommandPool _gpu1TransferCommandPool;
		vk::Fence _gpu1TransferFence;
		vk::CommandBuffer _gpu1TransferCommandBuffer;
		vk::CommandPool _gpu0CompositeCommandPool;
		vk::Fence _gpu0CompositeFence;
		vk::CommandBuffer _gpu0CompositeCommandBuffer;

		void TransferGpu1ImageToGpu0(uint32_t frameIndex, std::shared_ptr<Image> gpu1ColorAttachment);
		void RunCompositePass(uint32_t frameIndex);
	};

} // namespace Cave
