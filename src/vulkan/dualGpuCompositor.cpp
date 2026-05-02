#include "dualGpuCompositor.h"
#include "../common/logger.h"

#include <cstring>

namespace Cave
{

	DualGpuCompositor::DualGpuCompositor(DeviceContext& gpu0, DeviceContext& gpu1)
		: _gpu0{gpu0}, _gpu1{gpu1}
	{
	}

	DualGpuCompositor::~DualGpuCompositor()
	{
		if (_compositeShaderModule)
			_gpu0.GetDevice().destroyShaderModule(_compositeShaderModule);
	}

	void DualGpuCompositor::Initialize(vk::Extent2D renderExtent, vk::Format colorFormat, uint32_t framesInFlight)
	{
		_renderExtent = renderExtent;
		_colorFormat = colorFormat;
		_framesInFlight = framesInFlight;

		// Staging buffer size: width * height * bytes per pixel (R16G16B16A16Sfloat = 8 bytes)
		uint32_t bytesPerPixel = 8; // R16G16B16A16Sfloat
		_stagingBufferSize = static_cast<uint64_t>(renderExtent.width) * renderExtent.height * bytesPerPixel;

		_gpu1ReadbackStaging = std::make_shared<Buffer>(_gpu1, _stagingBufferSize,
			vk::BufferUsageFlagBits::eTransferDst,
			VMA_MEMORY_USAGE_AUTO,
			VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
			false, 0, "DualGpuCompositor GPU1 Readback Staging");

		_gpu0UploadStaging = std::make_shared<Buffer>(_gpu0, _stagingBufferSize,
			vk::BufferUsageFlagBits::eTransferSrc,
			VMA_MEMORY_USAGE_AUTO,
			VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
			false, 0, "DualGpuCompositor GPU0 Upload Staging");

		// GPU0-local images for GPU1's transferred color (one per frame in flight)
		_gpu1ColorOnGpu0.resize(framesInFlight);
		_compositeOutputImages.resize(framesInFlight);
		for (uint32_t frameIndex = 0; frameIndex < framesInFlight; frameIndex++)
		{
			_gpu1ColorOnGpu0[frameIndex] = std::make_shared<Image>(
				_gpu0,
				colorFormat,
				1,                                                                         // arrayCount
				1,                                                                         // mipLevels
				vk::Extent3D(renderExtent.width, renderExtent.height, 1),                  // extent
				vk::ImageTiling::eOptimal,                                                 // imageTiling
				vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,   // imageUsageFlags
				vk::MemoryPropertyFlagBits::eDeviceLocal,                                  // memoryPropertyFlags
				vk::ImageCreateFlags(),                                                    // imageCreateFlags
				vk::ImageAspectFlagBits::eColor,                                           // imageAspectFlags
				vk::ImageViewType::e2D,                                                    // imageViewType
				vk::SharingMode::eExclusive,                                               // sharingMode
				std::vector<uint32_t>{}                                                    // queueFamilyIndices
			);

			_compositeOutputImages[frameIndex] = std::make_shared<Image>(
				_gpu0,
				colorFormat,
				1,                                                                                              // arrayCount
				1,                                                                                              // mipLevels
				vk::Extent3D(renderExtent.width, renderExtent.height, 1),                                       // extent
				vk::ImageTiling::eOptimal,                                                                      // imageTiling
				vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,                        // imageUsageFlags
				vk::MemoryPropertyFlagBits::eDeviceLocal,                                                       // memoryPropertyFlags
				vk::ImageCreateFlags(),                                                                         // imageCreateFlags
				vk::ImageAspectFlagBits::eColor,                                                                // imageAspectFlags
				vk::ImageViewType::e2D,                                                                         // imageViewType
				vk::SharingMode::eExclusive,                                                                    // sharingMode
				std::vector<uint32_t>{}                                                                         // queueFamilyIndices
			);
		}

		// Command pools, fences, and pre-allocated command buffers
		_gpu1TransferCommandPool = _gpu1.CreateCommandPool(_gpu1.GetQueueFamilies().GraphicsFamily.Index.value());
		_gpu1TransferFence = _gpu1.CreateFence();
		_gpu1TransferCommandBuffer = _gpu1.CreateCommandBuffer(_gpu1TransferCommandPool);
		_gpu0CompositeCommandPool = _gpu0.CreateCommandPool(_gpu0.GetQueueFamilies().ComputeFamily.Index.value());
		_gpu0CompositeFence = _gpu0.CreateFence();
		_gpu0CompositeCommandBuffer = _gpu0.CreateCommandBuffer(_gpu0CompositeCommandPool);

		// Compile composite shader
		std::map<std::string, std::string> defines = {};
		_compositeShaderModule = _slangCompiler.Compile(_gpu0, "shaders/slang", "composite", "compositeMain", defines);

		LOG_INFO("DualGpuCompositor initialized: {}x{}, format={}, staging={:.1f}MB",
			renderExtent.width, renderExtent.height, vk::to_string(colorFormat),
			_stagingBufferSize / (1024.0 * 1024.0));
	}

	void DualGpuCompositor::SetupDescriptors(const std::vector<std::shared_ptr<Image>>& gpu0ColorAttachments)
	{
		_gpu0ColorAttachments = gpu0ColorAttachments;
		_compositeDescriptor = std::make_shared<Descriptor>(_gpu0);
		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			// Binding 0: GPU0 color attachment (from RayMarchRenderSystem, stable across ticks)
			_compositeDescriptor->BindImageToDescriptorSet(
				0, vk::DescriptorType::eStorageImage,
				vk::ShaderStageFlagBits::eCompute, gpu0ColorAttachments[frameIndex]);

			// Binding 1: GPU1 color (transferred to GPU0, owned by compositor)
			_compositeDescriptor->BindImageToDescriptorSet(
				1, vk::DescriptorType::eStorageImage,
				vk::ShaderStageFlagBits::eCompute, _gpu1ColorOnGpu0[frameIndex]);

			// Binding 2: composite output (owned by compositor)
			_compositeDescriptor->BindImageToDescriptorSet(
				2, vk::DescriptorType::eStorageImage,
				vk::ShaderStageFlagBits::eCompute, _compositeOutputImages[frameIndex]);
		}
		_compositeDescriptor->Build();

		auto compositeShader = std::make_shared<Shader>(_gpu0, _compositeShaderModule);
		_compositePipeline = std::make_unique<ComputePipeline>(_gpu0, compositeShader);
		_compositePipeline->AddPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(CompositePushConstants));
		_compositePipeline->AddDescriptorSet(0, _compositeDescriptor);
		_compositePipeline->Build();

		LOG_INFO("DualGpuCompositor: descriptors and pipeline built (framesInFlight={})", _framesInFlight);
	}

	void DualGpuCompositor::TransferGpu1ImageToGpu0(uint32_t frameIndex, std::shared_ptr<Image> gpu1ColorAttachment)
	{
		// Step 1: Copy GPU1's color attachment to GPU1 staging buffer
		// The attachment is already in eTransferSrcOptimal from the render pass
		_gpu1.GetDevice().waitForFences(_gpu1TransferFence, true, UINT64_MAX);
		_gpu1.GetDevice().resetFences(_gpu1TransferFence);

		vk::CommandBuffer gpu1CommandBuffer = _gpu1TransferCommandBuffer;
		gpu1CommandBuffer.reset();
		gpu1CommandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

		vk::BufferImageCopy copyRegion = vk::BufferImageCopy(
			0,                                                                          // bufferOffset
			0,                                                                          // bufferRowLength
			0,                                                                          // bufferImageHeight
			vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),       // imageSubresource
			vk::Offset3D(0, 0, 0),                                                      // imageOffset
			vk::Extent3D(_renderExtent.width, _renderExtent.height, 1)                   // imageExtent
		);
		gpu1CommandBuffer.copyImageToBuffer(
			*gpu1ColorAttachment->GetImage(),
			vk::ImageLayout::eTransferSrcOptimal,
			_gpu1ReadbackStaging->GetBuffer(),
			copyRegion);

		vk::MemoryBarrier memoryBarrier = vk::MemoryBarrier(
			vk::AccessFlagBits::eTransferWrite,   // srcAccessMask
			vk::AccessFlagBits::eHostRead          // dstAccessMask
		);
		gpu1CommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer,  // srcStageMask
			vk::PipelineStageFlagBits::eHost,      // dstStageMask
			{}, memoryBarrier, {}, {});

		gpu1CommandBuffer.end();
		vk::SubmitInfo submitInfo({}, {}, {}, 1, &gpu1CommandBuffer, 0, {});
		_gpu1.GetGraphicsQueue().submit(submitInfo, _gpu1TransferFence);
		_gpu1.GetDevice().waitForFences(_gpu1TransferFence, true, UINT64_MAX);

		// Step 2: Host memcpy GPU1 staging → GPU0 staging
		void* gpu1Mapped = _gpu1ReadbackStaging->GetVmaAllocationInfo().pMappedData;
		void* gpu0Mapped = _gpu0UploadStaging->GetVmaAllocationInfo().pMappedData;
		std::memcpy(gpu0Mapped, gpu1Mapped, _stagingBufferSize);

		// Steps 3+4 merged: GPU0 upload + composite in a single command buffer + fence wait.
		// Replaces two EndSingleTimeCommands (each calls queue.waitIdle) with one fence wait.
		_gpu0.GetDevice().waitForFences(_gpu0CompositeFence, true, UINT64_MAX);
		_gpu0.GetDevice().resetFences(_gpu0CompositeFence);

		vk::CommandBuffer gpu0CommandBuffer = _gpu0CompositeCommandBuffer;
		gpu0CommandBuffer.reset();
		gpu0CommandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

		// --- Upload: staging → gpu1ColorOnGpu0 ---

		// Transition gpu1ColorOnGpu0 to transfer dst
		vk::ImageMemoryBarrier barrierToDst = vk::ImageMemoryBarrier(
			vk::AccessFlagBits::eNone,                    // srcAccessMask
			vk::AccessFlagBits::eTransferWrite,            // dstAccessMask
			vk::ImageLayout::eUndefined,                   // oldLayout
			vk::ImageLayout::eTransferDstOptimal,          // newLayout
			VK_QUEUE_FAMILY_IGNORED,                       // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                       // dstQueueFamilyIndex
			*_gpu1ColorOnGpu0[frameIndex]->GetImage(),     // image
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1) // subresourceRange
		);
		gpu0CommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
			{}, {}, {}, barrierToDst);

		gpu0CommandBuffer.copyBufferToImage(
			_gpu0UploadStaging->GetBuffer(),
			*_gpu1ColorOnGpu0[frameIndex]->GetImage(),
			vk::ImageLayout::eTransferDstOptimal,
			copyRegion);

		// Transition gpu1ColorOnGpu0 to general for compute read
		vk::ImageMemoryBarrier gpu1ImageToGeneral = vk::ImageMemoryBarrier(
			vk::AccessFlagBits::eTransferWrite,            // srcAccessMask
			vk::AccessFlagBits::eShaderRead,               // dstAccessMask
			vk::ImageLayout::eTransferDstOptimal,          // oldLayout
			vk::ImageLayout::eGeneral,                     // newLayout
			VK_QUEUE_FAMILY_IGNORED,                       // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                       // dstQueueFamilyIndex
			*_gpu1ColorOnGpu0[frameIndex]->GetImage(),     // image
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1) // subresourceRange
		);

		// --- Composite: read gpu0Color + gpu1Color, write output ---

		// Transition GPU0's color attachment from eTransferSrcOptimal to eGeneral for compute read
		vk::ImageMemoryBarrier gpu0ColorToGeneral = vk::ImageMemoryBarrier(
			vk::AccessFlagBits::eTransferRead,             // srcAccessMask
			vk::AccessFlagBits::eShaderRead,               // dstAccessMask
			vk::ImageLayout::eTransferSrcOptimal,          // oldLayout
			vk::ImageLayout::eGeneral,                     // newLayout
			VK_QUEUE_FAMILY_IGNORED,                       // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                       // dstQueueFamilyIndex
			*_gpu0ColorAttachments[frameIndex]->GetImage(), // image
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1) // subresourceRange
		);

		// Transition composite output to eGeneral for compute write
		vk::ImageMemoryBarrier outputToGeneral = vk::ImageMemoryBarrier(
			vk::AccessFlagBits::eNone,                           // srcAccessMask
			vk::AccessFlagBits::eShaderWrite,                    // dstAccessMask
			vk::ImageLayout::eUndefined,                         // oldLayout
			vk::ImageLayout::eGeneral,                           // newLayout
			VK_QUEUE_FAMILY_IGNORED,                             // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                             // dstQueueFamilyIndex
			*_compositeOutputImages[frameIndex]->GetImage(),     // image
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1) // subresourceRange
		);

		// Batch all three barriers (upload→general, gpu0Color→general, output→general)
		std::array<vk::ImageMemoryBarrier, 3> preCompositeBarriers = {
			gpu1ImageToGeneral, gpu0ColorToGeneral, outputToGeneral
		};
		gpu0CommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eComputeShader,
			{}, {}, {}, preCompositeBarriers);

		// Bind pipeline and dispatch composite
		_compositePipeline->Bind(gpu0CommandBuffer, vk::PipelineBindPoint::eCompute,
			static_cast<uint8_t>(frameIndex), Pipeline::DescriptorOption{frameIndex});

		CompositePushConstants compositePushConstants{};
		compositePushConstants.width = _renderExtent.width;
		compositePushConstants.height = _renderExtent.height;
		gpu0CommandBuffer.pushConstants<CompositePushConstants>(
			_compositePipeline->GetPipelineLayout(),
			vk::ShaderStageFlagBits::eCompute,
			0,
			compositePushConstants);

		uint32_t dispatchX = (_renderExtent.width + 7) / 8;
		uint32_t dispatchY = (_renderExtent.height + 7) / 8;
		gpu0CommandBuffer.dispatch(dispatchX, dispatchY, 1);

		// Transition composite output to eTransferSrcOptimal for presentation blit
		vk::ImageMemoryBarrier outputToTransfer = vk::ImageMemoryBarrier(
			vk::AccessFlagBits::eShaderWrite,                    // srcAccessMask
			vk::AccessFlagBits::eTransferRead,                   // dstAccessMask
			vk::ImageLayout::eGeneral,                           // oldLayout
			vk::ImageLayout::eTransferSrcOptimal,                // newLayout
			VK_QUEUE_FAMILY_IGNORED,                             // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                             // dstQueueFamilyIndex
			*_compositeOutputImages[frameIndex]->GetImage(),     // image
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1) // subresourceRange
		);
		gpu0CommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
			{}, {}, {}, outputToTransfer);

		gpu0CommandBuffer.end();

		// Single fence-based submit (replaces 2x queue.waitIdle)
		vk::SubmitInfo gpu0SubmitInfo = vk::SubmitInfo(
			0,                   // waitSemaphoreCount
			{},                  // pWaitSemaphores
			{},                  // pWaitDstStageMask
			1,                   // commandBufferCount
			&gpu0CommandBuffer,  // pCommandBuffers
			0,                   // signalSemaphoreCount
			{}                   // pSignalSemaphores
		);
		_gpu0.GetComputeQueue().submit(gpu0SubmitInfo, _gpu0CompositeFence);
		_gpu0.GetDevice().waitForFences(_gpu0CompositeFence, true, UINT64_MAX);
	}

	void DualGpuCompositor::RunCompositePass(uint32_t frameIndex)
	{
		// Merged into TransferGpu1ImageToGpu0 (single command buffer submission)
	}

	vk::Image DualGpuCompositor::Composite(uint32_t frameIndex, std::shared_ptr<Image> gpu1ColorAttachment)
	{
		// TransferGpu1ImageToGpu0 now includes the composite pass in a single command buffer
		TransferGpu1ImageToGpu0(frameIndex, gpu1ColorAttachment);
		return *_compositeOutputImages[frameIndex]->GetImage();
	}

	vk::Image DualGpuCompositor::GetOutputImage(uint32_t frameIndex) const
	{
		return *_compositeOutputImages[frameIndex]->GetImage();
	}

} // namespace Cave
