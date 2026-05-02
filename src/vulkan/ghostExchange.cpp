#include "ghostExchange.h"
#include "../common/logger.h"

#include <cstring>

namespace Cave
{

	GhostExchange::GhostExchange(DeviceContext& gpu0, DeviceContext& gpu1)
		: _gpu0{gpu0}, _gpu1{gpu1}
	{
	}

	GhostExchange::~GhostExchange()
	{
		if (_gpu0ExtractShaderModule)
			_gpu0.GetDevice().destroyShaderModule(_gpu0ExtractShaderModule);
		if (_gpu0InsertShaderModule)
			_gpu0.GetDevice().destroyShaderModule(_gpu0InsertShaderModule);
		if (_gpu1ExtractShaderModule)
			_gpu1.GetDevice().destroyShaderModule(_gpu1ExtractShaderModule);
		if (_gpu1InsertShaderModule)
			_gpu1.GetDevice().destroyShaderModule(_gpu1InsertShaderModule);
	}

	void GhostExchange::Initialize(uint32_t gridDimensionX, uint32_t gridDimensionY, uint32_t gridDimensionZ,
									uint32_t zMid, uint32_t ghostDepth, bool wrapEnabled,
									CellStateStorageMode storageMode)
	{
		_storageMode = storageMode;
		_ghostDepth = ghostDepth;
		_imageWidth = (gridDimensionX + 1) / 2;
		_gridDimensionY = gridDimensionY;
		_gridDimensionZ = gridDimensionZ;
		_zMid = zMid;
		_wrapEnabled = wrapEnabled;

		_gpu0LocalZSize = zMid + 2 * ghostDepth;
		_gpu1LocalZSize = (gridDimensionZ - zMid) + 2 * ghostDepth;

		// Staging buffer size depends on storage mode:
		//   Buffer mode: uint32_t per texel (tiled Morton layout, 4 bytes)
		//   Image mode:  R8Uint per texel (1 byte, packed 4-bit format)
		uint32_t bytesPerTexel = (storageMode == CellStateStorageMode::Image) ? 1 : 4;
		_stagingBufferSize = static_cast<uint64_t>(ghostDepth) * _imageWidth * gridDimensionY * bytesPerTexel;

		_gpu0Staging = std::make_shared<Buffer>(_gpu0, _stagingBufferSize,
			vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
			VMA_MEMORY_USAGE_AUTO,
			VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
			false, 0, "GhostExchange GPU0 Staging");
		_gpu1Staging = std::make_shared<Buffer>(_gpu1, _stagingBufferSize,
			vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
			VMA_MEMORY_USAGE_AUTO,
			VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
			false, 0, "GhostExchange GPU1 Staging");

		_gpu0CommandPool = _gpu0.CreateCommandPool(_gpu0.GetQueueFamilies().ComputeFamily.Index.value());
		_gpu1CommandPool = _gpu1.CreateCommandPool(_gpu1.GetQueueFamilies().ComputeFamily.Index.value());
		_gpu0Fence = _gpu0.CreateFence();
		_gpu1Fence = _gpu1.CreateFence();
		_gpu0CommandBuffer = _gpu0.CreateCommandBuffer(_gpu0CommandPool);
		_gpu1CommandBuffer = _gpu1.CreateCommandBuffer(_gpu1CommandPool);

		if (storageMode == CellStateStorageMode::Buffer)
			CompileShaders();

		LOG_INFO("GhostExchange initialized: mode={}, ghostDepth={}, imageWidth={}, gridDimY={}, gridDimZ={}, zMid={}, wrap={}, staging={:.1f}MB",
			storageMode == CellStateStorageMode::Image ? "image" : "buffer",
			ghostDepth, _imageWidth, gridDimensionY, gridDimensionZ, zMid, wrapEnabled,
			_stagingBufferSize / (1024.0 * 1024.0));
	}

	void GhostExchange::CompileShaders()
	{
		std::map<std::string, std::string> defines = {
			{"IMAGE_WIDTH", std::to_string(_imageWidth)},
			{"GRID_DIM_Y", std::to_string(_gridDimensionY)}
		};

		_gpu0ExtractShaderModule = _slangCompiler.Compile(_gpu0, "shaders/slang", "ghostCopy", "extractMain", defines);
		_gpu0InsertShaderModule = _slangCompiler.Compile(_gpu0, "shaders/slang", "ghostCopy", "insertMain", defines);
		_gpu1ExtractShaderModule = _slangCompiler.Compile(_gpu1, "shaders/slang", "ghostCopy", "extractMain", defines);
		_gpu1InsertShaderModule = _slangCompiler.Compile(_gpu1, "shaders/slang", "ghostCopy", "insertMain", defines);

		LOG_DEBUG("GhostExchange: compiled extract/insert shaders for both GPUs");
	}

	// ==============================
	// Buffer mode setup
	// ==============================

	void GhostExchange::SetupForBufferMode(
		std::shared_ptr<Buffer> gpu0CellBuffer0, std::shared_ptr<Buffer> gpu0CellBuffer1,
		std::shared_ptr<Buffer> gpu1CellBuffer0, std::shared_ptr<Buffer> gpu1CellBuffer1,
		glm::ivec3 gpu0NumBlocks, glm::ivec3 gpu1NumBlocks)
	{
		_gpu0CellBuffers[0] = gpu0CellBuffer0;
		_gpu0CellBuffers[1] = gpu0CellBuffer1;
		_gpu1CellBuffers[0] = gpu1CellBuffer0;
		_gpu1CellBuffers[1] = gpu1CellBuffer1;
		_gpu0NumBlocks = gpu0NumBlocks;
		_gpu1NumBlocks = gpu1NumBlocks;

		auto buildDescriptor = [](DeviceContext& gpu,
								  std::shared_ptr<Buffer> cellBuffer0, std::shared_ptr<Buffer> cellBuffer1,
								  std::shared_ptr<Buffer> staging) -> std::shared_ptr<Descriptor>
		{
			auto descriptor = std::make_shared<Descriptor>(gpu);
			for (int frame = 0; frame < 2; frame++)
			{
				auto& cellBuffer = (frame == 0) ? cellBuffer0 : cellBuffer1;
				descriptor->BindBufferToDescriptorSet(0, vk::DescriptorType::eStorageBuffer,
					vk::ShaderStageFlagBits::eCompute, cellBuffer);
				descriptor->BindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer,
					vk::ShaderStageFlagBits::eCompute, staging);
			}
			descriptor->Build();
			return descriptor;
		};

		_gpu0ExtractDescriptor = buildDescriptor(_gpu0, gpu0CellBuffer0, gpu0CellBuffer1, _gpu0Staging);
		_gpu0InsertDescriptor  = buildDescriptor(_gpu0, gpu0CellBuffer0, gpu0CellBuffer1, _gpu0Staging);
		_gpu1ExtractDescriptor = buildDescriptor(_gpu1, gpu1CellBuffer0, gpu1CellBuffer1, _gpu1Staging);
		_gpu1InsertDescriptor  = buildDescriptor(_gpu1, gpu1CellBuffer0, gpu1CellBuffer1, _gpu1Staging);

		auto buildPipeline = [](DeviceContext& gpu, vk::ShaderModule shaderModule,
								std::shared_ptr<Descriptor> descriptor) -> std::unique_ptr<ComputePipeline>
		{
			auto shader = std::make_shared<Shader>(gpu, shaderModule);
			auto pipeline = std::make_unique<ComputePipeline>(gpu, shader);
			pipeline->AddDescriptorSet(0, descriptor);
			pipeline->AddPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(GhostCopyPushConstants));
			pipeline->Build();
			return pipeline;
		};

		_gpu0ExtractPipeline = buildPipeline(_gpu0, _gpu0ExtractShaderModule, _gpu0ExtractDescriptor);
		_gpu0InsertPipeline  = buildPipeline(_gpu0, _gpu0InsertShaderModule, _gpu0InsertDescriptor);
		_gpu1ExtractPipeline = buildPipeline(_gpu1, _gpu1ExtractShaderModule, _gpu1ExtractDescriptor);
		_gpu1InsertPipeline  = buildPipeline(_gpu1, _gpu1InsertShaderModule, _gpu1InsertDescriptor);

		LOG_INFO("GhostExchange: buffer mode — pre-built 4 descriptors + 4 pipelines");
	}

	// ==============================
	// Image mode setup
	// ==============================

	void GhostExchange::SetupForImageMode(
		std::shared_ptr<Image> gpu0CellImage0, std::shared_ptr<Image> gpu0CellImage1,
		std::shared_ptr<Image> gpu1CellImage0, std::shared_ptr<Image> gpu1CellImage1)
	{
		_gpu0CellImages[0] = gpu0CellImage0;
		_gpu0CellImages[1] = gpu0CellImage1;
		_gpu1CellImages[0] = gpu1CellImage0;
		_gpu1CellImages[1] = gpu1CellImage1;

		LOG_INFO("GhostExchange: image mode — no compute shaders needed (direct vkCmdCopy)");
	}

	// ==============================
	// Buffer mode transfer (compute shader extract/insert)
	// ==============================

	void GhostExchange::DispatchAndWait(DeviceContext& gpu, vk::CommandPool commandPool, vk::Fence fence,
										ComputePipeline& pipeline, uint32_t descriptorOption,
										GhostCopyPushConstants pushConstants, uint32_t sliceCount)
	{
		gpu.GetDevice().waitForFences(fence, true, UINT64_MAX);
		gpu.GetDevice().resetFences(fence);

		vk::CommandBuffer commandBuffer = (&gpu == &_gpu0) ? _gpu0CommandBuffer : _gpu1CommandBuffer;
		commandBuffer.reset();
		commandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

		pipeline.Bind(commandBuffer, vk::PipelineBindPoint::eCompute, 0,
			Pipeline::DescriptorOption{descriptorOption});
		commandBuffer.pushConstants<GhostCopyPushConstants>(
			pipeline.GetPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, pushConstants);

		uint32_t dispatchX = (_imageWidth + 7) / 8;
		uint32_t dispatchY = (_gridDimensionY + 7) / 8;
		commandBuffer.dispatch(dispatchX, dispatchY, sliceCount);

		vk::MemoryBarrier memoryBarrier = vk::MemoryBarrier(
			vk::AccessFlagBits::eShaderWrite,                                          // srcAccessMask
			vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eShaderRead            // dstAccessMask
		);
		commandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eComputeShader,
			vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eComputeShader,
			{}, memoryBarrier, {}, {});

		commandBuffer.end();

		vk::SubmitInfo submitInfo({}, {}, {}, 1, &commandBuffer, 0, {});
		gpu.GetComputeQueue().submit(submitInfo, fence);
		gpu.GetDevice().waitForFences(fence, true, UINT64_MAX);
	}

	void GhostExchange::BufferTransfer(DeviceContext& sourceGpu, vk::CommandPool sourceCommandPool, vk::Fence sourceFence,
								ComputePipeline& sourceExtractPipeline, uint32_t sourceDescriptorOption,
								glm::ivec3 sourceNumBlocks, int32_t sourceZStart,
								DeviceContext& destGpu, vk::CommandPool destCommandPool, vk::Fence destFence,
								ComputePipeline& destInsertPipeline, uint32_t destDescriptorOption,
								glm::ivec3 destNumBlocks, int32_t destZStart)
	{
		GhostCopyPushConstants extractPushConstants = {
			sourceZStart, static_cast<int32_t>(_imageWidth), static_cast<int32_t>(_gridDimensionY),
			sourceNumBlocks.x, sourceNumBlocks.y, sourceNumBlocks.z
		};
		DispatchAndWait(sourceGpu, sourceCommandPool, sourceFence,
						sourceExtractPipeline, sourceDescriptorOption, extractPushConstants, _ghostDepth);

		std::shared_ptr<Buffer>& sourceStaging = (&sourceGpu == &_gpu0) ? _gpu0Staging : _gpu1Staging;
		std::shared_ptr<Buffer>& destStaging = (&destGpu == &_gpu0) ? _gpu0Staging : _gpu1Staging;
		std::memcpy(destStaging->GetVmaAllocationInfo().pMappedData,
					sourceStaging->GetVmaAllocationInfo().pMappedData, _stagingBufferSize);

		GhostCopyPushConstants insertPushConstants = {
			destZStart, static_cast<int32_t>(_imageWidth), static_cast<int32_t>(_gridDimensionY),
			destNumBlocks.x, destNumBlocks.y, destNumBlocks.z
		};
		DispatchAndWait(destGpu, destCommandPool, destFence,
						destInsertPipeline, destDescriptorOption, insertPushConstants, _ghostDepth);
	}

	// ==============================
	// Image mode transfer (direct vkCmdCopy)
	// ==============================

	void GhostExchange::ImageTransfer(DeviceContext& sourceGpu, vk::CommandPool sourceCommandPool, vk::Fence sourceFence,
									   std::shared_ptr<Image> sourceImage, int32_t sourceZStart,
									   DeviceContext& destGpu, vk::CommandPool destCommandPool, vk::Fence destFence,
									   std::shared_ptr<Image> destImage, int32_t destZStart)
	{
		vk::ImageSubresourceRange subresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
		vk::ImageSubresourceLayers subresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);

		vk::BufferImageCopy copyRegion = vk::BufferImageCopy(
			0,                                                                                 // bufferOffset
			0,                                                                                 // bufferRowLength (tightly packed)
			0,                                                                                 // bufferImageHeight (tightly packed)
			subresourceLayers,                                                                 // imageSubresource
			vk::Offset3D(0, 0, 0),                                                            // imageOffset (set per step)
			vk::Extent3D(_imageWidth, _gridDimensionY, _ghostDepth)                            // imageExtent
		);

		std::shared_ptr<Buffer>& sourceStaging = (&sourceGpu == &_gpu0) ? _gpu0Staging : _gpu1Staging;
		std::shared_ptr<Buffer>& destStaging = (&destGpu == &_gpu0) ? _gpu0Staging : _gpu1Staging;

		// Step 1: Copy Z-slices from source image → source staging buffer
		sourceGpu.GetDevice().waitForFences(sourceFence, true, UINT64_MAX);
		sourceGpu.GetDevice().resetFences(sourceFence);

		vk::CommandBuffer sourceCommandBuffer = (&sourceGpu == &_gpu0) ? _gpu0CommandBuffer : _gpu1CommandBuffer;
		sourceCommandBuffer.reset();
		sourceCommandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

		// Transition source image: eGeneral → eTransferSrcOptimal
		vk::ImageMemoryBarrier sourceToTransferSrc = vk::ImageMemoryBarrier(
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, // srcAccessMask
			vk::AccessFlagBits::eTransferRead,                                  // dstAccessMask
			vk::ImageLayout::eGeneral,                                          // oldLayout
			vk::ImageLayout::eTransferSrcOptimal,                               // newLayout
			VK_QUEUE_FAMILY_IGNORED,                                            // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                                            // dstQueueFamilyIndex
			*sourceImage->GetImage(),                                           // image
			subresourceRange                                                    // subresourceRange
		);
		sourceCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
			{}, {}, {}, sourceToTransferSrc);

		vk::BufferImageCopy extractRegion = copyRegion;
		extractRegion.imageOffset = vk::Offset3D(0, 0, sourceZStart);
		sourceCommandBuffer.copyImageToBuffer(
			*sourceImage->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
			sourceStaging->GetBuffer(), extractRegion);

		// Transition source image back: eTransferSrcOptimal → eGeneral
		vk::ImageMemoryBarrier sourceBackToGeneral = vk::ImageMemoryBarrier(
			vk::AccessFlagBits::eTransferRead,                                  // srcAccessMask
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, // dstAccessMask
			vk::ImageLayout::eTransferSrcOptimal,                               // oldLayout
			vk::ImageLayout::eGeneral,                                          // newLayout
			VK_QUEUE_FAMILY_IGNORED,                                            // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                                            // dstQueueFamilyIndex
			*sourceImage->GetImage(),                                           // image
			subresourceRange                                                    // subresourceRange
		);
		sourceCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
			{}, {}, {}, sourceBackToGeneral);

		// Host-read barrier for staging
		vk::MemoryBarrier hostReadBarrier = vk::MemoryBarrier(
			vk::AccessFlagBits::eTransferWrite, // srcAccessMask
			vk::AccessFlagBits::eHostRead        // dstAccessMask
		);
		sourceCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
			{}, hostReadBarrier, {}, {});

		sourceCommandBuffer.end();
		vk::SubmitInfo sourceSubmitInfo({}, {}, {}, 1, &sourceCommandBuffer, 0, {});
		sourceGpu.GetComputeQueue().submit(sourceSubmitInfo, sourceFence);
		sourceGpu.GetDevice().waitForFences(sourceFence, true, UINT64_MAX);

		// Step 2: Host memcpy source staging → dest staging
		std::memcpy(destStaging->GetVmaAllocationInfo().pMappedData,
					sourceStaging->GetVmaAllocationInfo().pMappedData, _stagingBufferSize);

		// Step 3: Copy dest staging buffer → dest image Z-slices
		destGpu.GetDevice().waitForFences(destFence, true, UINT64_MAX);
		destGpu.GetDevice().resetFences(destFence);

		vk::CommandBuffer destCommandBuffer = (&destGpu == &_gpu0) ? _gpu0CommandBuffer : _gpu1CommandBuffer;
		destCommandBuffer.reset();
		destCommandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

		// Transition dest image: eGeneral → eTransferDstOptimal
		vk::ImageMemoryBarrier destToTransferDst = vk::ImageMemoryBarrier(
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, // srcAccessMask
			vk::AccessFlagBits::eTransferWrite,                                  // dstAccessMask
			vk::ImageLayout::eGeneral,                                           // oldLayout
			vk::ImageLayout::eTransferDstOptimal,                                // newLayout
			VK_QUEUE_FAMILY_IGNORED,                                             // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                                             // dstQueueFamilyIndex
			*destImage->GetImage(),                                              // image
			subresourceRange                                                     // subresourceRange
		);
		destCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
			{}, {}, {}, destToTransferDst);

		vk::BufferImageCopy insertRegion = copyRegion;
		insertRegion.imageOffset = vk::Offset3D(0, 0, destZStart);
		destCommandBuffer.copyBufferToImage(
			destStaging->GetBuffer(), *destImage->GetImage(),
			vk::ImageLayout::eTransferDstOptimal, insertRegion);

		// Transition dest image back: eTransferDstOptimal → eGeneral
		vk::ImageMemoryBarrier destBackToGeneral = vk::ImageMemoryBarrier(
			vk::AccessFlagBits::eTransferWrite,                                  // srcAccessMask
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,  // dstAccessMask
			vk::ImageLayout::eTransferDstOptimal,                                // oldLayout
			vk::ImageLayout::eGeneral,                                           // newLayout
			VK_QUEUE_FAMILY_IGNORED,                                             // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                                             // dstQueueFamilyIndex
			*destImage->GetImage(),                                              // image
			subresourceRange                                                     // subresourceRange
		);
		destCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
			{}, {}, {}, destBackToGeneral);

		destCommandBuffer.end();
		vk::SubmitInfo destSubmitInfo({}, {}, {}, 1, &destCommandBuffer, 0, {});
		destGpu.GetComputeQueue().submit(destSubmitInfo, destFence);
		destGpu.GetDevice().waitForFences(destFence, true, UINT64_MAX);
	}

	// ==============================
	// Exchange (dispatches to buffer or image path)
	// ==============================

	void GhostExchange::Exchange(uint32_t frameIndex)
	{
		// GPU0 local layout: [ghost_lo(ghostDepth), owned(zMid), ghost_hi(ghostDepth)]
		// GPU1 local layout: [ghost_lo(ghostDepth), owned(gridZ - zMid), ghost_hi(ghostDepth)]

		if (_storageMode == CellStateStorageMode::Image)
		{
			auto& gpu0Image = _gpu0CellImages[frameIndex];
			auto& gpu1Image = _gpu1CellImages[frameIndex];

			// Transfer 1: GPU0 boundary → GPU1 ghost_lo
			ImageTransfer(_gpu0, _gpu0CommandPool, _gpu0Fence, gpu0Image,
						  static_cast<int32_t>(_zMid),
						  _gpu1, _gpu1CommandPool, _gpu1Fence, gpu1Image,
						  0);

			// Transfer 2: GPU1 boundary → GPU0 ghost_hi
			ImageTransfer(_gpu1, _gpu1CommandPool, _gpu1Fence, gpu1Image,
						  static_cast<int32_t>(_ghostDepth),
						  _gpu0, _gpu0CommandPool, _gpu0Fence, gpu0Image,
						  static_cast<int32_t>(_ghostDepth + _zMid));

			if (_wrapEnabled)
			{
				int32_t gpu1OwnedCount = static_cast<int32_t>(_gridDimensionZ - _zMid);

				// Transfer 3: GPU1 top → GPU0 ghost_lo
				ImageTransfer(_gpu1, _gpu1CommandPool, _gpu1Fence, gpu1Image,
							  static_cast<int32_t>(_ghostDepth) + gpu1OwnedCount - static_cast<int32_t>(_ghostDepth),
							  _gpu0, _gpu0CommandPool, _gpu0Fence, gpu0Image,
							  0);

				// Transfer 4: GPU0 bottom → GPU1 ghost_hi
				ImageTransfer(_gpu0, _gpu0CommandPool, _gpu0Fence, gpu0Image,
							  static_cast<int32_t>(_ghostDepth),
							  _gpu1, _gpu1CommandPool, _gpu1Fence, gpu1Image,
							  static_cast<int32_t>(_ghostDepth) + gpu1OwnedCount);
			}
		}
		else
		{
			// Transfer 1: GPU0 boundary → GPU1 ghost_lo
			BufferTransfer(_gpu0, _gpu0CommandPool, _gpu0Fence,
						   *_gpu0ExtractPipeline, frameIndex, _gpu0NumBlocks,
						   static_cast<int32_t>(_zMid),
						   _gpu1, _gpu1CommandPool, _gpu1Fence,
						   *_gpu1InsertPipeline, frameIndex, _gpu1NumBlocks,
						   0);

			// Transfer 2: GPU1 boundary → GPU0 ghost_hi
			BufferTransfer(_gpu1, _gpu1CommandPool, _gpu1Fence,
						   *_gpu1ExtractPipeline, frameIndex, _gpu1NumBlocks,
						   static_cast<int32_t>(_ghostDepth),
						   _gpu0, _gpu0CommandPool, _gpu0Fence,
						   *_gpu0InsertPipeline, frameIndex, _gpu0NumBlocks,
						   static_cast<int32_t>(_ghostDepth + _zMid));

			if (_wrapEnabled)
			{
				int32_t gpu1OwnedCount = static_cast<int32_t>(_gridDimensionZ - _zMid);

				BufferTransfer(_gpu1, _gpu1CommandPool, _gpu1Fence,
							   *_gpu1ExtractPipeline, frameIndex, _gpu1NumBlocks,
							   static_cast<int32_t>(_ghostDepth) + gpu1OwnedCount - static_cast<int32_t>(_ghostDepth),
							   _gpu0, _gpu0CommandPool, _gpu0Fence,
							   *_gpu0InsertPipeline, frameIndex, _gpu0NumBlocks,
							   0);

				BufferTransfer(_gpu0, _gpu0CommandPool, _gpu0Fence,
							   *_gpu0ExtractPipeline, frameIndex, _gpu0NumBlocks,
							   static_cast<int32_t>(_ghostDepth),
							   _gpu1, _gpu1CommandPool, _gpu1Fence,
							   *_gpu1InsertPipeline, frameIndex, _gpu1NumBlocks,
							   static_cast<int32_t>(_ghostDepth) + gpu1OwnedCount);
			}
		}
	}

} // namespace Cave
