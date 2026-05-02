#include "simulationRenderer.h"
#include "../common/logger.h"
#include "pipelines/computePipeline.h"
#include "descriptors.h"
#include <algorithm>
#include <fstream>

namespace Cave
{
	VulkanSimulationRenderer::VulkanSimulationRenderer(DeviceContext &deviceContext, Simulation &simulation, uint32_t framesInFlight,
													CellStateStorageMode storageMode, const DomainConfig* domainConfig)
		: _deviceContext{deviceContext}, _simulation{simulation}, _framesInFlight{framesInFlight}, _storageMode{storageMode}
	{
		_slangCompiler = std::make_unique<SlangCompiler>();

		const glm::uvec4* globalDimensions = _simulation.GetDimensions();

		if (domainConfig)
		{
			_domainConfig = std::make_unique<DomainConfig>(*domainConfig);
			_localZSize = domainConfig->GetLocalZSize();
			_ownedZStart = domainConfig->ghostDepth;
			_ownedZEnd = domainConfig->ghostDepth + domainConfig->ownedZSize;
			_zDomainOffset = domainConfig->zDomainOffset;
			_ghostDepth = domainConfig->ghostDepth;

			LOG_INFO("Domain decomposition: globalZ={}, offset={}, owned={}, ghost={}, localZ={}, storage={}",
				domainConfig->globalGridDimensionZ, _zDomainOffset,
				domainConfig->ownedZSize, _ghostDepth, _localZSize,
				_storageMode == CellStateStorageMode::Image ? "image" : "buffer");
		}
		else
		{
			_localZSize = globalDimensions->z;
			_ownedZStart = 0;
			_ownedZEnd = globalDimensions->z;
			_zDomainOffset = 0;
			_ghostDepth = 0;
		}

		if (_storageMode == CellStateStorageMode::Image)
			InitImages();
		else
			InitBuffers();

		GenerateShaders();
	}

	// ==============================
	// Buffer mode initialization
	// ==============================

	void VulkanSimulationRenderer::InitBuffers()
	{
		const glm::uvec4* dimensions = _simulation.GetDimensions();

		// Packed 4-bit format: 2 cells per byte along X axis
		_cellStateImageWidth = (dimensions->x + 1) / 2;

		LOG_INFO("Cell state buffer (packed 4-bit, texel width {} for grid width {}, localZ={})",
			_cellStateImageWidth, dimensions->x, _localZSize);

		// Tiled 8x8x8 block layout with Morton order within each block
		// Use _localZSize for Z dimension (includes ghost slices when domain decomposed)
		uint32_t cellNumBlocksX = (_cellStateImageWidth + 7) / 8;
		uint32_t cellNumBlocksY = (dimensions->y + 7) / 8;
		uint32_t cellNumBlocksZ = (_localZSize + 7) / 8;
		_cellStateBufferElementCount = cellNumBlocksX * cellNumBlocksY * cellNumBlocksZ * 512;
		uint64_t cellStateBufferSize = static_cast<uint64_t>(_cellStateBufferElementCount) * sizeof(uint32_t);

		_cellStateBuffers.resize(_framesInFlight);
		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			_cellStateBuffers[frameIndex] = Buffer::Storage(
				_deviceContext,
				cellStateBufferSize,
				true, // concurrentSharing
				0,    // alignment
				"Cell State Buffer " + std::to_string(frameIndex));
		}

		LOG_INFO("Created {} cell state buffers ({} elements, {} bytes each, tiled {}x{}x{} blocks)",
			_framesInFlight, _cellStateBufferElementCount, cellStateBufferSize,
			cellNumBlocksX, cellNumBlocksY, cellNumBlocksZ);

		// Skip grid for empty-space acceleration (uses _localZSize for Z)
		_skipGridDimensionX = (dimensions->x + _skipGridChunkSize - 1) / _skipGridChunkSize;
		_skipGridDimensionY = (dimensions->y + _skipGridChunkSize - 1) / _skipGridChunkSize;
		_skipGridDimensionZ = (_localZSize + _skipGridChunkSize - 1) / _skipGridChunkSize;

		uint32_t skipNumBlocksX = (_skipGridDimensionX + 7) / 8;
		uint32_t skipNumBlocksY = (_skipGridDimensionY + 7) / 8;
		uint32_t skipNumBlocksZ = (_skipGridDimensionZ + 7) / 8;
		_skipGridBufferElementCount = skipNumBlocksX * skipNumBlocksY * skipNumBlocksZ * 512;
		uint64_t skipGridBufferSize = static_cast<uint64_t>(_skipGridBufferElementCount) * sizeof(uint32_t);

		_skipGridBuffers.resize(_framesInFlight);
		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			_skipGridBuffers[frameIndex] = Buffer::Storage(
				_deviceContext,
				skipGridBufferSize,
				true, // concurrentSharing
				0,    // alignment
				"Skip Grid Buffer " + std::to_string(frameIndex));
		}

		LOG_INFO("Created {} skip grid buffers ({}x{}x{}, chunk size {})",
			_framesInFlight, _skipGridDimensionX, _skipGridDimensionY, _skipGridDimensionZ, _skipGridChunkSize);

		// Initialize buffers with spawn data
		InitCellStateBuffersOnGPU();
	}

	void VulkanSimulationRenderer::InitCellStateBuffersOnGPU()
	{
		const glm::uvec4* dims = _simulation.GetDimensions();
		const glm::uvec4* spawnDims = _simulation.GetCenterSpawnAreaDimensions();

		// Compute global spawn bounds (centered in the full grid)
		glm::ivec3 globalSpawnMin(
			(dims->x - spawnDims->x) / 2,
			(dims->y - spawnDims->y) / 2,
			(dims->z - spawnDims->z) / 2);
		glm::ivec3 globalSpawnMax(
			(dims->x + spawnDims->x) / 2,
			(dims->y + spawnDims->y) / 2,
			(dims->z + spawnDims->z) / 2);

		// Adjust spawn Z coordinates to local domain coordinates
		// Local Z = global Z - zDomainOffset + ghostDepth
		glm::ivec3 spawnMin = globalSpawnMin;
		glm::ivec3 spawnMax = globalSpawnMax;
		if (_domainConfig)
		{
			int32_t zOffset = static_cast<int32_t>(_ghostDepth) - static_cast<int32_t>(_zDomainOffset);
			spawnMin.z = std::max(static_cast<int32_t>(_ownedZStart), globalSpawnMin.z + zOffset);
			spawnMax.z = std::min(static_cast<int32_t>(_ownedZEnd), globalSpawnMax.z + zOffset);

			if (spawnMin.z >= spawnMax.z)
				LOG_INFO("Spawn area does not overlap with this domain (offset={}, owned=[{},{}])",
					_zDomainOffset, _ownedZStart, _ownedZEnd);
		}

		// Zero-fill all buffers and set skip grids to 1 (all chunks active)
		{
			vk::CommandBuffer commandBuffer = _deviceContext.BeginSingleTimeCommands(
				_deviceContext.GetComputeCommandPool());

			for (auto& buffer : _cellStateBuffers)
				commandBuffer.fillBuffer(buffer->GetBuffer(), 0, buffer->GetSize(), 0u);

			// Skip grid starts at 0 (all chunks inactive). The cellInit shader scatter-writes
			// chunks containing alive cells before the first render. Pre-fix this was 1u
			// (all chunks active), which TDR'd the GPU on the first frame's ray march for
			// large grids since every ray traversed the entire skip grid cell-by-cell.
			for (auto& buffer : _skipGridBuffers)
				commandBuffer.fillBuffer(buffer->GetBuffer(), 0, buffer->GetSize(), 0u);

			vk::MemoryBarrier fillBarrier = vk::MemoryBarrier(
				vk::AccessFlagBits::eTransferWrite,                          // srcAccessMask
				vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite // dstAccessMask
			);
			commandBuffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eComputeShader,
				{}, fillBarrier, {}, {});

			_deviceContext.EndSingleTimeCommands(std::move(commandBuffer),
				_deviceContext.GetComputeCommandPool(), _deviceContext.GetComputeQueue());
		}

		// Compute max neighbor reach so cellInit can scatter-write the skip grid the same
		// way the simulation tick does (mark this cell's chunk + neighbor chunks within
		// reach). Same calculation as GenerateRayMarchComputeShader uses for the sim shader.
		const auto& cellInitNeighborDeltaConfigurations = _simulation.GetNeighborDeltaConfigurations();
		int initMaxReachX = 0, initMaxReachY = 0, initMaxReachZ = 0;
		for (const auto& configurationDeltas : cellInitNeighborDeltaConfigurations)
			for (const auto& delta : configurationDeltas)
			{
				initMaxReachX = std::max(initMaxReachX, std::abs(delta.x));
				initMaxReachY = std::max(initMaxReachY, std::abs(delta.y));
				initMaxReachZ = std::max(initMaxReachZ, std::abs(delta.z));
			}

		if (!_rayMarchInitShaderModule)
		{
			// GRID_DIM_Z uses _localZSize (includes ghost slices when domain decomposed)
			std::map<std::string, std::string> initDefines = {
				{"IMAGE_WIDTH", std::to_string(_cellStateImageWidth)},
				{"GRID_DIM_X", std::to_string(dims->x)},
				{"GRID_DIM_Y", std::to_string(dims->y)},
				{"GRID_DIM_Z", std::to_string(_localZSize)},
				{"SPAWN_MIN_X", std::to_string(spawnMin.x)},
				{"SPAWN_MIN_Y", std::to_string(spawnMin.y)},
				{"SPAWN_MIN_Z", std::to_string(spawnMin.z)},
				{"SPAWN_MAX_X", std::to_string(spawnMax.x)},
				{"SPAWN_MAX_Y", std::to_string(spawnMax.y)},
				{"SPAWN_MAX_Z", std::to_string(spawnMax.z)},
				{"SPAWN_MODE", std::to_string(spawnDims->w)},
				{"SPAWN_RANDOM_DENSITY", std::to_string(_simulation.GetSpawnRandomDensity())},
				{"CHUNK_SIZE", std::to_string(_skipGridChunkSize)},
				{"SKIP_GRID_DIM_X", std::to_string(_skipGridDimensionX)},
				{"SKIP_GRID_DIM_Y", std::to_string(_skipGridDimensionY)},
				{"SKIP_GRID_DIM_Z", std::to_string(_skipGridDimensionZ)},
				{"MAX_NEIGHBOR_REACH_X", std::to_string(initMaxReachX)},
				{"MAX_NEIGHBOR_REACH_Y", std::to_string(initMaxReachY)},
				{"MAX_NEIGHBOR_REACH_Z", std::to_string(initMaxReachZ)},
			};

			_rayMarchInitShaderModule = _slangCompiler->Compile(
				_deviceContext, "shaders/slang", "cellInit", "initMain", initDefines);

			if (!_rayMarchInitShaderModule)
			{
				LOG_ERROR("Failed to compile cell init compute shader via Slang");
				return;
			}
			LOG_INFO("Cell init compute shader compiled via Slang (buffer mode, spawnMode={}, density={:.3f})",
				spawnDims->w, _simulation.GetSpawnRandomDensity());
		}

		uint32_t dispatchX = (_cellStateImageWidth + 7) / 8;
		uint32_t dispatchY = (dims->y + 7) / 8;
		uint32_t dispatchZ = (_localZSize + 3) / 4;

		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			auto initDescriptor = std::make_shared<Descriptor>(_deviceContext);
			initDescriptor->BindBufferToDescriptorSet(
				0, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, _cellStateBuffers[frameIndex]);
			initDescriptor->BindBufferToDescriptorSet(
				1, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, _skipGridBuffers[frameIndex]);
			initDescriptor->Build();

			auto initShader = std::make_shared<Shader>(_deviceContext, _rayMarchInitShaderModule);
			auto initPipeline = std::make_unique<ComputePipeline>(_deviceContext, initShader);
			initPipeline->AddPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(RayMarchComputePushConstants));
			initPipeline->AddDescriptorSet(0, initDescriptor);
			initPipeline->Build();

			RayMarchComputePushConstants initPushConstants{};
			initPushConstants.simulationParameters = *_simulation.GetSimulationParameters();

			vk::CommandBuffer commandBuffer = _deviceContext.BeginSingleTimeCommands(
				_deviceContext.GetComputeCommandPool());

			initPipeline->Bind(commandBuffer, vk::PipelineBindPoint::eCompute, 0, Pipeline::DescriptorOption{0});
			commandBuffer.pushConstants<RayMarchComputePushConstants>(
				initPipeline->GetPipelineLayout(),
				vk::ShaderStageFlagBits::eCompute,
				0,
				initPushConstants
			);
			commandBuffer.dispatch(dispatchX, dispatchY, dispatchZ);

			vk::MemoryBarrier initBarrier = vk::MemoryBarrier(
				vk::AccessFlagBits::eShaderWrite,  // srcAccessMask
				vk::AccessFlagBits::eShaderRead    // dstAccessMask
			);
			commandBuffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader,
				{}, initBarrier, {}, {});

			_deviceContext.EndSingleTimeCommands(std::move(commandBuffer),
				_deviceContext.GetComputeCommandPool(), _deviceContext.GetComputeQueue());
		}

		LOG_INFO("GPU-side buffer initialization complete for {} frames", _framesInFlight);
	}

	// ==============================
	// Image mode initialization
	// ==============================

	void VulkanSimulationRenderer::InitImages()
	{
		const glm::uvec4* dimensions = _simulation.GetDimensions();
		QueueFamilies queueFamilies = _deviceContext.GetQueueFamilies();
		std::vector<uint32_t> computeGraphicsQueueFamilyIndices = {
			queueFamilies.ComputeFamily.Index.value(),
			queueFamilies.GraphicsFamily.Index.value()
		};

		_cellStateImageFormat = Image::FindSupportedFormat(
			_deviceContext.GetPhysicalDevice(),
			{vk::Format::eR8Uint},
			vk::ImageTiling::eOptimal,
			vk::FormatFeatureFlagBits::eStorageImage);

		// Packed 4-bit format: 2 cells per byte along X axis
		_cellStateImageWidth = (dimensions->x + 1) / 2;

		LOG_INFO("Cell state 3D image format: {} (packed 4-bit, image width {} for grid width {}, localZ={})",
			vk::to_string(_cellStateImageFormat), _cellStateImageWidth, dimensions->x, _localZSize);

		_cellStateImages.resize(_framesInFlight);
		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			_cellStateImages[frameIndex] = std::make_shared<Image>(
				_deviceContext,
				_cellStateImageFormat,
				1,																		// arrayCount
				1,																		// mipLevels
				vk::Extent3D(_cellStateImageWidth, dimensions->y, _localZSize),			// extent
				vk::ImageTiling::eOptimal,												// imageTiling
				vk::ImageUsageFlagBits::eStorage |
					vk::ImageUsageFlagBits::eTransferSrc |
					vk::ImageUsageFlagBits::eTransferDst,								// imageUsageFlags
				vk::MemoryPropertyFlagBits::eDeviceLocal,								// memoryPropertyFlags
				vk::ImageCreateFlags(),													// imageCreateFlags
				vk::ImageAspectFlagBits::eColor,										// imageAspectFlags
				vk::ImageViewType::e3D,													// imageViewType
				vk::SharingMode::eConcurrent,											// sharingMode
				computeGraphicsQueueFamilyIndices,										// queueFamilyIndices
				vk::ImageType::e3D														// imageType
			);
		}

		LOG_INFO("Created {} 3D cell state images ({}x{}x{})", _framesInFlight, _cellStateImageWidth, dimensions->y, _localZSize);

		// Create skip grid images for empty-space acceleration (uses _localZSize for Z)
		_skipGridDimensionX = (dimensions->x + _skipGridChunkSize - 1) / _skipGridChunkSize;
		_skipGridDimensionY = (dimensions->y + _skipGridChunkSize - 1) / _skipGridChunkSize;
		_skipGridDimensionZ = (_localZSize + _skipGridChunkSize - 1) / _skipGridChunkSize;

		_skipGridImages.resize(_framesInFlight);
		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			_skipGridImages[frameIndex] = std::make_shared<Image>(
				_deviceContext,
				vk::Format::eR32Uint,
				1,																						// arrayCount
				1,																						// mipLevels
				vk::Extent3D(_skipGridDimensionX, _skipGridDimensionY, _skipGridDimensionZ),			// extent
				vk::ImageTiling::eOptimal,																// imageTiling
				vk::ImageUsageFlagBits::eStorage |
					vk::ImageUsageFlagBits::eTransferDst,												// imageUsageFlags
				vk::MemoryPropertyFlagBits::eDeviceLocal,												// memoryPropertyFlags
				vk::ImageCreateFlags(),																	// imageCreateFlags
				vk::ImageAspectFlagBits::eColor,														// imageAspectFlags
				vk::ImageViewType::e3D,																	// imageViewType
				vk::SharingMode::eConcurrent,															// sharingMode
				computeGraphicsQueueFamilyIndices,														// queueFamilyIndices
				vk::ImageType::e3D																		// imageType
			);
		}

		LOG_INFO("Created {} ping-pong skip grid images ({}x{}x{}, chunk size {})",
			_framesInFlight, _skipGridDimensionX, _skipGridDimensionY, _skipGridDimensionZ, _skipGridChunkSize);

		// Initialize 3D cell state images with spawn data
		InitCellStateImagesOnGPU();
	}

	void VulkanSimulationRenderer::InitCellStateImagesOnGPU()
	{
		const glm::uvec4* dims = _simulation.GetDimensions();
		const glm::uvec4* spawnDims = _simulation.GetCenterSpawnAreaDimensions();

		// Compute global spawn bounds (centered in the full grid)
		glm::ivec3 globalSpawnMin(
			(dims->x - spawnDims->x) / 2,
			(dims->y - spawnDims->y) / 2,
			(dims->z - spawnDims->z) / 2);
		glm::ivec3 globalSpawnMax(
			(dims->x + spawnDims->x) / 2,
			(dims->y + spawnDims->y) / 2,
			(dims->z + spawnDims->z) / 2);

		// Adjust spawn Z for domain decomposition (same logic as buffer init)
		glm::ivec3 spawnMin = globalSpawnMin;
		glm::ivec3 spawnMax = globalSpawnMax;
		if (_domainConfig)
		{
			int32_t zOffset = static_cast<int32_t>(_ghostDepth) - static_cast<int32_t>(_zDomainOffset);
			spawnMin.z = std::max(static_cast<int32_t>(_ownedZStart), globalSpawnMin.z + zOffset);
			spawnMax.z = std::min(static_cast<int32_t>(_ownedZEnd), globalSpawnMax.z + zOffset);
		}

		// Pre-initialize all images on the GRAPHICS queue to establish visibility.
		{
			vk::CommandBuffer graphicsInitCommandBuffer = _deviceContext.BeginSingleTimeCommands(
				_deviceContext.GetGraphicsCommandPool());

			vk::ImageSubresourceRange subresourceRange = vk::ImageSubresourceRange(
				vk::ImageAspectFlagBits::eColor,	// aspectMask
				0,									// baseMipLevel
				1,									// levelCount
				0,									// baseArrayLayer
				1									// layerCount
			);

			vk::ClearColorValue zeroClear;
			zeroClear.setUint32({0u, 0u, 0u, 0u});
			// Skip grid starts at 0 (all chunks inactive). The cellInit shader scatter-writes
			// chunks containing alive cells before the first render. Pre-fix this was 1u
			// (all chunks active), which TDR'd the GPU on the first frame's ray march for
			// large grids since every ray traversed the entire skip grid cell-by-cell.
			vk::ClearColorValue skipGridInitClear;
			skipGridInitClear.setUint32({0u, 0u, 0u, 0u});

			// Transition and clear all cell state images + skip grid images
			std::vector<std::shared_ptr<Image>>* imageGroups[] = {&_cellStateImages, &_skipGridImages};
			for (auto* imageGroup : imageGroups)
			{
				bool isSkipGrid = (imageGroup == &_skipGridImages);
				for (auto& image : *imageGroup)
				{
					vk::ImageMemoryBarrier barrierToClear = vk::ImageMemoryBarrier(
						vk::AccessFlagBits::eNone,					// srcAccessMask
						vk::AccessFlagBits::eTransferWrite,			// dstAccessMask
						vk::ImageLayout::eUndefined,				// oldLayout
						vk::ImageLayout::eTransferDstOptimal,		// newLayout
						VK_QUEUE_FAMILY_IGNORED,					// srcQueueFamilyIndex
						VK_QUEUE_FAMILY_IGNORED,					// dstQueueFamilyIndex
						*image->GetImage(),							// image
						subresourceRange							// subresourceRange
					);
					graphicsInitCommandBuffer.pipelineBarrier(
						vk::PipelineStageFlagBits::eTopOfPipe,
						vk::PipelineStageFlagBits::eTransfer,
						{}, {}, {}, barrierToClear);

					graphicsInitCommandBuffer.clearColorImage(
						*image->GetImage(),
						vk::ImageLayout::eTransferDstOptimal,
						isSkipGrid ? skipGridInitClear : zeroClear,
						subresourceRange);

					vk::ImageMemoryBarrier barrierToGeneral = vk::ImageMemoryBarrier(
						vk::AccessFlagBits::eTransferWrite,			// srcAccessMask
						vk::AccessFlagBits::eShaderRead |
							vk::AccessFlagBits::eShaderWrite,		// dstAccessMask
						vk::ImageLayout::eTransferDstOptimal,		// oldLayout
						vk::ImageLayout::eGeneral,					// newLayout
						VK_QUEUE_FAMILY_IGNORED,					// srcQueueFamilyIndex
						VK_QUEUE_FAMILY_IGNORED,					// dstQueueFamilyIndex
						*image->GetImage(),							// image
						subresourceRange							// subresourceRange
					);
					graphicsInitCommandBuffer.pipelineBarrier(
						vk::PipelineStageFlagBits::eTransfer,
						vk::PipelineStageFlagBits::eComputeShader,
						{}, {}, {}, barrierToGeneral);
				}
			}

			_deviceContext.EndSingleTimeCommands(std::move(graphicsInitCommandBuffer),
				_deviceContext.GetGraphicsCommandPool(), _deviceContext.GetGraphicsQueue());

			LOG_INFO("Graphics queue pre-initialized {} cell state + {} skip grid images",
				_cellStateImages.size(), _skipGridImages.size());
		}

		// Compute max neighbor reach so cellInit can scatter-write the skip grid the same
		// way the simulation tick does (mark this cell's chunk + neighbor chunks within
		// reach). Same calculation as GenerateRayMarchComputeShader uses for the sim shader.
		const auto& cellInitNeighborDeltaConfigurations = _simulation.GetNeighborDeltaConfigurations();
		int initMaxReachX = 0, initMaxReachY = 0, initMaxReachZ = 0;
		for (const auto& configurationDeltas : cellInitNeighborDeltaConfigurations)
			for (const auto& delta : configurationDeltas)
			{
				initMaxReachX = std::max(initMaxReachX, std::abs(delta.x));
				initMaxReachY = std::max(initMaxReachY, std::abs(delta.y));
				initMaxReachZ = std::max(initMaxReachZ, std::abs(delta.z));
			}

		if (!_rayMarchInitShaderModule)
		{
			std::map<std::string, std::string> initDefines = {
				{"USE_IMAGE", "1"},
				{"IMAGE_WIDTH", std::to_string(_cellStateImageWidth)},
				{"GRID_DIM_X", std::to_string(dims->x)},
				{"GRID_DIM_Y", std::to_string(dims->y)},
				{"GRID_DIM_Z", std::to_string(_localZSize)},
				{"SPAWN_MIN_X", std::to_string(spawnMin.x)},
				{"SPAWN_MIN_Y", std::to_string(spawnMin.y)},
				{"SPAWN_MIN_Z", std::to_string(spawnMin.z)},
				{"SPAWN_MAX_X", std::to_string(spawnMax.x)},
				{"SPAWN_MAX_Y", std::to_string(spawnMax.y)},
				{"SPAWN_MAX_Z", std::to_string(spawnMax.z)},
				{"SPAWN_MODE", std::to_string(spawnDims->w)},
				{"SPAWN_RANDOM_DENSITY", std::to_string(_simulation.GetSpawnRandomDensity())},
				{"CHUNK_SIZE", std::to_string(_skipGridChunkSize)},
				{"SKIP_GRID_DIM_X", std::to_string(_skipGridDimensionX)},
				{"SKIP_GRID_DIM_Y", std::to_string(_skipGridDimensionY)},
				{"SKIP_GRID_DIM_Z", std::to_string(_skipGridDimensionZ)},
				{"MAX_NEIGHBOR_REACH_X", std::to_string(initMaxReachX)},
				{"MAX_NEIGHBOR_REACH_Y", std::to_string(initMaxReachY)},
				{"MAX_NEIGHBOR_REACH_Z", std::to_string(initMaxReachZ)},
			};

			_rayMarchInitShaderModule = _slangCompiler->Compile(
				_deviceContext, "shaders/slang", "cellInit", "initMain", initDefines);

			if (!_rayMarchInitShaderModule)
			{
				LOG_ERROR("Failed to compile cell init compute shader via Slang (image mode)");
				return;
			}
			LOG_INFO("Cell init compute shader compiled via Slang (image mode, spawnMode={}, density={:.3f})",
				spawnDims->w, _simulation.GetSpawnRandomDensity());
		}

		uint32_t dispatchX = (_cellStateImageWidth + 7) / 8;
		uint32_t dispatchY = (dims->y + 7) / 8;
		uint32_t dispatchZ = (_localZSize + 3) / 4;

		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			auto initDescriptor = std::make_shared<Descriptor>(_deviceContext);
			initDescriptor->BindImageToDescriptorSet(
				0, vk::DescriptorType::eStorageImage,
				vk::ShaderStageFlagBits::eCompute, _cellStateImages[frameIndex]);
			initDescriptor->BindImageToDescriptorSet(
				1, vk::DescriptorType::eStorageImage,
				vk::ShaderStageFlagBits::eCompute, _skipGridImages[frameIndex]);
			initDescriptor->Build();

			auto initShader = std::make_shared<Shader>(_deviceContext, _rayMarchInitShaderModule);
			auto initPipeline = std::make_unique<ComputePipeline>(_deviceContext, initShader);
			initPipeline->AddPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(RayMarchComputePushConstants));
			initPipeline->AddDescriptorSet(0, initDescriptor);
			initPipeline->Build();

			RayMarchComputePushConstants initPushConstants{};
			initPushConstants.simulationParameters = *_simulation.GetSimulationParameters();

			vk::CommandBuffer commandBuffer = _deviceContext.BeginSingleTimeCommands(
				_deviceContext.GetComputeCommandPool());

			initPipeline->Bind(commandBuffer, vk::PipelineBindPoint::eCompute, 0, Pipeline::DescriptorOption{0});
			commandBuffer.pushConstants<RayMarchComputePushConstants>(
				initPipeline->GetPipelineLayout(),
				vk::ShaderStageFlagBits::eCompute,
				0,
				initPushConstants
			);
			commandBuffer.dispatch(dispatchX, dispatchY, dispatchZ);

			vk::MemoryBarrier initBarrier = vk::MemoryBarrier(
				vk::AccessFlagBits::eShaderWrite,  // srcAccessMask
				vk::AccessFlagBits::eShaderRead    // dstAccessMask
			);
			commandBuffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader,
				{}, initBarrier, {}, {});

			_deviceContext.EndSingleTimeCommands(std::move(commandBuffer),
				_deviceContext.GetComputeCommandPool(), _deviceContext.GetComputeQueue());
		}

		LOG_INFO("GPU-side 3D image initialization complete for {} frames", _framesInFlight);
	}

	// ==============================
	// Reset
	// ==============================

	void VulkanSimulationRenderer::Reset()
	{
		_deviceContext.GetDevice().waitIdle();

		if (_storageMode == CellStateStorageMode::Image)
			InitCellStateImagesOnGPU();
		else
			InitCellStateBuffersOnGPU();

		GenerateRayMarchComputeShader();
		GenerateRayMarchVertexShader();
		GenerateRayMarchFragmentShader();
	}

	// ==============================
	// Shader generation
	// ==============================

	void VulkanSimulationRenderer::GenerateShaders()
	{
		GenerateRayMarchComputeShader();
		GenerateRayMarchVertexShader();
		GenerateRayMarchFragmentShader();
	}

	void VulkanSimulationRenderer::GenerateRayMarchComputeShader()
	{
		const auto &gridDimensions = *_simulation.GetDimensions();
		const auto& neighborDeltaConfigurations = _simulation.GetNeighborDeltaConfigurations();
		uint32_t neighborCount = _simulation.GetNumberOfNeighbors();
		const bool wrapsAtBoundary = ((*_simulation.GetSimulationParameters()).survivalAndNeighborhoodRules >> Simulation::BIT_SHIFT) & Simulation::WRAP_NEIGHBORS_MASK;

		// Determine shape type for Slang compilation
		std::string shapeName = _simulation.GetShape()->GetName();

		int maxNeighborReachX = 0, maxNeighborReachY = 0, maxNeighborReachZ = 0;
		for (const auto& configurationDeltas : neighborDeltaConfigurations)
		{
			for (const auto& delta : configurationDeltas)
			{
				maxNeighborReachX = std::max(maxNeighborReachX, std::abs(delta.x));
				maxNeighborReachY = std::max(maxNeighborReachY, std::abs(delta.y));
				maxNeighborReachZ = std::max(maxNeighborReachZ, std::abs(delta.z));
			}
		}

		std::map<std::string, std::string> defines = {
			{"GRID_DIM_X", std::to_string(gridDimensions.x)},
			{"GRID_DIM_Y", std::to_string(gridDimensions.y)},
			{"GRID_DIM_Z", std::to_string(_localZSize)},
			{"IMAGE_WIDTH", std::to_string(_cellStateImageWidth)},
			{"CHUNK_SIZE", std::to_string(_skipGridChunkSize)},
			{"SKIP_GRID_DIM_X", std::to_string(_skipGridDimensionX)},
			{"SKIP_GRID_DIM_Y", std::to_string(_skipGridDimensionY)},
			{"SKIP_GRID_DIM_Z", std::to_string(_skipGridDimensionZ)},
			{"MAX_NEIGHBOR_REACH_X", std::to_string(maxNeighborReachX)},
			{"MAX_NEIGHBOR_REACH_Y", std::to_string(maxNeighborReachY)},
			{"MAX_NEIGHBOR_REACH_Z", std::to_string(maxNeighborReachZ)},
			{"WRAP_AT_BOUNDARY", wrapsAtBoundary ? "1" : "0"},
			{"SHAPE_TYPE", [&]() -> std::string {
				if (shapeName == "cube")
				{
					if (neighborCount == 6) return "0";       // CubeFace
					if (neighborCount == 18) return "1";      // CubeFaceEdge
					return "2";                                // CubeFaceEdgeCorner
				}
				if (shapeName == "ElongatedDodecahedron")
				{
					if (neighborCount == 12) return "3";      // ElongatedDodecahedronFace
					return "4";                                // ElongatedDodecahedronFaceEdge
				}
				return "0";
			}()},
		};

		if (_domainConfig)
		{
			defines["DOMAIN_DECOMPOSITION"] = "1";
			defines["Z_OWNED_START"] = std::to_string(_ownedZStart);
			defines["Z_OWNED_END"] = std::to_string(_ownedZEnd);
		}

		if (_storageMode == CellStateStorageMode::Image)
			defines["USE_IMAGE"] = "1";

		_rayMarchComputeShaderModule = _slangCompiler->Compile(
			_deviceContext, "shaders/slang", "rayMarchCompute", "computeMain", defines);

		if (!_rayMarchComputeShaderModule)
		{
			LOG_ERROR("Failed to compile ray march compute shader via Slang");
		}
		else
		{
			LOG_INFO("Ray march compute shader compiled via Slang (shape: {}, storage: {})",
				shapeName, _storageMode == CellStateStorageMode::Image ? "image" : "buffer");
		}
	}


	void VulkanSimulationRenderer::GenerateRayMarchVertexShader()
	{
		// VALIDATION: Slang path under test
		_rayMarchVertexShaderModule = _slangCompiler->Compile(
			_deviceContext, "shaders/slang", "rayMarchVertex", "vertexMain");

		if (_rayMarchVertexShaderModule)
		{
			LOG_INFO("Ray march vertex shader compiled via Slang");
		}
		else
		{
			LOG_ERROR("Slang vertex shader compile failed");
		}
	}

	void VulkanSimulationRenderer::GenerateRayMarchFragmentShader()
	{
		// VALIDATION: Slang path under test
		std::string shapeName = _simulation.GetShape()->GetName();
		uint32_t shapeNeighborCount = _simulation.GetNumberOfNeighbors();
		std::string shapeTypeValue = "0";
		if (shapeName == "cube")
		{
			if (shapeNeighborCount == 6) shapeTypeValue = "0";
			else if (shapeNeighborCount == 18) shapeTypeValue = "1";
			else shapeTypeValue = "2";
		}
		else if (shapeName == "ElongatedDodecahedron")
		{
			shapeTypeValue = (shapeNeighborCount == 12) ? "3" : "4";
		}
		std::map<std::string, std::string> slangDefines;
		slangDefines["SHAPE_TYPE"] = shapeTypeValue;
		if (_storageMode == CellStateStorageMode::Image)
		{
			slangDefines["USE_IMAGE"] = "1";
		}
		else
		{
			// Buffer mode needs IMAGE_WIDTH for the per-axis tile-block count math.
			slangDefines["IMAGE_WIDTH"] = std::to_string(_cellStateImageWidth);
		}

		// Spatial reach — max neighbor delta magnitude per axis. Used by the fragment shader
		// to check overlapping cell geometry from neighbor cells at each fine-DDA step.
		const auto& neighborDeltaConfigurations = _simulation.GetNeighborDeltaConfigurations();
		int maxReachX = 0, maxReachY = 0, maxReachZ = 0;
		for (const auto& configurationDeltas : neighborDeltaConfigurations)
		{
			for (const auto& delta : configurationDeltas)
			{
				maxReachX = std::max(maxReachX, std::abs(delta.x));
				maxReachY = std::max(maxReachY, std::abs(delta.y));
				maxReachZ = std::max(maxReachZ, std::abs(delta.z));
			}
		}
		slangDefines["SPATIAL_REACH_X"] = std::to_string(maxReachX);
		slangDefines["SPATIAL_REACH_Y"] = std::to_string(maxReachY);
		slangDefines["SPATIAL_REACH_Z"] = std::to_string(maxReachZ);

		// Domain decomposition flag — fragment encodes linear distance in alpha for compositor
		if (_domainConfig)
		{
			slangDefines["ENCODE_DEPTH_IN_ALPHA"] = "1";
		}

		_rayMarchFragmentShaderModule = _slangCompiler->Compile(
			_deviceContext, "shaders/slang", "rayMarchFragment", "fragmentMain", slangDefines);

		if (_rayMarchFragmentShaderModule)
		{
			LOG_INFO("Ray march fragment shader compiled via Slang");
		}
		else
		{
			LOG_ERROR("Slang fragment shader compile failed");
		}
	}

}
