#include "computeSystem.h"
#include "../../common/logger.h"

namespace Cave
{
	ComputeSystem::ComputeSystem(DeviceContext& deviceContext, VulkanSimulationRenderer& simulationRenderer)
		: _deviceContext{deviceContext}, _simulationRenderer{simulationRenderer},
		  _framesInFlight{deviceContext.GetFramesInFlight()}
	{
		_computeCommandPool = _deviceContext.GetComputeCommandPool();

		_inFlightFences.resize(_framesInFlight);
		_completedSemaphores.resize(_framesInFlight);
		_completedSemaphoreSignalValues.resize(_framesInFlight, 0);

		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			_inFlightFences[frameIndex]      = _deviceContext.CreateFence();
			_completedSemaphores[frameIndex] = _deviceContext.CreateTimelineSemaphore(0);
		}

		BuildDescriptors();
		BuildPipeline();
		BuildCommandBuffers();
	}

	ComputeSystem::~ComputeSystem()
	{
		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			_deviceContext.GetDevice().destroyFence(_inFlightFences[frameIndex]);
			_deviceContext.GetDevice().destroySemaphore(_completedSemaphores[frameIndex]);
		}
	}

	void ComputeSystem::BuildDescriptors()
	{
		LOG_INFO("ComputeSystem::BuildDescriptors — rayMarch module: {:#x}",
			(uint64_t)(VkShaderModule)_simulationRenderer.GetRayMarchComputeShaderModule());

		_rayMarchDescriptor = std::make_shared<Descriptor>(_deviceContext);
		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			uint32_t nextFrame = (frameIndex + 1) % _framesInFlight;

			if (_simulationRenderer.UsesImages())
			{
				// binding 0: cell state input image (read from current frame)
				_rayMarchDescriptor->BindImageToDescriptorSet(
					0,													// binding
					vk::DescriptorType::eStorageImage,					// type
					vk::ShaderStageFlagBits::eCompute,					// stage
					_simulationRenderer.GetCellStateImage(frameIndex)	// image
				);
				// binding 1: cell state output image (write to next frame)
				_rayMarchDescriptor->BindImageToDescriptorSet(
					1,													// binding
					vk::DescriptorType::eStorageImage,					// type
					vk::ShaderStageFlagBits::eCompute,					// stage
					_simulationRenderer.GetCellStateImage(nextFrame)	// image
				);
				// binding 2: skip grid input image (read from current frame)
				_rayMarchDescriptor->BindImageToDescriptorSet(
					2,													// binding
					vk::DescriptorType::eStorageImage,					// type
					vk::ShaderStageFlagBits::eCompute,					// stage
					_simulationRenderer.GetSkipGridImage(frameIndex)	// image
				);
				// binding 3: skip grid output image (write to next frame)
				_rayMarchDescriptor->BindImageToDescriptorSet(
					3,													// binding
					vk::DescriptorType::eStorageImage,					// type
					vk::ShaderStageFlagBits::eCompute,					// stage
					_simulationRenderer.GetSkipGridImage(nextFrame)		// image
				);
			}
			else
			{
				// binding 0: cell state input buffer (read from current frame)
				_rayMarchDescriptor->BindBufferToDescriptorSet(
					0,													// binding
					vk::DescriptorType::eStorageBuffer,					// type
					vk::ShaderStageFlagBits::eCompute,					// stage
					_simulationRenderer.GetCellStateBuffer(frameIndex)	// buffer
				);
				// binding 1: cell state output buffer (write to next frame)
				_rayMarchDescriptor->BindBufferToDescriptorSet(
					1,													// binding
					vk::DescriptorType::eStorageBuffer,					// type
					vk::ShaderStageFlagBits::eCompute,					// stage
					_simulationRenderer.GetCellStateBuffer(nextFrame)	// buffer
				);
				// binding 2: skip grid input buffer (read from current frame)
				_rayMarchDescriptor->BindBufferToDescriptorSet(
					2,													// binding
					vk::DescriptorType::eStorageBuffer,					// type
					vk::ShaderStageFlagBits::eCompute,					// stage
					_simulationRenderer.GetSkipGridBuffer(frameIndex)	// buffer
				);
				// binding 3: skip grid output buffer (write to next frame)
				_rayMarchDescriptor->BindBufferToDescriptorSet(
					3,													// binding
					vk::DescriptorType::eStorageBuffer,					// type
					vk::ShaderStageFlagBits::eCompute,					// stage
					_simulationRenderer.GetSkipGridBuffer(nextFrame)	// buffer
				);
			}
		}
		_rayMarchDescriptor->Build();
	}

	void ComputeSystem::BuildPipeline()
	{
		LOG_INFO("ComputeSystem::BuildPipeline — rayMarch module: {:#x}",
			(uint64_t)(VkShaderModule)_simulationRenderer.GetRayMarchComputeShaderModule());

		auto rayMarchComputeShader = std::make_shared<Shader>(
			_deviceContext,
			_simulationRenderer.GetRayMarchComputeShaderModule());

		_rayMarchSimulationPipeline = std::make_unique<ComputePipeline>(_deviceContext, rayMarchComputeShader);
		_rayMarchSimulationPipeline->AddPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(RayMarchComputePushConstants));
		_rayMarchSimulationPipeline->AddDescriptorSet(0, _rayMarchDescriptor);
		_rayMarchSimulationPipeline->Build();
	}

	void ComputeSystem::RebuildPipeline()
	{
		_rayMarchSimulationPipeline.reset();
		_rayMarchDescriptor.reset();
		BuildDescriptors();
		BuildPipeline();
	}

	void ComputeSystem::BuildCommandBuffers()
	{
		_computeCommandBuffers.resize(_framesInFlight);
		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			_computeCommandBuffers[frameIndex] = _deviceContext.CreateCommandBuffer(_computeCommandPool);
		}
	}

	void ComputeSystem::Tick(uint32_t frameIndex, vk::Fence renderFence, bool dispatchCompute,
	                         bool enableComputeSkip,
	                         std::shared_ptr<QueryManager> queryManager, vk::QueryPool queryPool)
	{
		vk::Device device = _deviceContext.GetDevice();

		// Wait for both the previous compute and render on this frame slot to finish.
		// This ensures the render has consumed the completed semaphore before we signal it again.
		// When renderFence is null (e.g., GPU1 in domain decomposition has no render system),
		// only wait on the compute fence.
		vk::Result waitResult;
		if (renderFence)
		{
			std::array<vk::Fence, 2> computeAndRenderInFlightFences = {_inFlightFences[frameIndex], renderFence};
			waitResult = device.waitForFences(computeAndRenderInFlightFences, vk::True, UINT64_MAX);
		}
		else
		{
			waitResult = device.waitForFences(_inFlightFences[frameIndex], vk::True, UINT64_MAX);
		}
		if (waitResult != vk::Result::eSuccess)
		{
			LOG_ERROR("ComputeSystem: timed out waiting for in-flight fences");
			return;
		}
		device.resetFences(_inFlightFences[frameIndex]);

		// Record and submit compute command buffer.
		vk::CommandBuffer computeCommandBuffer = _computeCommandBuffers[frameIndex];
		computeCommandBuffer.reset();

		vk::CommandBufferBeginInfo commandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		computeCommandBuffer.begin(commandBufferBeginInfo);

		// Reset query pool inside command buffer (after fences confirm previous GPU work is done)
		if (queryManager && queryPool)
			computeCommandBuffer.resetQueryPool(queryPool, 0, 32);

		// Memory barrier: ensure prior transfer/compute writes are visible to compute reads.
		vk::MemoryBarrier computeMemoryBarrier = vk::MemoryBarrier(
			vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderWrite, // srcAccessMask
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite     // dstAccessMask
		);
		computeCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader, // srcStageMask
			vk::PipelineStageFlagBits::eComputeShader,                                        // dstStageMask
			{}, computeMemoryBarrier, {}, {});

		if (dispatchCompute)
		{
			uint32_t nextFrame = (frameIndex + 1) % _framesInFlight;

			bool usesImages = _simulationRenderer.UsesImages();

			// Clear skip grid before simulation dispatch
			if (usesImages)
			{
				vk::ImageSubresourceRange subresourceRange = vk::ImageSubresourceRange(
					vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

				vk::ClearColorValue zeroClear;
				zeroClear.setUint32({0u, 0u, 0u, 0u});

				// Transition skip grid output to transfer dst
				vk::ImageMemoryBarrier skipGridPreClearBarrier = vk::ImageMemoryBarrier(
					vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,	// srcAccessMask
					vk::AccessFlagBits::eTransferWrite,									// dstAccessMask
					vk::ImageLayout::eGeneral,											// oldLayout
					vk::ImageLayout::eTransferDstOptimal,								// newLayout
					VK_QUEUE_FAMILY_IGNORED,											// srcQueueFamilyIndex
					VK_QUEUE_FAMILY_IGNORED,											// dstQueueFamilyIndex
					*_simulationRenderer.GetSkipGridImage(nextFrame)->GetImage(),		// image
					subresourceRange													// subresourceRange
				);
				computeCommandBuffer.pipelineBarrier(
					vk::PipelineStageFlagBits::eComputeShader,
					vk::PipelineStageFlagBits::eTransfer,
					{}, {}, {}, skipGridPreClearBarrier);

				computeCommandBuffer.clearColorImage(
					*_simulationRenderer.GetSkipGridImage(nextFrame)->GetImage(),
					vk::ImageLayout::eTransferDstOptimal,
					zeroClear,
					subresourceRange);

				// Transition back to general for compute shader use
				vk::ImageMemoryBarrier skipGridPostClearBarrier = vk::ImageMemoryBarrier(
					vk::AccessFlagBits::eTransferWrite,									// srcAccessMask
					vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,	// dstAccessMask
					vk::ImageLayout::eTransferDstOptimal,								// oldLayout
					vk::ImageLayout::eGeneral,											// newLayout
					VK_QUEUE_FAMILY_IGNORED,											// srcQueueFamilyIndex
					VK_QUEUE_FAMILY_IGNORED,											// dstQueueFamilyIndex
					*_simulationRenderer.GetSkipGridImage(nextFrame)->GetImage(),		// image
					subresourceRange													// subresourceRange
				);
				computeCommandBuffer.pipelineBarrier(
					vk::PipelineStageFlagBits::eTransfer,
					vk::PipelineStageFlagBits::eComputeShader,
					{}, {}, {}, skipGridPostClearBarrier);
			}
			else
			{
				// Buffer mode: clear skip grid buffer to 0
				computeCommandBuffer.fillBuffer(
					_simulationRenderer.GetSkipGridBuffer(nextFrame)->GetBuffer(), // buffer
					0,                                                            // offset
					_simulationRenderer.GetSkipGridBuffer(nextFrame)->GetSize(),   // size
					0u                                                            // data
				);

				// Barrier: fill writes visible to compute shader
				vk::BufferMemoryBarrier skipGridClearBarrier = vk::BufferMemoryBarrier(
					vk::AccessFlagBits::eTransferWrite,                                     // srcAccessMask
					vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,     // dstAccessMask
					VK_QUEUE_FAMILY_IGNORED,                                                // srcQueueFamilyIndex
					VK_QUEUE_FAMILY_IGNORED,                                                // dstQueueFamilyIndex
					_simulationRenderer.GetSkipGridBuffer(nextFrame)->GetBuffer(),           // buffer
					0,                                                                      // offset
					_simulationRenderer.GetSkipGridBuffer(nextFrame)->GetSize()              // size
				);
				computeCommandBuffer.pipelineBarrier(
					vk::PipelineStageFlagBits::eTransfer,       // srcStageMask
					vk::PipelineStageFlagBits::eComputeShader,  // dstStageMask
					{}, {}, skipGridClearBarrier, {});
			}

			// Bind pipeline and dispatch
			_rayMarchSimulationPipeline->Bind(computeCommandBuffer, vk::PipelineBindPoint::eCompute,
			                                  static_cast<uint8_t>(frameIndex),
			                                  Pipeline::DescriptorOption{frameIndex});

			RayMarchComputePushConstants rayMarchComputePushConstants{};
			rayMarchComputePushConstants.simulationParameters = _simulationRenderer.GetSimulationParameters();
			rayMarchComputePushConstants.enableComputeSkip = enableComputeSkip ? 1u : 0u;

			computeCommandBuffer.pushConstants<RayMarchComputePushConstants>(
				_rayMarchSimulationPipeline->GetPipelineLayout(),  // layout
				vk::ShaderStageFlagBits::eCompute,                // stageFlags
				0,                                                // offset
				rayMarchComputePushConstants                      // values
			);

			// Dispatch in texel space: X is halved (packed 4-bit, 2 cells per byte)
			// Z uses owned size (when domain decomposed, dispatch only covers owned cells;
			// the shader offsets thread Z by Z_OWNED_START to skip ghost_lo)
			const glm::uvec4* gridDimensions = _simulationRenderer.GetGridDimensions();
			uint32_t dispatchX = (_simulationRenderer.GetCellStateImageWidth() + 7) / 8;
			uint32_t dispatchY = (gridDimensions->y + 7) / 8;
			uint32_t dispatchZ = (_simulationRenderer.GetOwnedZSize() + 7) / 8;

			if (queryManager) queryManager->WriteTimestamp(computeCommandBuffer, queryPool, vk::PipelineStageFlagBits::eTopOfPipe, "compute_start");
			computeCommandBuffer.dispatch(dispatchX, dispatchY, dispatchZ);
			if (queryManager) queryManager->WriteTimestamp(computeCommandBuffer, queryPool, vk::PipelineStageFlagBits::eBottomOfPipe, "compute_end");

			// Post-dispatch barriers: make writes available
			if (usesImages)
			{
				vk::ImageSubresourceRange subresourceRange = vk::ImageSubresourceRange(
					vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

				vk::ImageMemoryBarrier cellStateImageBarrier = vk::ImageMemoryBarrier(
					vk::AccessFlagBits::eShaderWrite,									// srcAccessMask
					{},																	// dstAccessMask
					vk::ImageLayout::eGeneral,											// oldLayout
					vk::ImageLayout::eGeneral,											// newLayout
					VK_QUEUE_FAMILY_IGNORED,											// srcQueueFamilyIndex
					VK_QUEUE_FAMILY_IGNORED,											// dstQueueFamilyIndex
					*_simulationRenderer.GetCellStateImage(nextFrame)->GetImage(),		// image
					subresourceRange													// subresourceRange
				);

				vk::ImageMemoryBarrier skipGridImageBarrier = vk::ImageMemoryBarrier(
					vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,	// srcAccessMask
					{},																		// dstAccessMask
					vk::ImageLayout::eGeneral,												// oldLayout
					vk::ImageLayout::eGeneral,												// newLayout
					VK_QUEUE_FAMILY_IGNORED,												// srcQueueFamilyIndex
					VK_QUEUE_FAMILY_IGNORED,												// dstQueueFamilyIndex
					*_simulationRenderer.GetSkipGridImage(nextFrame)->GetImage(),			// image
					subresourceRange														// subresourceRange
				);

				std::array<vk::ImageMemoryBarrier, 2> postDispatchImageBarriers = {cellStateImageBarrier, skipGridImageBarrier};
				computeCommandBuffer.pipelineBarrier(
					vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer, // srcStageMask
					vk::PipelineStageFlagBits::eBottomOfPipe,                                         // dstStageMask
					{},                                                                               // dependencyFlags
					{},                                                                               // memoryBarriers
					{},                                                                               // bufferMemoryBarriers
					postDispatchImageBarriers                                                         // imageMemoryBarriers
				);
			}
			else
			{
				vk::BufferMemoryBarrier cellStateBarrier = vk::BufferMemoryBarrier(
					vk::AccessFlagBits::eShaderWrite,                                       // srcAccessMask
					{},                                                                     // dstAccessMask
					VK_QUEUE_FAMILY_IGNORED,                                                // srcQueueFamilyIndex
					VK_QUEUE_FAMILY_IGNORED,                                                // dstQueueFamilyIndex
					_simulationRenderer.GetCellStateBuffer(nextFrame)->GetBuffer(),          // buffer
					0,                                                                      // offset
					_simulationRenderer.GetCellStateBuffer(nextFrame)->GetSize()             // size
				);

				vk::BufferMemoryBarrier skipGridBarrier = vk::BufferMemoryBarrier(
					vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,  // srcAccessMask
					{},                                                                     // dstAccessMask
					VK_QUEUE_FAMILY_IGNORED,                                                // srcQueueFamilyIndex
					VK_QUEUE_FAMILY_IGNORED,                                                // dstQueueFamilyIndex
					_simulationRenderer.GetSkipGridBuffer(nextFrame)->GetBuffer(),           // buffer
					0,                                                                      // offset
					_simulationRenderer.GetSkipGridBuffer(nextFrame)->GetSize()              // size
				);

				std::array<vk::BufferMemoryBarrier, 2> postDispatchBarriers = {cellStateBarrier, skipGridBarrier};
				computeCommandBuffer.pipelineBarrier(
					vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer, // srcStageMask
					vk::PipelineStageFlagBits::eBottomOfPipe,                                         // dstStageMask
					{},                                                                               // dependencyFlags
					{},                                                                               // memoryBarriers
					postDispatchBarriers,                                                             // bufferMemoryBarriers
					{}                                                                                // imageMemoryBarriers
				);
			}
		}

		computeCommandBuffer.end();

		_completedSemaphoreSignalValues[frameIndex]++;

		vk::TimelineSemaphoreSubmitInfo timelineSemaphoreSubmitInfo = vk::TimelineSemaphoreSubmitInfo(
			0,                                              // waitSemaphoreValueCount
			nullptr,                                        // pWaitSemaphoreValues
			1,                                              // signalSemaphoreValueCount
			&_completedSemaphoreSignalValues[frameIndex]    // pSignalSemaphoreValues
		);

		vk::SubmitInfo submitInfo = vk::SubmitInfo(
			0,                                 // waitSemaphoreCount
			nullptr,                           // pWaitSemaphores
			nullptr,                           // pWaitDstStageMask
			1,                                 // commandBufferCount
			&computeCommandBuffer,             // pCommandBuffers
			1,                                 // signalSemaphoreCount
			&_completedSemaphores[frameIndex]  // pSignalSemaphores
		);
		submitInfo.pNext = &timelineSemaphoreSubmitInfo;

		_deviceContext.GetComputeQueue().submit(submitInfo, _inFlightFences[frameIndex]);

	}
}
