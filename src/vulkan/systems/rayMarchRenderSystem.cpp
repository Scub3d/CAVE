#include "rayMarchRenderSystem.h"
#include "../../common/logger.h"
#include "../../common/structs.h"

namespace Cave
{
	RayMarchRenderSystem::RayMarchRenderSystem(DeviceContext& deviceContext, VulkanSimulationRenderer& simulationRenderer,
											   vk::Extent2D renderExtent, bool compositeEnabled)
		: _deviceContext{deviceContext}, _simulationRenderer{simulationRenderer},
		  _framesInFlight{deviceContext.GetFramesInFlight()}, _renderExtent{renderExtent},
		  _compositeEnabled{compositeEnabled}
	{
		// Composite mode uses R16G16B16A16Sfloat for depth precision in alpha channel.
		// Standard mode uses the device's default color format (typically B8G8R8A8).
		_colorFormat = compositeEnabled ? vk::Format::eR16G16B16A16Sfloat : _deviceContext.GetColorFormat();
		_graphicsCommandPool = _deviceContext.GetGraphicsCommandPool();

		_inFlightFences.resize(_framesInFlight);
		_renderFinishedSemaphores.resize(_framesInFlight);
		_renderFinishedSemaphoreSignalValues.resize(_framesInFlight, 0);

		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			_inFlightFences[frameIndex] = _deviceContext.CreateFence();
			_renderFinishedSemaphores[frameIndex] = _deviceContext.CreateTimelineSemaphore(0);
		}

		BuildBoundingBoxMesh();
		BuildCameraBuffers();
		BuildAttachments();
		BuildDescriptors();
		BuildShaders();
		BuildPipeline();
		BuildCommandBuffers();
	}

	RayMarchRenderSystem::~RayMarchRenderSystem()
	{
		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			_deviceContext.GetDevice().destroyFence(_inFlightFences[frameIndex]);
			_deviceContext.GetDevice().destroySemaphore(_renderFinishedSemaphores[frameIndex]);
		}
	}

	void RayMarchRenderSystem::BuildBoundingBoxMesh()
	{
		// Unit cube vertices [-1, 1] in each axis
		std::vector<Vertex> vertices = {
			{{-1.0f, -1.0f, -1.0f, 1.0f}},
			{{ 1.0f, -1.0f, -1.0f, 1.0f}},
			{{ 1.0f,  1.0f, -1.0f, 1.0f}},
			{{-1.0f,  1.0f, -1.0f, 1.0f}},
			{{-1.0f, -1.0f,  1.0f, 1.0f}},
			{{ 1.0f, -1.0f,  1.0f, 1.0f}},
			{{ 1.0f,  1.0f,  1.0f, 1.0f}},
			{{-1.0f,  1.0f,  1.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0, // -Z face
			4, 6, 5, 6, 4, 7, // +Z face
			0, 4, 5, 5, 1, 0, // -Y face
			2, 6, 7, 7, 3, 2, // +Y face
			0, 3, 7, 7, 4, 0, // -X face
			1, 5, 6, 6, 2, 1, // +X face
		};

		_boundingBoxIndexCount = static_cast<uint32_t>(indices.size());

		_boundingBoxVertexBuffer = std::make_unique<Buffer>(
			_deviceContext,
			sizeof(Vertex) * vertices.size(),
			vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			VMA_MEMORY_USAGE_GPU_ONLY, 0, false, 0, "Ray March Bounding Box Vertices");
		_boundingBoxVertexBuffer->Upload(vertices.data(), static_cast<uint32_t>(sizeof(Vertex) * vertices.size()));

		_boundingBoxIndexBuffer = std::make_unique<Buffer>(
			_deviceContext,
			sizeof(uint32_t) * indices.size(),
			vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			VMA_MEMORY_USAGE_GPU_ONLY, 0, false, 0, "Ray March Bounding Box Indices");
		_boundingBoxIndexBuffer->Upload(indices.data(), static_cast<uint32_t>(sizeof(uint32_t) * indices.size()));
	}

	void RayMarchRenderSystem::BuildCameraBuffers()
	{
		_cameraUniformBuffers.resize(_framesInFlight);
		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			_cameraUniformBuffers[frameIndex] = Buffer::Uniform(_deviceContext, sizeof(CameraData));
		}
	}

	void RayMarchRenderSystem::BuildAttachments()
	{
		vk::Extent3D attachmentExtent(_renderExtent.width, _renderExtent.height, 1);

		_colorAttachments.resize(_framesInFlight);
		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			vk::ImageUsageFlags colorUsage = vk::ImageUsageFlagBits::eColorAttachment
				| vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled;
			if (_compositeEnabled)
				colorUsage |= vk::ImageUsageFlagBits::eStorage;

			_colorAttachments[frameIndex] = std::make_shared<Image>(
				_deviceContext,
				_colorFormat,
				1, 1,
				attachmentExtent,
				vk::ImageTiling::eOptimal,
				colorUsage,
				vk::MemoryPropertyFlagBits::eDeviceLocal,
				vk::ImageCreateFlags(),
				vk::ImageAspectFlagBits::eColor,
				vk::ImageViewType::e2D,
				vk::SharingMode::eExclusive,
				std::vector<uint32_t>{});
		}

		vk::Format depthFormat = Image::FindSupportedFormat(
			_deviceContext.GetPhysicalDevice(),
			{vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
			vk::ImageTiling::eOptimal,
			vk::FormatFeatureFlagBits::eDepthStencilAttachment);

		_depthAttachment = std::make_unique<Image>(
			_deviceContext,
			depthFormat,
			1, 1,
			attachmentExtent,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eDepthStencilAttachment,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vk::ImageCreateFlags(),
			vk::ImageAspectFlagBits::eDepth,
			vk::ImageViewType::e2D,
			vk::SharingMode::eExclusive,
			std::vector<uint32_t>{});
	}

	void RayMarchRenderSystem::BuildDescriptors()
	{
		_descriptor = std::make_shared<Descriptor>(_deviceContext);
		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			// binding 0 — camera UBO
			_descriptor->BindBufferToDescriptorSet(0, vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, _cameraUniformBuffers[frameIndex]);

			uint32_t nextFrame = (frameIndex + 1) % _framesInFlight;

			if (_simulationRenderer.UsesImages())
			{
				// binding 1 — cell state image (read in fragment shader)
				_descriptor->BindImageToDescriptorSet(1, vk::DescriptorType::eStorageImage,
					vk::ShaderStageFlagBits::eFragment, _simulationRenderer.GetCellStateImage(nextFrame));

				// binding 2 — skip grid image (read in fragment shader)
				_descriptor->BindImageToDescriptorSet(2, vk::DescriptorType::eStorageImage,
					vk::ShaderStageFlagBits::eFragment, _simulationRenderer.GetSkipGridImage(nextFrame));
			}
			else
			{
				// binding 1 — cell state buffer (read in fragment shader)
				_descriptor->BindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer,
					vk::ShaderStageFlagBits::eFragment, _simulationRenderer.GetCellStateBuffer(nextFrame));

				// binding 2 — skip grid buffer (read in fragment shader)
				_descriptor->BindBufferToDescriptorSet(2, vk::DescriptorType::eStorageBuffer,
					vk::ShaderStageFlagBits::eFragment, _simulationRenderer.GetSkipGridBuffer(nextFrame));
			}
		}
		_descriptor->Build();
	}

	void RayMarchRenderSystem::RebuildDescriptors()
	{
		_descriptor.reset();
		BuildDescriptors();
		// Rebuild pipeline since it references the descriptor layout
		_graphicsPipeline.reset();
		BuildPipeline();
		LOG_INFO("RayMarchRenderSystem: descriptors and pipeline rebuilt");
	}

	void RayMarchRenderSystem::BuildShaders()
	{
		_vertexShaderModule = _simulationRenderer.GetRayMarchVertexShaderModule();
		_fragmentShaderModule = _simulationRenderer.GetRayMarchFragmentShaderModule();
		LOG_INFO("Ray march shader modules — vertex: {}, fragment: {}",
			_vertexShaderModule ? "OK" : "FAILED",
			_fragmentShaderModule ? "OK" : "FAILED");
	}

	void RayMarchRenderSystem::BuildPipeline()
	{
		auto vertexShader = std::make_shared<Shader>(_deviceContext, _vertexShaderModule);
		auto fragmentShader = std::make_shared<Shader>(_deviceContext, _fragmentShaderModule);
		_graphicsPipeline = std::make_unique<GraphicsPipeline>(_deviceContext, vertexShader, fragmentShader);

		// Vertex input: single binding for the bounding box cube (vec4 position)
		_graphicsPipeline->SetVertexInput(
			{
				vk::VertexInputBindingDescription(
					0,								// binding
					sizeof(Vertex),					// stride
					vk::VertexInputRate::eVertex	// inputRate
				),
			},
			{
				vk::VertexInputAttributeDescription(
					0,									// location
					0,									// binding
					vk::Format::eR32G32B32A32Sfloat,	// format
					offsetof(Vertex, position)			// offset
				),
			});

		vk::Format depthFormat = Image::FindSupportedFormat(
			_deviceContext.GetPhysicalDevice(),
			{vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
			vk::ImageTiling::eOptimal,
			vk::FormatFeatureFlagBits::eDepthStencilAttachment);

		_graphicsPipeline->SetCullMode(vk::CullModeFlagBits::eNone);
		// Ray march outputs a solid color per pixel (via discard-or-write) and encodes
		// distance in the alpha channel when compositing. Blending with eSrcAlpha would
		// multiply RGB by the distance value, producing saturation/negative feedback.
		_graphicsPipeline->SetBlendEnabled(false);
		_graphicsPipeline->SetDynamicRenderingInfo(_colorFormat, depthFormat);
		_graphicsPipeline->AddDescriptorSet(0, _descriptor);
		_graphicsPipeline->AddPushConstant(
			vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
			0, sizeof(RayMarchPushConstants));
		_graphicsPipeline->Build();
	}

	void RayMarchRenderSystem::BuildCommandBuffers()
	{
		_graphicsCommandBuffers.resize(_framesInFlight);
		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			_graphicsCommandBuffers[frameIndex] = _deviceContext.CreateCommandBuffer(_graphicsCommandPool);
		}
	}

	void RayMarchRenderSystem::UpdateCameraBuffer(uint32_t frameIndex, const CameraData& cameraData)
	{
		_cameraUniformBuffers[frameIndex]->Upload(&cameraData, sizeof(CameraData));
	}

	void RayMarchRenderSystem::RenderFrame(uint32_t frameIndex, vk::Semaphore computeCompletedSemaphore,
										   uint64_t computeCompletedSemaphoreWaitValue, const CameraData& cameraData,
										   const RayMarchPushConstants& rayMarchPushConstants,
										   std::shared_ptr<QueryManager> queryManager, vk::QueryPool queryPool,
										   bool waitForCompute)
	{
		vk::Device device = _deviceContext.GetDevice();

		auto waitResult = device.waitForFences(_inFlightFences[frameIndex], vk::True, UINT64_MAX);
		if (waitResult != vk::Result::eSuccess)
		{
			LOG_ERROR("RayMarchRenderSystem: timed out waiting for in-flight fence");
			return;
		}
		device.resetFences(_inFlightFences[frameIndex]);

		UpdateCameraBuffer(frameIndex, cameraData);
		_rayMarchPushConstants = rayMarchPushConstants;

		vk::CommandBuffer graphicsCommandBuffer = _graphicsCommandBuffers[frameIndex];
		graphicsCommandBuffer.reset();

		vk::CommandBufferBeginInfo commandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		graphicsCommandBuffer.begin(commandBufferBeginInfo);

		vk::ImageSubresourceRange colorRange = vk::ImageSubresourceRange(
			vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
		vk::ImageSubresourceRange depthRange = vk::ImageSubresourceRange(
			vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);

		// Transition color attachment: undefined -> color attachment optimal
		vk::ImageMemoryBarrier colorBarrierIn = vk::ImageMemoryBarrier(
			{},															// srcAccessMask
			vk::AccessFlagBits::eColorAttachmentWrite,					// dstAccessMask
			vk::ImageLayout::eUndefined,								// oldLayout
			vk::ImageLayout::eColorAttachmentOptimal,					// newLayout
			VK_QUEUE_FAMILY_IGNORED,									// srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,									// dstQueueFamilyIndex
			*_colorAttachments[frameIndex]->GetImage(),					// image
			colorRange													// subresourceRange
		);
		graphicsCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
			{}, {}, {}, colorBarrierIn);

		// Transition depth attachment: undefined -> depth stencil optimal
		vk::ImageMemoryBarrier depthBarrierIn = vk::ImageMemoryBarrier(
			{},															// srcAccessMask
			vk::AccessFlagBits::eDepthStencilAttachmentWrite,			// dstAccessMask
			vk::ImageLayout::eUndefined,								// oldLayout
			vk::ImageLayout::eDepthStencilAttachmentOptimal,			// newLayout
			VK_QUEUE_FAMILY_IGNORED,									// srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,									// dstQueueFamilyIndex
			*_depthAttachment->GetImage(),								// image
			depthRange													// subresourceRange
		);
		graphicsCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eEarlyFragmentTests,
			{}, {}, {}, depthBarrierIn);

		// Begin dynamic rendering
		// Composite mode clears alpha to 0 so non-hit pixels are distinguishable from hits.
		float clearAlpha = _compositeEnabled ? 0.0f : 1.0f;
		vk::ClearValue colorClear = vk::ClearValue(
			vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, clearAlpha}});
		vk::ClearValue depthClear = vk::ClearValue(
			vk::ClearDepthStencilValue{1.0f, 0});

		vk::RenderingAttachmentInfo colorAttachmentInfo = vk::RenderingAttachmentInfo(
			*_colorAttachments[frameIndex]->GetImageView(),	// imageView
			vk::ImageLayout::eColorAttachmentOptimal,		// imageLayout
			vk::ResolveModeFlagBits::eNone,					// resolveMode
			{},												// resolveImageView
			{},												// resolveImageLayout
			vk::AttachmentLoadOp::eClear,					// loadOp
			vk::AttachmentStoreOp::eStore,					// storeOp
			colorClear										// clearValue
		);

		vk::RenderingAttachmentInfo depthAttachmentInfo = vk::RenderingAttachmentInfo(
			*_depthAttachment->GetImageView(),					// imageView
			vk::ImageLayout::eDepthStencilAttachmentOptimal,	// imageLayout
			vk::ResolveModeFlagBits::eNone,						// resolveMode
			{},													// resolveImageView
			{},													// resolveImageLayout
			vk::AttachmentLoadOp::eClear,						// loadOp
			vk::AttachmentStoreOp::eDontCare,					// storeOp
			depthClear											// clearValue
		);

		vk::RenderingInfo renderingInfo = vk::RenderingInfo(
			{},												// flags
			vk::Rect2D{{0, 0}, _renderExtent},				// renderArea
			1,												// layerCount
			0,												// viewMask
			colorAttachmentInfo,							// colorAttachmentCount + pColorAttachments
			&depthAttachmentInfo,							// pDepthAttachment
			nullptr											// pStencilAttachment
		);

		graphicsCommandBuffer.beginRendering(renderingInfo);

		// Set viewport and scissor (dynamic state)
		vk::Viewport viewport = vk::Viewport(
			0.0f,												// x
			static_cast<float>(_renderExtent.height),			// y
			static_cast<float>(_renderExtent.width),			// width
			-static_cast<float>(_renderExtent.height),			// height (negative flips Y)
			0.0f,												// minDepth
			1.0f												// maxDepth
		);
		vk::Rect2D scissor = vk::Rect2D({0, 0}, _renderExtent);
		graphicsCommandBuffer.setViewport(0, viewport);
		graphicsCommandBuffer.setScissor(0, scissor);

		// Bind pipeline + descriptors
		_graphicsPipeline->Bind(graphicsCommandBuffer, vk::PipelineBindPoint::eGraphics,
								static_cast<uint8_t>(frameIndex), Pipeline::DescriptorOption{frameIndex});

		// Push ray march constants
		graphicsCommandBuffer.pushConstants<RayMarchPushConstants>(
			_graphicsPipeline->GetPipelineLayout(),								// layout
			vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,	// stageFlags
			0,																		// offset
			_rayMarchPushConstants												// values
		);

		// Bind bounding box mesh and draw
		vk::Buffer vertexBuffers[] = {_boundingBoxVertexBuffer->GetBuffer()};
		vk::DeviceSize offsets[] = {0};
		graphicsCommandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
		graphicsCommandBuffer.bindIndexBuffer(_boundingBoxIndexBuffer->GetBuffer(), 0, vk::IndexType::eUint32);
		if (queryManager) queryManager->WriteTimestamp(graphicsCommandBuffer, queryPool, vk::PipelineStageFlagBits::eTopOfPipe, "render_start");
		graphicsCommandBuffer.drawIndexed(_boundingBoxIndexCount, 1, 0, 0, 0);
		if (queryManager) queryManager->WriteTimestamp(graphicsCommandBuffer, queryPool, vk::PipelineStageFlagBits::eBottomOfPipe, "render_end");

		graphicsCommandBuffer.endRendering();

		// Transition color attachment: color attachment -> transfer src
		vk::ImageMemoryBarrier colorBarrierOut = vk::ImageMemoryBarrier(
			vk::AccessFlagBits::eColorAttachmentWrite,		// srcAccessMask
			vk::AccessFlagBits::eTransferRead,				// dstAccessMask
			vk::ImageLayout::eColorAttachmentOptimal,		// oldLayout
			vk::ImageLayout::eTransferSrcOptimal,			// newLayout
			VK_QUEUE_FAMILY_IGNORED,						// srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,						// dstQueueFamilyIndex
			*_colorAttachments[frameIndex]->GetImage(),		// image
			colorRange										// subresourceRange
		);
		graphicsCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTransfer,
			{}, {}, {}, colorBarrierOut);

		graphicsCommandBuffer.end();

		_renderFinishedSemaphoreSignalValues[frameIndex]++;

		if (waitForCompute)
		{
			// Normal path: wait for compute to finish before fragment shader reads cell data
			vk::TimelineSemaphoreSubmitInfo timelineSemaphoreSubmitInfo = vk::TimelineSemaphoreSubmitInfo(
				1,													// waitSemaphoreValueCount
				&computeCompletedSemaphoreWaitValue,				// pWaitSemaphoreValues
				1,													// signalSemaphoreValueCount
				&_renderFinishedSemaphoreSignalValues[frameIndex]	// pSignalSemaphoreValues
			);

			vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eFragmentShader;
			vk::SubmitInfo submitInfo = vk::SubmitInfo(
				1, &computeCompletedSemaphore, &waitStage,
				1, &graphicsCommandBuffer,
				1, &_renderFinishedSemaphores[frameIndex]);
			submitInfo.pNext = &timelineSemaphoreSubmitInfo;
			_deviceContext.GetGraphicsQueue().submit(submitInfo, _inFlightFences[frameIndex]);
		}
		else
		{
			// Paused path: no compute dependency, just render and signal fence
			_deviceContext.GetGraphicsQueue().submit(
				vk::SubmitInfo(0, nullptr, nullptr, 1, &graphicsCommandBuffer, 0, nullptr),
				_inFlightFences[frameIndex]);
		}
	}

	vk::Image RayMarchRenderSystem::GetOutputImage(uint32_t frameIndex) const
	{
		return *_colorAttachments[frameIndex]->GetImage();
	}

	vk::ImageView RayMarchRenderSystem::GetOutputImageView(uint32_t frameIndex) const
	{
		return *_colorAttachments[frameIndex]->GetImageView();
	}
}
