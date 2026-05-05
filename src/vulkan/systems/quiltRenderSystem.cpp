#include "quiltRenderSystem.h"
#include "../buffer.h"
#include "../../common/logger.h"

#include <glm/gtc/matrix_transform.hpp>
#include <cstring>
#include <cmath>

namespace Cave
{
	namespace
	{
		void HsvToRgb255(float h, float s, float v, uint8_t& r, uint8_t& g, uint8_t& b)
		{
			float c = v * s;
			float hp = h * 6.0f;
			float x = c * (1.0f - std::fabs(std::fmod(hp, 2.0f) - 1.0f));
			float r1 = 0, g1 = 0, b1 = 0;
			if      (hp < 1) { r1 = c; g1 = x; }
			else if (hp < 2) { r1 = x; g1 = c; }
			else if (hp < 3) { g1 = c; b1 = x; }
			else if (hp < 4) { g1 = x; b1 = c; }
			else if (hp < 5) { r1 = x; b1 = c; }
			else             { r1 = c; b1 = x; }
			float m = v - c;
			r = static_cast<uint8_t>((r1 + m) * 255.0f);
			g = static_cast<uint8_t>((g1 + m) * 255.0f);
			b = static_cast<uint8_t>((b1 + m) * 255.0f);
		}
	}

	QuiltRenderSystem::QuiltRenderSystem(DeviceContext& deviceContext)
		: _deviceContext{deviceContext}
	{
		_quiltImage = Image::CreateExportable(
			_deviceContext,
			vk::Format::eR8G8B8A8Unorm,
			vk::Extent3D{kQuiltWidth, kQuiltHeight, 1},
			vk::ImageUsageFlagBits::eTransferDst
				| vk::ImageUsageFlagBits::eTransferSrc
				| vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eColorAttachment);
		LOG_INFO("QuiltRenderSystem: allocated exportable {}x{} RGBA8 quilt image (memorySize={} MiB)",
			kQuiltWidth, kQuiltHeight, _quiltImage->GetAllocatedMemorySize() / (1024 * 1024));

		_graphicsCommandPool = _deviceContext.GetGraphicsCommandPool();
	}

	QuiltRenderSystem::~QuiltRenderSystem()
	{
		_deviceContext.GetDevice().waitIdle();
		if (_renderFence)
		{
			_deviceContext.GetDevice().destroyFence(_renderFence);
			_renderFence = vk::Fence{};
		}
	}

	// ---- Real-render path setup --------------------------------------------

	void QuiltRenderSystem::EnsureRealRenderPath(VulkanSimulationRenderer& simulationRenderer)
	{
		if (_realRenderPathReady) return;

		BuildBoundingBoxMesh();
		BuildDepthAttachment();
		BuildCameraUbosAndDescriptors(simulationRenderer);
		BuildGraphicsPipeline(simulationRenderer);

		_renderCommandBuffer = _deviceContext.CreateCommandBuffer(_graphicsCommandPool);
		_renderFence = _deviceContext.CreateFence();
		// Initial state = signaled so the first wait succeeds without blocking.
		// CreateFence (in this codebase) returns unsignaled — bring up to signaled
		// by submitting nothing through it once. Simpler: reset on first use only.

		_realRenderPathReady = true;
		LOG_INFO("QuiltRenderSystem: real-render path ready (48 views, depth attachment {}x{})",
			kQuiltWidth, kQuiltHeight);
	}

	void QuiltRenderSystem::BuildBoundingBoxMesh()
	{
		// Same unit cube + face indices as RayMarchRenderSystem (we share the
		// shader so the vertex format must match exactly).
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
			0, 1, 2, 2, 3, 0,
			4, 6, 5, 6, 4, 7,
			0, 4, 5, 5, 1, 0,
			2, 6, 7, 7, 3, 2,
			0, 3, 7, 7, 4, 0,
			1, 5, 6, 6, 2, 1,
		};
		_boundingBoxIndexCount = static_cast<uint32_t>(indices.size());

		_boundingBoxVertexBuffer = std::make_unique<Buffer>(
			_deviceContext, sizeof(Vertex) * vertices.size(),
			vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			VMA_MEMORY_USAGE_GPU_ONLY, 0, false, 0, "Quilt BBox Vertices");
		_boundingBoxVertexBuffer->Upload(vertices.data(), static_cast<uint32_t>(sizeof(Vertex) * vertices.size()));

		_boundingBoxIndexBuffer = std::make_unique<Buffer>(
			_deviceContext, sizeof(uint32_t) * indices.size(),
			vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			VMA_MEMORY_USAGE_GPU_ONLY, 0, false, 0, "Quilt BBox Indices");
		_boundingBoxIndexBuffer->Upload(indices.data(), static_cast<uint32_t>(sizeof(uint32_t) * indices.size()));
	}

	void QuiltRenderSystem::BuildDepthAttachment()
	{
		vk::Format depthFormat = Image::FindSupportedFormat(
			_deviceContext.GetPhysicalDevice(),
			{vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
			vk::ImageTiling::eOptimal,
			vk::FormatFeatureFlagBits::eDepthStencilAttachment);

		_depthAttachment = std::make_unique<Image>(
			_deviceContext, depthFormat,
			1, 1, vk::Extent3D{kQuiltWidth, kQuiltHeight, 1},
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eDepthStencilAttachment,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vk::ImageCreateFlags(),
			vk::ImageAspectFlagBits::eDepth,
			vk::ImageViewType::e2D,
			vk::SharingMode::eExclusive,
			std::vector<uint32_t>{});
	}

	void QuiltRenderSystem::BuildCameraUbosAndDescriptors(VulkanSimulationRenderer& simulationRenderer)
	{
		const uint32_t framesInFlight = simulationRenderer.GetFramesInFlight();

		_viewCameraBuffers.resize(kQuiltViewCount);
		_viewDescriptorsPerFrame.assign(framesInFlight, {});

		for (uint32_t v = 0; v < kQuiltViewCount; v++)
		{
			_viewCameraBuffers[v] = Buffer::Uniform(_deviceContext, sizeof(CameraData));
		}

		for (uint32_t f = 0; f < framesInFlight; f++)
		{
			_viewDescriptorsPerFrame[f].resize(kQuiltViewCount);
			for (uint32_t v = 0; v < kQuiltViewCount; v++)
			{
				auto descriptor = std::make_shared<Descriptor>(_deviceContext);

				// Binding 0 — camera UBO (per-view). Vertex+Fragment stages match the
				// existing rayMarch shaders' declared binding.
				descriptor->BindBufferToDescriptorSet(0, vk::DescriptorType::eUniformBuffer,
					vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, _viewCameraBuffers[v]);

				// Bindings 1+2 — cell state + skip grid for THIS frame's read
				// buffer. Caller passes renderFromFrameIndex matching the
				// buffer the compute pass just wrote to.
				if (simulationRenderer.UsesImages())
				{
					descriptor->BindImageToDescriptorSet(1, vk::DescriptorType::eStorageImage,
						vk::ShaderStageFlagBits::eFragment, simulationRenderer.GetCellStateImage(f));
					descriptor->BindImageToDescriptorSet(2, vk::DescriptorType::eStorageImage,
						vk::ShaderStageFlagBits::eFragment, simulationRenderer.GetSkipGridImage(f));
				}
				else
				{
					descriptor->BindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer,
						vk::ShaderStageFlagBits::eFragment, simulationRenderer.GetCellStateBuffer(f));
					descriptor->BindBufferToDescriptorSet(2, vk::DescriptorType::eStorageBuffer,
						vk::ShaderStageFlagBits::eFragment, simulationRenderer.GetSkipGridBuffer(f));
				}
				descriptor->Build();
				_viewDescriptorsPerFrame[f][v] = std::move(descriptor);
			}
		}
	}

	void QuiltRenderSystem::BuildGraphicsPipeline(VulkanSimulationRenderer& simulationRenderer)
	{
		auto vertexShader = std::make_shared<Shader>(_deviceContext, simulationRenderer.GetRayMarchVertexShaderModule());
		auto fragmentShader = std::make_shared<Shader>(_deviceContext, simulationRenderer.GetRayMarchFragmentShaderModule());

		_graphicsPipeline = std::make_unique<GraphicsPipeline>(_deviceContext, vertexShader, fragmentShader);

		_graphicsPipeline->SetVertexInput(
			{ vk::VertexInputBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex) },
			{ vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, position)) });

		vk::Format depthFormat = Image::FindSupportedFormat(
			_deviceContext.GetPhysicalDevice(),
			{vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
			vk::ImageTiling::eOptimal,
			vk::FormatFeatureFlagBits::eDepthStencilAttachment);

		_graphicsPipeline->SetCullMode(vk::CullModeFlagBits::eNone);
		_graphicsPipeline->SetBlendEnabled(false);
		_graphicsPipeline->SetDynamicRenderingInfo(vk::Format::eR8G8B8A8Unorm, depthFormat);

		// Bind one of the descriptor sets — the layouts are identical across all
		// (frameIndex, view) pairs (same bindings 0/1/2 with same types/stages).
		// The pipeline layout only needs the layout, not the actual sets, so any
		// descriptor works.
		_graphicsPipeline->AddDescriptorSet(0, _viewDescriptorsPerFrame[0][0]);
		_graphicsPipeline->AddPushConstant(
			vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
			0, sizeof(RayMarchPushConstants));
		_graphicsPipeline->Build();
	}

	// ---- Per-view camera math ------------------------------------------------

	CameraData QuiltRenderSystem::ComputePerViewCamera(const CameraData& base, float t, float viewconeRadians) const
	{
		// Lenticular displays need PARALLEL cameras with an off-axis (sheared)
		// projection — NOT a rotated lookAt that toes each camera toward the
		// target. Toe-in introduces vertical disparity and keystone distortion
		// across views; the user perceives this as discrete "snaps" when
		// moving the head horizontally instead of smooth parallax.
		//
		// Pipeline per view:
		//   1. Translate the eye sideways along the base camera's right axis.
		//   2. Build the view matrix with the SAME orientation as the base
		//      (forward + up unchanged).
		//   3. Shear the projection matrix horizontally so the convergence
		//      plane (= the target distance) re-centers the image.
		//
		// All cameras share a focus plane at the orbit target; closer/farther
		// objects show parallax while the orbit center stays anchored.

		glm::vec3 baseEye = glm::vec3(base.cameraPosition);
		glm::vec3 target  = glm::vec3(0.0f);

		float focusDistance = glm::length(baseEye - target);
		if (focusDistance < 1e-4f) focusDistance = 1.0f;

		// Extract base camera basis vectors from the view matrix's rows
		// (the inverse rotation of the camera). glm matrices are column-major
		// so element [col][row] picks columns first.
		glm::vec3 cameraRight   = glm::normalize(glm::vec3(base.view[0][0], base.view[1][0], base.view[2][0]));
		glm::vec3 cameraUp      = glm::normalize(glm::vec3(base.view[0][1], base.view[1][1], base.view[2][1]));
		glm::vec3 cameraForward = -glm::normalize(glm::vec3(base.view[0][2], base.view[1][2], base.view[2][2]));

		float halfViewcone = viewconeRadians * 0.5f;
		float offsetMag = std::tan(halfViewcone) * focusDistance;

		float s = (t - 0.5f) * 2.0f; // map [0,1] -> [-1,+1]
		float eyeShift = s * offsetMag;
		glm::vec3 shiftedEye = baseEye + cameraRight * eyeShift;

		CameraData perView = base;
		perView.cameraPosition = glm::vec4(shiftedEye, 1.0f);

		// 2. Parallel view: same forward/up, just translated eye.
		perView.view = glm::lookAt(shiftedEye, shiftedEye + cameraForward, cameraUp);

		// 3. Off-axis projection shear. For a perspective matrix where
		//    P[0][0] = 1/(aspect * tan(fov/2)), we re-center the frustum on
		//    the target by adjusting P[2][0] (horizontal asymmetry term):
		//        P[2][0] -= eyeShift * P[0][0] / focusDistance
		//    Derivation: shifting the frustum's left/right by `shift` (at the
		//    near plane) gives P[2][0] = -2*shift / width = -shift * P[0][0] / n.
		//    With shift = eyeShift * n / focusDistance, the n cancels.
		perView.projection[2][0] -= eyeShift * base.projection[0][0] / focusDistance;

		return perView;
	}

	// ---- Real-render frame -------------------------------------------------

	void QuiltRenderSystem::RenderFrame(VulkanSimulationRenderer& simulationRenderer,
	                                    const CameraData& baseCameraData,
	                                    const RayMarchPushConstants& pushConstants,
	                                    float viewconeRadians,
	                                    vk::Semaphore computeWaitSemaphore,
	                                    uint64_t computeWaitValue,
	                                    ExternalBinarySemaphore* signalSem,
	                                    uint32_t renderFromFrameIndex)
	{
		EnsureRealRenderPath(simulationRenderer);

		vk::Device device = _deviceContext.GetDevice();

		// Wait on the previous frame's render fence so we can safely reuse the
		// command buffer + UBOs.
		if (_renderFence)
		{
			// First call: the fence has never been signaled. Vulkan-Hpp's
			// waitForFences with an unsignaled fence + UINT64_MAX would block
			// forever. Skip the wait if this is the very first frame.
			static bool firstFrame = true;
			if (!firstFrame)
			{
				vk::Result waitResult = device.waitForFences(_renderFence, vk::True, UINT64_MAX);
				if (waitResult != vk::Result::eSuccess)
				{
					LOG_ERROR("QuiltRenderSystem: timed out waiting for render fence");
					return;
				}
				device.resetFences(_renderFence);
			}
			firstFrame = false;
		}

		// Update the 48 per-view camera UBOs.
		for (uint32_t v = 0; v < kQuiltViewCount; v++)
		{
			float t = (kQuiltViewCount > 1)
				? static_cast<float>(v) / static_cast<float>(kQuiltViewCount - 1)
				: 0.5f;
			CameraData perView = ComputePerViewCamera(baseCameraData, t, viewconeRadians);
			_viewCameraBuffers[v]->Upload(&perView, sizeof(CameraData));
		}

		_renderCommandBuffer.reset();
		vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		_renderCommandBuffer.begin(beginInfo);

		vk::ImageSubresourceRange colorRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
		vk::ImageSubresourceRange depthRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);

		// Transition quilt to ColorAttachmentOptimal (from previous Undefined or General).
		vk::ImageLayout oldLayout = _initialLayoutTransitioned ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined;
		vk::ImageMemoryBarrier quiltIn(
			(_initialLayoutTransitioned ? vk::AccessFlagBits::eShaderRead : vk::AccessFlagBits{}),
			vk::AccessFlagBits::eColorAttachmentWrite,
			oldLayout, vk::ImageLayout::eColorAttachmentOptimal,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			*_quiltImage->GetImage(), colorRange);
		_renderCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eColorAttachmentOutput,
			{}, {}, {}, quiltIn);

		// Transition depth to DepthStencilAttachmentOptimal.
		vk::ImageMemoryBarrier depthIn(
			{}, vk::AccessFlagBits::eDepthStencilAttachmentWrite,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			*_depthAttachment->GetImage(), depthRange);
		_renderCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eEarlyFragmentTests,
			{}, {}, {}, depthIn);

		// Single dynamic-rendering scope covering the whole quilt. We do 48
		// viewport-tiled draws inside it — each tile gets its own camera UBO
		// (via descriptor set switch).
		vk::ClearValue colorClear(vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}});
		vk::ClearValue depthClear(vk::ClearDepthStencilValue{1.0f, 0});

		vk::RenderingAttachmentInfo colorAtt(
			*_quiltImage->GetImageView(), vk::ImageLayout::eColorAttachmentOptimal,
			vk::ResolveModeFlagBits::eNone, {}, {},
			vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, colorClear);
		vk::RenderingAttachmentInfo depthAtt(
			*_depthAttachment->GetImageView(), vk::ImageLayout::eDepthStencilAttachmentOptimal,
			vk::ResolveModeFlagBits::eNone, {}, {},
			vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, depthClear);

		vk::RenderingInfo renderInfo(
			{}, vk::Rect2D{{0, 0}, {kQuiltWidth, kQuiltHeight}},
			1, 0, colorAtt, &depthAtt, nullptr);
		_renderCommandBuffer.beginRendering(renderInfo);

		vk::Buffer vertexBuffers[] = { _boundingBoxVertexBuffer->GetBuffer() };
		vk::DeviceSize offsets[] = { 0 };
		_renderCommandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
		_renderCommandBuffer.bindIndexBuffer(_boundingBoxIndexBuffer->GetBuffer(), 0, vk::IndexType::eUint32);

		// 48 viewport-tiled draws. Looking Glass quilt convention places view 0
		// at the BOTTOM-LEFT of the texture and fills left-to-right, bottom-to-top.
		// Our texture is rendered with Y-down indexing, so the "bottom" tile-row
		// corresponds to the highest tile-row index — flip the row mapping so
		// view 0 lands at row=kQuiltRows-1 (the bottom tile row in image-Y-up
		// terms, which is what Bridge samples). Without this flip, vertically-
		// adjacent views in the quilt are interpreted as views ~kQuiltColumns
		// apart by the lenticular optics, producing the choppy parallax.
		for (uint32_t v = 0; v < kQuiltViewCount; v++)
		{
			uint32_t col = v % kQuiltColumns;
			uint32_t logicalRow = v / kQuiltColumns;
			uint32_t row = (kQuiltRows - 1) - logicalRow;
			uint32_t tileX = col * kViewWidth;
			uint32_t tileY = row * kViewHeight;

			// Y-flip via negative height (matches RayMarchRenderSystem's convention)
			vk::Viewport viewport(
				static_cast<float>(tileX),
				static_cast<float>(tileY) + static_cast<float>(kViewHeight),
				static_cast<float>(kViewWidth),
				-static_cast<float>(kViewHeight),
				0.0f, 1.0f);
			vk::Rect2D scissor({static_cast<int32_t>(tileX), static_cast<int32_t>(tileY)},
			                    {kViewWidth, kViewHeight});
			_renderCommandBuffer.setViewport(0, viewport);
			_renderCommandBuffer.setScissor(0, scissor);

			// Bind per-view descriptor set. Pipeline::DescriptorOption{0} since each
			// of the 48 Descriptor objects has only its own single set (option 0).
			_graphicsPipeline->Bind(_renderCommandBuffer, vk::PipelineBindPoint::eGraphics,
				0, Pipeline::DescriptorOption{0});

			// The pipeline binds the (frame=0, view=0) descriptor by default.
			// Bind the descriptor for the requested (frame, view) here so the
			// fragment shader reads from the cell-state buffer the compute pass
			// just wrote, and uses the right per-view camera UBO.
			{
				vk::DescriptorSet ds = _viewDescriptorsPerFrame[renderFromFrameIndex][v]->GetDescriptorSet(0);
				_renderCommandBuffer.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					_graphicsPipeline->GetPipelineLayout(),
					0, 1, &ds, 0, nullptr);
			}

			_renderCommandBuffer.pushConstants<RayMarchPushConstants>(
				_graphicsPipeline->GetPipelineLayout(),
				vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
				0, pushConstants);

			_renderCommandBuffer.drawIndexed(_boundingBoxIndexCount, 1, 0, 0, 0);
		}

		_renderCommandBuffer.endRendering();

		// Transition quilt: ColorAttachmentOptimal -> General (so GL can read).
		vk::ImageMemoryBarrier quiltOut(
			vk::AccessFlagBits::eColorAttachmentWrite,
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead,
			vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eGeneral,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			*_quiltImage->GetImage(), colorRange);
		_renderCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eAllCommands,
			{}, {}, {}, quiltOut);

		_renderCommandBuffer.end();
		_initialLayoutTransitioned = true;

		// Submit: wait on compute timeline + signal binary external semaphore for GL.
		vk::SubmitInfo submit{};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &_renderCommandBuffer;

		uint64_t signalValueDummy = 0;
		std::vector<vk::Semaphore> waitSemaphores;
		std::vector<vk::PipelineStageFlags> waitStages;
		std::vector<uint64_t> waitValues;

		if (computeWaitSemaphore)
		{
			waitSemaphores.push_back(computeWaitSemaphore);
			waitStages.push_back(vk::PipelineStageFlagBits::eFragmentShader);
			waitValues.push_back(computeWaitValue);
		}
		submit.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
		submit.pWaitSemaphores = waitSemaphores.data();
		submit.pWaitDstStageMask = waitStages.data();

		std::vector<vk::Semaphore> signalSemaphores;
		std::vector<uint64_t> signalValues;
		if (signalSem)
		{
			signalSemaphores.push_back(signalSem->GetSemaphore());
			signalValues.push_back(0); // ignored for binary semaphore
		}
		submit.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size());
		submit.pSignalSemaphores = signalSemaphores.data();

		// If we have any timeline semaphore waits, attach the timeline submit info.
		// The compute timeline is timeline; the signal external is binary. Mixing
		// requires the timeline submit info regardless (binary values are ignored).
		vk::TimelineSemaphoreSubmitInfo timelineInfo{};
		if (!waitValues.empty() || !signalValues.empty())
		{
			timelineInfo.waitSemaphoreValueCount = static_cast<uint32_t>(waitValues.size());
			timelineInfo.pWaitSemaphoreValues = waitValues.data();
			timelineInfo.signalSemaphoreValueCount = static_cast<uint32_t>(signalValues.size());
			timelineInfo.pSignalSemaphoreValues = signalValues.data();
			submit.pNext = &timelineInfo;
		}

		_deviceContext.GetGraphicsQueue().submit(submit, _renderFence);
		(void)signalValueDummy;
	}

	// ---- Layout helper -----------------------------------------------------

	void QuiltRenderSystem::EnsureGeneralLayout(vk::CommandBuffer cmd)
	{
		vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
		vk::ImageMemoryBarrier barrier(
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferWrite,
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eColorAttachmentWrite,
			_initialLayoutTransitioned ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined,
			vk::ImageLayout::eGeneral,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			*_quiltImage->GetImage(), range);
		cmd.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eAllCommands,
			{}, {}, {}, barrier);
		_initialLayoutTransitioned = true;
	}

	// ---- Test pattern (M3 smoke-test path, unchanged) -----------------------

	void QuiltRenderSystem::RenderTestPattern()
	{
		const size_t totalPixels = static_cast<size_t>(kQuiltWidth) * kQuiltHeight;
		std::vector<uint8_t> pixels(totalPixels * 4, 0);
		for (uint32_t v = 0; v < kQuiltViewCount; v++)
		{
			float hue = static_cast<float>(v) / static_cast<float>(kQuiltViewCount - 1);
			uint8_t r, g, b;
			HsvToRgb255(hue, 0.85f, 0.95f, r, g, b);

			uint32_t col = v % kQuiltColumns;
			uint32_t row = v / kQuiltColumns;
			uint32_t tileX0 = col * kViewWidth;
			uint32_t tileY0 = row * kViewHeight;

			for (uint32_t ty = 0; ty < kViewHeight; ty++)
			{
				for (uint32_t tx = 0; tx < kViewWidth; tx++)
				{
					uint32_t px = tileX0 + tx;
					uint32_t py = tileY0 + ty;
					size_t idx = (static_cast<size_t>(py) * kQuiltWidth + px) * 4;
					bool topLeftStripe = (tx < 4) || (ty < 4);
					bool bottomRightStripe = (tx >= kViewWidth - 4) || (ty >= kViewHeight - 4);
					if (topLeftStripe)
					{
						pixels[idx + 0] = 255; pixels[idx + 1] = 255; pixels[idx + 2] = 255; pixels[idx + 3] = 255;
					}
					else if (bottomRightStripe)
					{
						pixels[idx + 0] = 0; pixels[idx + 1] = 0; pixels[idx + 2] = 0; pixels[idx + 3] = 255;
					}
					else
					{
						pixels[idx + 0] = r; pixels[idx + 1] = g; pixels[idx + 2] = b; pixels[idx + 3] = 255;
					}
				}
			}
		}

		Buffer staging(_deviceContext, static_cast<uint32_t>(pixels.size()),
			vk::BufferUsageFlagBits::eTransferSrc,
			VMA_MEMORY_USAGE_AUTO,
			VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
			false, 0, "QuiltTestPatternStaging");
		std::memcpy(staging.GetVmaAllocationInfo().pMappedData, pixels.data(), pixels.size());

		auto pool = _deviceContext.GetGraphicsCommandPool();
		auto cmd = _deviceContext.BeginSingleTimeCommands(pool);

		vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
		vk::ImageMemoryBarrier toDst(
			{}, vk::AccessFlagBits::eTransferWrite,
			_initialLayoutTransitioned ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			*_quiltImage->GetImage(), range);
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
			vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, toDst);

		vk::BufferImageCopy copy{};
		copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		copy.imageSubresource.layerCount = 1;
		copy.imageExtent = vk::Extent3D(kQuiltWidth, kQuiltHeight, 1);
		cmd.copyBufferToImage(staging.GetBuffer(), *_quiltImage->GetImage(),
			vk::ImageLayout::eTransferDstOptimal, 1, &copy);

		vk::ImageMemoryBarrier toGeneral(
			vk::AccessFlagBits::eTransferWrite,
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead,
			vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			*_quiltImage->GetImage(), range);
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eAllCommands, {}, {}, {}, toGeneral);

		_deviceContext.EndSingleTimeCommands(cmd, pool, _deviceContext.GetGraphicsQueue());
		_initialLayoutTransitioned = true;
		LOG_INFO("QuiltRenderSystem: rendered test pattern (HSV gradient, 48 tiles)");
	}

	void QuiltRenderSystem::ReadbackQuiltToHost(std::vector<uint8_t>& outRgba8)
	{
		const size_t totalBytes = static_cast<size_t>(kQuiltWidth) * kQuiltHeight * 4;
		outRgba8.resize(totalBytes);

		Buffer staging(_deviceContext, static_cast<uint32_t>(totalBytes),
			vk::BufferUsageFlagBits::eTransferDst,
			VMA_MEMORY_USAGE_AUTO,
			VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
			false, 0, "QuiltReadbackStaging");

		auto pool = _deviceContext.GetGraphicsCommandPool();
		auto cmd = _deviceContext.BeginSingleTimeCommands(pool);

		vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
		vk::ImageMemoryBarrier toSrc(
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferWrite,
			vk::AccessFlagBits::eTransferRead,
			vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			*_quiltImage->GetImage(), range);
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
			vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, toSrc);

		vk::BufferImageCopy copy{};
		copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		copy.imageSubresource.layerCount = 1;
		copy.imageExtent = vk::Extent3D(kQuiltWidth, kQuiltHeight, 1);
		cmd.copyImageToBuffer(*_quiltImage->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
			staging.GetBuffer(), 1, &copy);

		vk::ImageMemoryBarrier toGeneral(
			vk::AccessFlagBits::eTransferRead,
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferWrite,
			vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			*_quiltImage->GetImage(), range);
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eAllCommands, {}, {}, {}, toGeneral);

		_deviceContext.EndSingleTimeCommands(cmd, pool, _deviceContext.GetGraphicsQueue());

		std::memcpy(outRgba8.data(), staging.GetVmaAllocationInfo().pMappedData, totalBytes);
	}

} // namespace Cave
