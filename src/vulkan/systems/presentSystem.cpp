#include "presentSystem.h"
#include "guiSystem.h"
#include "../../common/logger.h"

#include <GLFW/glfw3.h>
#include <algorithm>

namespace Cave
{
	PresentSystem::PresentSystem(VulkanInstance& vulkanInstance, DeviceContext& deviceContext)
		: _vulkanInstance{vulkanInstance}, _deviceContext{deviceContext}, _framesInFlight{deviceContext.GetFramesInFlight()}
	{
		_graphicsCommandPool = _deviceContext.GetGraphicsCommandPool();

		CreateSwapchain();
		CreateSyncObjects();

		_blitCommandBuffers.resize(_framesInFlight);
		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			_blitCommandBuffers[frameIndex] = _deviceContext.CreateCommandBuffer(_graphicsCommandPool);
		}
	}

	PresentSystem::~PresentSystem()
	{
		vk::Device device = _deviceContext.GetDevice();

		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			device.destroySemaphore(_imageAvailableSemaphores[frameIndex]);
			device.destroySemaphore(_blitFinishedSemaphores[frameIndex]);
			device.destroyFence(_blitInFlightFences[frameIndex]);
		}

		CleanupSwapchain();
	}

	vk::SurfaceFormatKHR PresentSystem::ChooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
	{
		for (const auto& availableFormat : availableFormats)
		{
			if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
				availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
			{
				return availableFormat;
			}
		}

		for (const auto& availableFormat : availableFormats)
		{
			if (availableFormat.format == vk::Format::eB8G8R8A8Unorm)
			{
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	vk::PresentModeKHR PresentSystem::ChoosePresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
	{
		for (const auto& availablePresentMode : availablePresentModes)
		{
			if (availablePresentMode == vk::PresentModeKHR::eMailbox)
			{
				return availablePresentMode;
			}
		}

		return vk::PresentModeKHR::eFifo;
	}

	vk::Extent2D PresentSystem::ChooseExtent(const vk::SurfaceCapabilitiesKHR& surfaceCapabilities)
	{
		if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return surfaceCapabilities.currentExtent;
		}

		int framebufferWidth = 0;
		int framebufferHeight = 0;
		glfwGetFramebufferSize(_vulkanInstance.GetVulkanGLFWWindow(), &framebufferWidth, &framebufferHeight);

		vk::Extent2D actualExtent = {
			static_cast<uint32_t>(framebufferWidth),
			static_cast<uint32_t>(framebufferHeight)
		};

		actualExtent.width = std::clamp(actualExtent.width,
			surfaceCapabilities.minImageExtent.width,
			surfaceCapabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height,
			surfaceCapabilities.minImageExtent.height,
			surfaceCapabilities.maxImageExtent.height);

		return actualExtent;
	}

	void PresentSystem::CreateSwapchain()
	{
		vk::PhysicalDevice physicalDevice = _deviceContext.GetPhysicalDevice();
		vk::SurfaceKHR surface = _vulkanInstance.GetSurface();

		vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
		std::vector<vk::SurfaceFormatKHR> surfaceFormats = physicalDevice.getSurfaceFormatsKHR(surface);
		std::vector<vk::PresentModeKHR> presentModes = physicalDevice.getSurfacePresentModesKHR(surface);

		vk::SurfaceFormatKHR surfaceFormat = ChooseSurfaceFormat(surfaceFormats);
		vk::PresentModeKHR presentMode = ChoosePresentMode(presentModes);
		vk::Extent2D extent = ChooseExtent(surfaceCapabilities);

		uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
		if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount)
		{
			imageCount = surfaceCapabilities.maxImageCount;
		}

		vk::SwapchainCreateInfoKHR swapchainCreateInfo = vk::SwapchainCreateInfoKHR(
			{},                                        // flags
			surface,                                   // surface
			imageCount,                                // minImageCount
			surfaceFormat.format,                      // imageFormat
			surfaceFormat.colorSpace,                  // imageColorSpace
			extent,                                    // imageExtent
			1,                                         // imageArrayLayers
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment, // imageUsage
			vk::SharingMode::eExclusive,               // imageSharingMode
			0,                                         // queueFamilyIndexCount
			nullptr,                                   // pQueueFamilyIndices
			surfaceCapabilities.currentTransform,       // preTransform
			vk::CompositeAlphaFlagBitsKHR::eOpaque,    // compositeAlpha
			presentMode,                               // presentMode
			vk::True,                                  // clipped
			nullptr                                    // oldSwapchain
		);

		// If present and graphics queues are different families, use concurrent sharing
		QueueFamilies queueFamilies = _deviceContext.GetQueueFamilies();
		uint32_t graphicsFamilyIndex = queueFamilies.GraphicsFamily.Index.value();
		uint32_t presentFamilyIndex = queueFamilies.PresentFamily.Index.value();

		std::array<uint32_t, 2> queueFamilyIndices = {graphicsFamilyIndex, presentFamilyIndex};
		if (graphicsFamilyIndex != presentFamilyIndex)
		{
			swapchainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
			swapchainCreateInfo.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size());
			swapchainCreateInfo.pQueueFamilyIndices = queueFamilyIndices.data();
		}

		vk::Device device = _deviceContext.GetDevice();
		_swapchain = device.createSwapchainKHR(swapchainCreateInfo);
		_swapchainImageFormat = surfaceFormat.format;
		_swapchainExtent = extent;

		_swapchainImages = device.getSwapchainImagesKHR(_swapchain);

		// Create image views for swapchain images
		_swapchainImageViews.resize(_swapchainImages.size());
		for (size_t imageIndex = 0; imageIndex < _swapchainImages.size(); imageIndex++)
		{
			vk::ImageSubresourceRange subresourceRange = vk::ImageSubresourceRange(
				vk::ImageAspectFlagBits::eColor, // aspectMask
				0,                               // baseMipLevel
				1,                               // levelCount
				0,                               // baseArrayLayer
				1                                // layerCount
			);

			vk::ImageViewCreateInfo imageViewCreateInfo = vk::ImageViewCreateInfo(
				{},                              // flags
				_swapchainImages[imageIndex],     // image
				vk::ImageViewType::e2D,          // viewType
				_swapchainImageFormat,            // format
				{},                              // components (identity swizzle)
				subresourceRange                 // subresourceRange
			);

			_swapchainImageViews[imageIndex] = device.createImageView(imageViewCreateInfo);
		}

		LOG_INFO("Created swapchain: {}x{}, {} images, format {}",
			_swapchainExtent.width, _swapchainExtent.height,
			_swapchainImages.size(), vk::to_string(_swapchainImageFormat));
	}

	void PresentSystem::CleanupSwapchain()
	{
		vk::Device device = _deviceContext.GetDevice();

		for (auto imageView : _swapchainImageViews)
		{
			device.destroyImageView(imageView);
		}
		_swapchainImageViews.clear();

		device.destroySwapchainKHR(_swapchain);
	}

	void PresentSystem::RecreateSwapchain()
	{
		// Wait for minimized window to be restored
		int framebufferWidth = 0;
		int framebufferHeight = 0;
		glfwGetFramebufferSize(_vulkanInstance.GetVulkanGLFWWindow(), &framebufferWidth, &framebufferHeight);
		while (framebufferWidth == 0 || framebufferHeight == 0)
		{
			glfwGetFramebufferSize(_vulkanInstance.GetVulkanGLFWWindow(), &framebufferWidth, &framebufferHeight);
			glfwWaitEvents();
		}

		_deviceContext.GetDevice().waitIdle();

		CleanupSwapchain();
		CreateSwapchain();

		_swapchainNeedsRecreation = false;
	}

	void PresentSystem::CreateSyncObjects()
	{
		_imageAvailableSemaphores.resize(_framesInFlight);
		_blitFinishedSemaphores.resize(_framesInFlight);
		_blitInFlightFences.resize(_framesInFlight);

		for (uint32_t frameIndex = 0; frameIndex < _framesInFlight; frameIndex++)
		{
			_imageAvailableSemaphores[frameIndex] = _deviceContext.CreateSemaphore();
			_blitFinishedSemaphores[frameIndex] = _deviceContext.CreateSemaphore();
			_blitInFlightFences[frameIndex] = _deviceContext.CreateFence();
		}
	}

	void PresentSystem::WaitForPreviousBlit(uint32_t frameIndex)
	{
		vk::Device device = _deviceContext.GetDevice();
		auto waitResult = device.waitForFences(_blitInFlightFences[frameIndex], vk::True, UINT64_MAX);
		if (waitResult != vk::Result::eSuccess)
		{
			LOG_ERROR("PresentSystem: timed out waiting for blit fence");
			return;
		}
		device.resetFences(_blitInFlightFences[frameIndex]);
	}

	void PresentSystem::PresentGuiOnly(uint32_t frameIndex)
	{
		if (_swapchainNeedsRecreation)
		{
			RecreateSwapchain();
		}

		vk::Device device = _deviceContext.GetDevice();

		auto waitResult = device.waitForFences(_blitInFlightFences[frameIndex], vk::True, UINT64_MAX);
		if (waitResult != vk::Result::eSuccess)
		{
			LOG_ERROR("PresentSystem: timed out waiting for blit fence");
			return;
		}
		device.resetFences(_blitInFlightFences[frameIndex]);

		uint32_t swapchainImageIndex = 0;
		vk::Result acquireResult = device.acquireNextImageKHR(
			_swapchain, UINT64_MAX,
			_imageAvailableSemaphores[frameIndex],
			nullptr, &swapchainImageIndex);

		if (acquireResult == vk::Result::eErrorOutOfDateKHR ||
			(acquireResult != vk::Result::eSuccess && acquireResult != vk::Result::eSuboptimalKHR))
		{
			// Signal the fence so the next frame doesn't deadlock
			vk::SubmitInfo emptySubmitInfo{};
			_deviceContext.GetGraphicsQueue().submit(emptySubmitInfo, _blitInFlightFences[frameIndex]);
			if (acquireResult == vk::Result::eErrorOutOfDateKHR)
				RecreateSwapchain();
			return;
		}

		vk::CommandBuffer commandBuffer = _blitCommandBuffers[frameIndex];
		commandBuffer.reset();

		vk::CommandBufferBeginInfo commandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		commandBuffer.begin(commandBufferBeginInfo);

		vk::ImageSubresourceRange colorSubresourceRange = vk::ImageSubresourceRange(
			vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

		// Transition swapchain image: undefined -> color attachment
		vk::ImageMemoryBarrier swapchainBarrierToColorAttachment = vk::ImageMemoryBarrier(
			{},                                              // srcAccessMask
			vk::AccessFlagBits::eColorAttachmentWrite,       // dstAccessMask
			vk::ImageLayout::eUndefined,                     // oldLayout
			vk::ImageLayout::eColorAttachmentOptimal,        // newLayout
			VK_QUEUE_FAMILY_IGNORED,                         // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                         // dstQueueFamilyIndex
			_swapchainImages[swapchainImageIndex],           // image
			colorSubresourceRange                            // subresourceRange
		);
		commandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
			{}, {}, {}, swapchainBarrierToColorAttachment);

		if (_guiSystem)
		{
			vk::ClearValue clearValue = vk::ClearValue(
				vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}});

			vk::RenderingAttachmentInfo guiColorAttachmentInfo = vk::RenderingAttachmentInfo(
				_swapchainImageViews[swapchainImageIndex],  // imageView
				vk::ImageLayout::eColorAttachmentOptimal,   // imageLayout
				vk::ResolveModeFlagBits::eNone,             // resolveMode
				{}, {},                                     // resolveImageView, resolveImageLayout
				vk::AttachmentLoadOp::eClear,               // loadOp
				vk::AttachmentStoreOp::eStore,              // storeOp
				clearValue                                  // clearValue
			);

			vk::RenderingInfo guiRenderingInfo = vk::RenderingInfo(
				{},                                     // flags
				vk::Rect2D{{0, 0}, _swapchainExtent},   // renderArea
				1,                                      // layerCount
				0,                                      // viewMask
				guiColorAttachmentInfo,                 // colorAttachmentCount + pColorAttachments
				nullptr,                                // pDepthAttachment
				nullptr                                 // pStencilAttachment
			);

			commandBuffer.beginRendering(guiRenderingInfo);
			_guiSystem->RecordDrawCommands(commandBuffer);
			commandBuffer.endRendering();
		}

		// Transition swapchain image: color attachment -> present src
		vk::ImageMemoryBarrier swapchainBarrierToPresent = vk::ImageMemoryBarrier(
			vk::AccessFlagBits::eColorAttachmentWrite,   // srcAccessMask
			{},                                          // dstAccessMask
			vk::ImageLayout::eColorAttachmentOptimal,    // oldLayout
			vk::ImageLayout::ePresentSrcKHR,             // newLayout
			VK_QUEUE_FAMILY_IGNORED,                     // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                     // dstQueueFamilyIndex
			_swapchainImages[swapchainImageIndex],       // image
			colorSubresourceRange                        // subresourceRange
		);
		commandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe,
			{}, {}, {}, swapchainBarrierToPresent);

		commandBuffer.end();

		// Submit — only wait on imageAvailable (no render semaphore needed)
		vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		vk::SubmitInfo submitInfo = vk::SubmitInfo(
			1, &_imageAvailableSemaphores[frameIndex], &waitStage,
			1, &commandBuffer,
			1, &_blitFinishedSemaphores[frameIndex]
		);
		_deviceContext.GetGraphicsQueue().submit(submitInfo, _blitInFlightFences[frameIndex]);

		// Present
		vk::PresentInfoKHR presentInfo = vk::PresentInfoKHR(
			1, &_blitFinishedSemaphores[frameIndex],
			1, &_swapchain, &swapchainImageIndex, nullptr
		);

		vk::Result presentResult;
		try { presentResult = _deviceContext.GetPresentQueue().presentKHR(presentInfo); }
		catch (vk::OutOfDateKHRError&) { _swapchainNeedsRecreation = true; return; }

		if (presentResult == vk::Result::eSuboptimalKHR || _vulkanInstance.WasWindowResized())
		{
			_vulkanInstance.ResetWindowResizedFlag();
			_swapchainNeedsRecreation = true;
		}
	}

	void PresentSystem::PresentPausedFrame(uint32_t frameIndex, vk::Image sourceImage, vk::Extent2D sourceExtent)
	{
		if (_swapchainNeedsRecreation)
			RecreateSwapchain();

		vk::Device device = _deviceContext.GetDevice();

		auto waitResult = device.waitForFences(_blitInFlightFences[frameIndex], vk::True, UINT64_MAX);
		if (waitResult != vk::Result::eSuccess) return;
		device.resetFences(_blitInFlightFences[frameIndex]);

		uint32_t swapchainImageIndex = 0;
		vk::Result acquireResult = device.acquireNextImageKHR(
			_swapchain, UINT64_MAX, _imageAvailableSemaphores[frameIndex], nullptr, &swapchainImageIndex);

		if (acquireResult == vk::Result::eErrorOutOfDateKHR ||
			(acquireResult != vk::Result::eSuccess && acquireResult != vk::Result::eSuboptimalKHR))
		{
			vk::SubmitInfo emptySubmitInfo{};
			_deviceContext.GetGraphicsQueue().submit(emptySubmitInfo, _blitInFlightFences[frameIndex]);
			if (acquireResult == vk::Result::eErrorOutOfDateKHR) RecreateSwapchain();
			return;
		}

		vk::CommandBuffer commandBuffer = _blitCommandBuffers[frameIndex];
		commandBuffer.reset();
		vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		commandBuffer.begin(beginInfo);

		vk::ImageSubresourceRange colorRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

		// Transition swapchain image: undefined → transfer dst
		vk::ImageMemoryBarrier toTransferDst({}, vk::AccessFlagBits::eTransferWrite,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			_swapchainImages[swapchainImageIndex], colorRange);
		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
			vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, toTransferDst);

		// Blit the source (render output) to swapchain
		vk::ImageSubresourceLayers subresource(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
		std::array<vk::Offset3D, 2> srcOffsets = {
			vk::Offset3D{0, 0, 0},
			vk::Offset3D{(int32_t)sourceExtent.width, (int32_t)sourceExtent.height, 1}};
		std::array<vk::Offset3D, 2> dstOffsets = {
			vk::Offset3D{0, 0, 0},
			vk::Offset3D{(int32_t)_swapchainExtent.width, (int32_t)_swapchainExtent.height, 1}};
		vk::ImageBlit blitRegion(subresource, srcOffsets, subresource, dstOffsets);
		commandBuffer.blitImage(sourceImage, vk::ImageLayout::eTransferSrcOptimal,
			_swapchainImages[swapchainImageIndex], vk::ImageLayout::eTransferDstOptimal,
			blitRegion, vk::Filter::eLinear);

		// Transition swapchain → color attachment for GUI overlay
		if (_guiSystem)
		{
			vk::ImageMemoryBarrier toColorAttach(vk::AccessFlagBits::eTransferWrite,
				vk::AccessFlagBits::eColorAttachmentWrite,
				vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eColorAttachmentOptimal,
				VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
				_swapchainImages[swapchainImageIndex], colorRange);
			commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, {}, {}, toColorAttach);

			vk::ClearValue clearValue(vk::ClearColorValue{std::array<float,4>{0,0,0,1}});
			vk::RenderingAttachmentInfo guiAttach(
				_swapchainImageViews[swapchainImageIndex], vk::ImageLayout::eColorAttachmentOptimal,
				vk::ResolveModeFlagBits::eNone, {}, {},
				vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore, clearValue);
			vk::RenderingInfo guiRendering({}, vk::Rect2D{{0,0}, _swapchainExtent}, 1, 0, guiAttach, nullptr, nullptr);
			commandBuffer.beginRendering(guiRendering);
			_guiSystem->RecordDrawCommands(commandBuffer);
			commandBuffer.endRendering();

			vk::ImageMemoryBarrier toPresent(vk::AccessFlagBits::eColorAttachmentWrite, {},
				vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
				VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
				_swapchainImages[swapchainImageIndex], colorRange);
			commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
				vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, toPresent);
		}
		else
		{
			vk::ImageMemoryBarrier toPresent(vk::AccessFlagBits::eTransferWrite, {},
				vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
				VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
				_swapchainImages[swapchainImageIndex], colorRange);
			commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, toPresent);
		}

		commandBuffer.end();

		// Submit — only wait on imageAvailable (NO timeline render semaphore)
		vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eTransfer;
		vk::SubmitInfo submitInfo(1, &_imageAvailableSemaphores[frameIndex], &waitStage,
			1, &commandBuffer, 1, &_blitFinishedSemaphores[frameIndex]);
		_deviceContext.GetGraphicsQueue().submit(submitInfo, _blitInFlightFences[frameIndex]);

		vk::PresentInfoKHR presentInfo(1, &_blitFinishedSemaphores[frameIndex],
			1, &_swapchain, &swapchainImageIndex, nullptr);
		try { _deviceContext.GetPresentQueue().presentKHR(presentInfo); }
		catch (vk::OutOfDateKHRError&) { _swapchainNeedsRecreation = true; return; }
	}

	void PresentSystem::PresentFrame(uint32_t frameIndex,
									 vk::Semaphore renderFinishedSemaphore,
									 uint64_t renderFinishedSemaphoreWaitValue,
									 vk::Image sourceImage,
									 vk::Extent2D sourceExtent,
									 std::shared_ptr<QueryManager> queryManager, vk::QueryPool queryPool)
	{
		if (_swapchainNeedsRecreation)
		{
			RecreateSwapchain();
		}

		vk::Device device = _deviceContext.GetDevice();

		// Acquire next swapchain image
		uint32_t swapchainImageIndex = 0;
		vk::Result acquireResult = device.acquireNextImageKHR(
			_swapchain, UINT64_MAX,
			_imageAvailableSemaphores[frameIndex],
			nullptr, &swapchainImageIndex);

		if (acquireResult == vk::Result::eErrorOutOfDateKHR ||
			(acquireResult != vk::Result::eSuccess && acquireResult != vk::Result::eSuboptimalKHR))
		{
			if (acquireResult != vk::Result::eErrorOutOfDateKHR)
			{
				LOG_ERROR("PresentSystem: failed to acquire swapchain image");
			}

			// The blit fence was already reset by WaitForPreviousBlit and we are returning
			// without submitting a blit. Submit an empty batch that:
			//   1. Waits on renderFinishedSemaphore (timeline) to ensure render is done
			//   2. Signals blitInFlightFence so the next WaitForPreviousBlit does not deadlock
			// acquireNextImageKHR did NOT signal imageAvailableSemaphore on failure,
			// so we must not wait on it here.
			vk::TimelineSemaphoreSubmitInfo drainTimelineSemaphoreSubmitInfo = vk::TimelineSemaphoreSubmitInfo(
				1,                                  // waitSemaphoreValueCount
				&renderFinishedSemaphoreWaitValue,  // pWaitSemaphoreValues
				0,                                  // signalSemaphoreValueCount
				nullptr                             // pSignalSemaphoreValues
			);

			vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eTopOfPipe;
			vk::SubmitInfo drainSubmitInfo = vk::SubmitInfo(
				1,                       // waitSemaphoreCount
				&renderFinishedSemaphore, // pWaitSemaphores
				&waitStage,              // pWaitDstStageMask
				0,                       // commandBufferCount
				nullptr,                 // pCommandBuffers
				0,                       // signalSemaphoreCount
				nullptr                  // pSignalSemaphores
			);
			drainSubmitInfo.pNext = &drainTimelineSemaphoreSubmitInfo;
			_deviceContext.GetGraphicsQueue().submit(drainSubmitInfo, _blitInFlightFences[frameIndex]);

			if (acquireResult == vk::Result::eErrorOutOfDateKHR)
			{
				RecreateSwapchain();
			}
			return;
		}

		// Record blit command buffer
		vk::CommandBuffer blitCommandBuffer = _blitCommandBuffers[frameIndex];
		blitCommandBuffer.reset();

		vk::CommandBufferBeginInfo commandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		blitCommandBuffer.begin(commandBufferBeginInfo);

		vk::ImageSubresourceRange colorSubresourceRange = vk::ImageSubresourceRange(
			vk::ImageAspectFlagBits::eColor, // aspectMask
			0,                               // baseMipLevel
			1,                               // levelCount
			0,                               // baseArrayLayer
			1                                // layerCount
		);

		// Transition swapchain image: undefined -> transfer dst
		vk::ImageMemoryBarrier swapchainBarrierToTransferDst = vk::ImageMemoryBarrier(
			{},                                          // srcAccessMask
			vk::AccessFlagBits::eTransferWrite,          // dstAccessMask
			vk::ImageLayout::eUndefined,                 // oldLayout
			vk::ImageLayout::eTransferDstOptimal,        // newLayout
			VK_QUEUE_FAMILY_IGNORED,                     // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                     // dstQueueFamilyIndex
			_swapchainImages[swapchainImageIndex],       // image
			colorSubresourceRange                        // subresourceRange
		);
		blitCommandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
			{}, {}, {}, swapchainBarrierToTransferDst);

		// Blit off-screen color attachment to swapchain image
		vk::ImageSubresourceLayers subresourceLayers = vk::ImageSubresourceLayers(
			vk::ImageAspectFlagBits::eColor, // aspectMask
			0,                               // mipLevel
			0,                               // baseArrayLayer
			1                                // layerCount
		);

		std::array<vk::Offset3D, 2> sourceOffsets = {
			vk::Offset3D{0, 0, 0},
			vk::Offset3D{static_cast<int32_t>(sourceExtent.width), static_cast<int32_t>(sourceExtent.height), 1}
		};
		std::array<vk::Offset3D, 2> destinationOffsets = {
			vk::Offset3D{0, 0, 0},
			vk::Offset3D{static_cast<int32_t>(_swapchainExtent.width), static_cast<int32_t>(_swapchainExtent.height), 1}
		};

		vk::ImageBlit blitRegion = vk::ImageBlit(
			subresourceLayers, // srcSubresource
			sourceOffsets,     // srcOffsets
			subresourceLayers, // dstSubresource
			destinationOffsets // dstOffsets
		);

		if (queryManager) queryManager->WriteTimestamp(blitCommandBuffer, queryPool, vk::PipelineStageFlagBits::eTopOfPipe, "present_start");
		blitCommandBuffer.blitImage(
			sourceImage, vk::ImageLayout::eTransferSrcOptimal,
			_swapchainImages[swapchainImageIndex], vk::ImageLayout::eTransferDstOptimal,
			blitRegion, vk::Filter::eLinear);

		if (_guiSystem)
		{
			// Transition swapchain image: transfer dst -> color attachment optimal (for ImGui rendering)
			vk::ImageMemoryBarrier swapchainBarrierToColorAttachment = vk::ImageMemoryBarrier(
				vk::AccessFlagBits::eTransferWrite,              // srcAccessMask
				vk::AccessFlagBits::eColorAttachmentWrite,       // dstAccessMask
				vk::ImageLayout::eTransferDstOptimal,            // oldLayout
				vk::ImageLayout::eColorAttachmentOptimal,        // newLayout
				VK_QUEUE_FAMILY_IGNORED,                         // srcQueueFamilyIndex
				VK_QUEUE_FAMILY_IGNORED,                         // dstQueueFamilyIndex
				_swapchainImages[swapchainImageIndex],           // image
				colorSubresourceRange                            // subresourceRange
			);
			blitCommandBuffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eColorAttachmentOutput,
				{}, {}, {}, swapchainBarrierToColorAttachment);

			// Render ImGui overlay using dynamic rendering
			vk::RenderingAttachmentInfo guiColorAttachmentInfo = vk::RenderingAttachmentInfo(
				_swapchainImageViews[swapchainImageIndex],  // imageView
				vk::ImageLayout::eColorAttachmentOptimal,   // imageLayout
				vk::ResolveModeFlagBits::eNone,             // resolveMode
				{},                                         // resolveImageView
				{},                                         // resolveImageLayout
				vk::AttachmentLoadOp::eLoad,                // loadOp
				vk::AttachmentStoreOp::eStore,              // storeOp
				{}                                          // clearValue
			);

			vk::RenderingInfo guiRenderingInfo = vk::RenderingInfo(
				{},                                     // flags
				vk::Rect2D{{0, 0}, _swapchainExtent},   // renderArea
				1,                                      // layerCount
				0,                                      // viewMask
				guiColorAttachmentInfo,                 // colorAttachmentCount + pColorAttachments
				nullptr,                                // pDepthAttachment
				nullptr                                 // pStencilAttachment
			);

			blitCommandBuffer.beginRendering(guiRenderingInfo);
			_guiSystem->RecordDrawCommands(blitCommandBuffer);
			blitCommandBuffer.endRendering();
			if (queryManager) queryManager->WriteTimestamp(blitCommandBuffer, queryPool, vk::PipelineStageFlagBits::eBottomOfPipe, "present_end");

			// Transition swapchain image: color attachment optimal -> present src
			vk::ImageMemoryBarrier swapchainBarrierToPresent = vk::ImageMemoryBarrier(
				vk::AccessFlagBits::eColorAttachmentWrite,   // srcAccessMask
				{},                                          // dstAccessMask
				vk::ImageLayout::eColorAttachmentOptimal,    // oldLayout
				vk::ImageLayout::ePresentSrcKHR,             // newLayout
				VK_QUEUE_FAMILY_IGNORED,                     // srcQueueFamilyIndex
				VK_QUEUE_FAMILY_IGNORED,                     // dstQueueFamilyIndex
				_swapchainImages[swapchainImageIndex],       // image
				colorSubresourceRange                        // subresourceRange
			);
			blitCommandBuffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe,
				{}, {}, {}, swapchainBarrierToPresent);
		}
		else
		{
			// No GUI — transition directly to present
			if (queryManager) queryManager->WriteTimestamp(blitCommandBuffer, queryPool, vk::PipelineStageFlagBits::eBottomOfPipe, "present_end");
			vk::ImageMemoryBarrier swapchainBarrierToPresent = vk::ImageMemoryBarrier(
				vk::AccessFlagBits::eTransferWrite,          // srcAccessMask
				{},                                          // dstAccessMask
				vk::ImageLayout::eTransferDstOptimal,        // oldLayout
				vk::ImageLayout::ePresentSrcKHR,             // newLayout
				VK_QUEUE_FAMILY_IGNORED,                     // srcQueueFamilyIndex
				VK_QUEUE_FAMILY_IGNORED,                     // dstQueueFamilyIndex
				_swapchainImages[swapchainImageIndex],       // image
				colorSubresourceRange                        // subresourceRange
			);
			blitCommandBuffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe,
				{}, {}, {}, swapchainBarrierToPresent);
		}

		blitCommandBuffer.end();

		// Submit blit command buffer.
		// Wait semaphores: renderFinished (timeline) + imageAvailable (binary).
		// Signal semaphore: blitFinished (binary).
		// TimelineSemaphoreSubmitInfo arrays must match the semaphore counts.
		// Binary semaphore values are ignored by the driver but must be present.
		std::array<vk::Semaphore, 2> waitSemaphores = {renderFinishedSemaphore, _imageAvailableSemaphores[frameIndex]};
		std::array<vk::PipelineStageFlags, 2> waitStages = {
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eTransfer
		};
		std::array<uint64_t, 2> waitSemaphoreValues = {
			renderFinishedSemaphoreWaitValue, // timeline value for renderFinished
			0                                 // ignored for binary imageAvailable
		};
		uint64_t signalSemaphoreValue = 0;    // ignored for binary blitFinished

		vk::TimelineSemaphoreSubmitInfo blitTimelineSemaphoreSubmitInfo = vk::TimelineSemaphoreSubmitInfo(
			static_cast<uint32_t>(waitSemaphoreValues.size()), // waitSemaphoreValueCount
			waitSemaphoreValues.data(),                        // pWaitSemaphoreValues
			1,                                                 // signalSemaphoreValueCount
			&signalSemaphoreValue                              // pSignalSemaphoreValues
		);

		vk::SubmitInfo blitSubmitInfo = vk::SubmitInfo(
			static_cast<uint32_t>(waitSemaphores.size()), // waitSemaphoreCount
			waitSemaphores.data(),                        // pWaitSemaphores
			waitStages.data(),                            // pWaitDstStageMask
			1,                                            // commandBufferCount
			&blitCommandBuffer,                           // pCommandBuffers
			1,                                            // signalSemaphoreCount
			&_blitFinishedSemaphores[frameIndex]          // pSignalSemaphores
		);
		blitSubmitInfo.pNext = &blitTimelineSemaphoreSubmitInfo;

		_deviceContext.GetGraphicsQueue().submit(blitSubmitInfo, _blitInFlightFences[frameIndex]);

		// Present
		vk::PresentInfoKHR presentInfo = vk::PresentInfoKHR(
			1,                                    // waitSemaphoreCount
			&_blitFinishedSemaphores[frameIndex], // pWaitSemaphores
			1,                                    // swapchainCount
			&_swapchain,                          // pSwapchains
			&swapchainImageIndex,                 // pImageIndices
			nullptr                               // pResults
		);

		vk::Result presentResult;
		try
		{
			presentResult = _deviceContext.GetPresentQueue().presentKHR(presentInfo);
		}
		catch (vk::OutOfDateKHRError&)
		{
			_swapchainNeedsRecreation = true;
			return;
		}

		if (presentResult == vk::Result::eSuboptimalKHR || _vulkanInstance.WasWindowResized())
		{
			_vulkanInstance.ResetWindowResizedFlag();
			_swapchainNeedsRecreation = true;
		}
	}
}
