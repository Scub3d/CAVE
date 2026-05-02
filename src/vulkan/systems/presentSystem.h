#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>

#include "../vulkanInstance.h"
#include "../deviceContext.h"
#include "../../common/queryManager.h"

namespace Cave
{
	class GuiSystem;

	// Manages the swapchain and presents off-screen rendered frames to the window.
	// Acquires a swapchain image, blits the RenderSystem color attachment to it,
	// and queues it for presentation.
	class PresentSystem
	{
	private:
		VulkanInstance& _vulkanInstance;
		DeviceContext& _deviceContext;
		uint32_t _framesInFlight;
		GuiSystem* _guiSystem = nullptr;

		vk::SwapchainKHR _swapchain;
		std::vector<vk::Image> _swapchainImages;
		std::vector<vk::ImageView> _swapchainImageViews;
		vk::Format _swapchainImageFormat;
		vk::Extent2D _swapchainExtent;

		vk::CommandPool _graphicsCommandPool;
		std::vector<vk::CommandBuffer> _blitCommandBuffers;

		std::vector<vk::Semaphore> _imageAvailableSemaphores;
		std::vector<vk::Semaphore> _blitFinishedSemaphores;
		std::vector<vk::Fence> _blitInFlightFences;

		bool _swapchainNeedsRecreation = false;

	private:
		vk::SurfaceFormatKHR ChooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
		vk::PresentModeKHR ChoosePresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
		vk::Extent2D ChooseExtent(const vk::SurfaceCapabilitiesKHR& surfaceCapabilities);

		void CreateSwapchain();
		void CleanupSwapchain();
		void RecreateSwapchain();
		void CreateSyncObjects();

	public:
		PresentSystem(VulkanInstance& vulkanInstance, DeviceContext& deviceContext);
		~PresentSystem();

		PresentSystem(const PresentSystem&) = delete;
		PresentSystem& operator=(const PresentSystem&) = delete;

		void SetGuiSystem(GuiSystem* guiSystem) { _guiSystem = guiSystem; }

		vk::Format GetSwapchainImageFormat() const { return _swapchainImageFormat; }

		// Call BEFORE RenderSystem::RenderFrame() to ensure the previous blit
		// on this frame slot has finished reading the color attachment.
		void WaitForPreviousBlit(uint32_t frameIndex);

		// Presents just the GUI overlay with a cleared background (no simulation blit).
		void PresentGuiOnly(uint32_t frameIndex);

		// Like PresentGuiOnly but blits a source image first (no timeline semaphore needed).
		// Used when paused to show the frozen simulation with GUI overlay.
		void PresentPausedFrame(uint32_t frameIndex, vk::Image sourceImage, vk::Extent2D sourceExtent);

		// Acquires a swapchain image, blits sourceImage to it, and presents.
		void PresentFrame(uint32_t frameIndex,
						  vk::Semaphore renderFinishedSemaphore,
						  uint64_t renderFinishedSemaphoreWaitValue,
						  vk::Image sourceImage,
						  vk::Extent2D sourceExtent,
						  std::shared_ptr<QueryManager> queryManager = nullptr, vk::QueryPool queryPool = {});
	};
}
