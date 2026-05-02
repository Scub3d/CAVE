#pragma once

#include <vulkan/vulkan.hpp>

#include "../deviceContext.h"
#include "../../common/structs.h"

struct ImGuiContext;

namespace Cave
{
	class VulkanInstance;

	// Renders a metadata overlay (grid size, rules, shape, etc.) into a video-encoder
	// color attachment. Owns a SECOND ImGui context with its own Vulkan backend init,
	// targeting the encoder's color format — needed because GuiSystem's main context is
	// initialized for the swapchain format and Vulkan dynamic rendering requires the
	// pipeline's color format to match the runtime attachment exactly.
	class VideoOverlay
	{
	public:
		// `depthFormat` must match the depth attachment of the renderpass that
		// RecordDrawCommands will be invoked within (or eUndefined if there is none).
		// Vulkan dynamic rendering validates pipeline format vs runtime attachment exactly.
		VideoOverlay(VulkanInstance& vulkanInstance, DeviceContext& deviceContext,
			vk::Format colorFormat, vk::Format depthFormat, vk::Extent2D extent, uint32_t framesInFlight);
		~VideoOverlay();

		VideoOverlay(const VideoOverlay&) = delete;
		VideoOverlay& operator=(const VideoOverlay&) = delete;

		void SetInfo(const OverlayInfo& info) { _info = info; }

		// Builds the overlay window and records ImGui draw commands into commandBuffer.
		// Caller must already have an active dynamic-rendering pass with a color attachment
		// matching the format passed to the constructor.
		void RecordDrawCommands(vk::CommandBuffer commandBuffer);

	private:
		void BuildOverlayWindow();

		DeviceContext& _deviceContext;
		vk::Extent2D _extent;
		ImGuiContext* _overlayContext = nullptr;
		vk::DescriptorPool _descriptorPool{};
		OverlayInfo _info{};
	};
}
