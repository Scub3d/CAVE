#pragma once

#include <string>
#include <vulkan/vulkan.hpp>
#include "deviceContext.h"

namespace Cave
{
	// Reads back a rendered color attachment from GPU and saves it as a PNG file.
	// Any source format (including BGRA and R16G16B16A16Sfloat) is normalized to
	// R8G8B8A8 via a GPU blit before writeback. The source is transitioned from
	// sourceLayout to eTransferSrcOptimal and back around the operation.
	void CaptureFrameToPNG(DeviceContext& deviceContext, vk::Image sourceImage,
						   vk::Extent2D extent, vk::Format format,
						   const std::string& outputPath,
						   vk::ImageLayout sourceLayout = vk::ImageLayout::eTransferSrcOptimal);
}
