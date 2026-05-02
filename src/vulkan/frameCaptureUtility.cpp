#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "frameCaptureUtility.h"
#include "buffer.h"
#include "image.h"
#include "../common/logger.h"

#include <array>

namespace Cave
{
	void CaptureFrameToPNG(DeviceContext& deviceContext, vk::Image sourceImage,
						   vk::Extent2D extent, vk::Format format,
						   const std::string& outputPath,
						   vk::ImageLayout sourceLayout)
	{
		vk::Device device = deviceContext.GetDevice();

		device.waitIdle();

		uint32_t pixelCount = extent.width * extent.height;
		vk::DeviceSize bufferSize = pixelCount * 4; // RGBA 8-bit output

		// Blit target in R8G8B8A8_Unorm — always normalize through a format-converting
		// blit so the source can be BGRA, R16G16B16A16Sfloat, or anything else.
		Image blitTargetImage(
			deviceContext,
			vk::Format::eR8G8B8A8Unorm,                                                                         // format
			1,                                                                                                  // arrayCount
			1,                                                                                                  // mipLevels
			vk::Extent3D(extent.width, extent.height, 1),                                                       // extent
			vk::ImageTiling::eOptimal,                                                                          // imageTiling
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc
				| vk::ImageUsageFlagBits::eSampled,                                                             // imageUsageFlags (Sampled satisfies image view creation)
			vk::MemoryPropertyFlagBits::eDeviceLocal,                                                           // memoryPropertyFlags
			vk::ImageCreateFlags(),                                                                             // imageCreateFlags
			vk::ImageAspectFlagBits::eColor,                                                                    // imageAspectFlags
			vk::ImageViewType::e2D,                                                                             // imageViewType
			vk::SharingMode::eExclusive,                                                                        // sharingMode
			std::vector<uint32_t>{}                                                                             // queueFamilyIndices
		);

		auto stagingBuffer = Buffer::Staging(deviceContext, static_cast<uint32_t>(bufferSize));

		vk::CommandBuffer commandBuffer = deviceContext.BeginSingleTimeCommands(
			deviceContext.GetGraphicsCommandPool());

		vk::ImageSubresourceRange subresourceRange = vk::ImageSubresourceRange(
			vk::ImageAspectFlagBits::eColor,	// aspectMask
			0,									// baseMipLevel
			1,									// levelCount
			0,									// baseArrayLayer
			1									// layerCount
		);

		// Transition source to eTransferSrcOptimal if not already there
		if (sourceLayout != vk::ImageLayout::eTransferSrcOptimal)
		{
			vk::ImageMemoryBarrier sourceToTransferSrc = vk::ImageMemoryBarrier(
				vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,     // srcAccessMask
				vk::AccessFlagBits::eTransferRead,                                      // dstAccessMask
				sourceLayout,                                                           // oldLayout
				vk::ImageLayout::eTransferSrcOptimal,                                   // newLayout
				VK_QUEUE_FAMILY_IGNORED,                                                // srcQueueFamilyIndex
				VK_QUEUE_FAMILY_IGNORED,                                                // dstQueueFamilyIndex
				sourceImage,                                                            // image
				subresourceRange                                                        // subresourceRange
			);
			commandBuffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eAllCommands,
				vk::PipelineStageFlagBits::eTransfer,
				{}, {}, {}, sourceToTransferSrc);
		}

		// Transition blit target to eTransferDstOptimal
		vk::ImageMemoryBarrier blitTargetToTransferDst = vk::ImageMemoryBarrier(
			vk::AccessFlagBits::eNone,                      // srcAccessMask
			vk::AccessFlagBits::eTransferWrite,              // dstAccessMask
			vk::ImageLayout::eUndefined,                     // oldLayout
			vk::ImageLayout::eTransferDstOptimal,            // newLayout
			VK_QUEUE_FAMILY_IGNORED,                         // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                         // dstQueueFamilyIndex
			*blitTargetImage.GetImage(),                     // image
			subresourceRange                                 // subresourceRange
		);
		commandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe,
			vk::PipelineStageFlagBits::eTransfer,
			{}, {}, {}, blitTargetToTransferDst);

		// Blit source → target (GPU performs format conversion; half-float → unorm, BGRA → RGBA, etc.)
		vk::ImageBlit blitRegion = vk::ImageBlit(
			vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),                                                           // srcSubresource
			std::array<vk::Offset3D, 2>{vk::Offset3D(0, 0, 0), vk::Offset3D(static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1)},   // srcOffsets
			vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),                                                           // dstSubresource
			std::array<vk::Offset3D, 2>{vk::Offset3D(0, 0, 0), vk::Offset3D(static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1)}    // dstOffsets
		);
		commandBuffer.blitImage(
			sourceImage,                                     // srcImage
			vk::ImageLayout::eTransferSrcOptimal,            // srcImageLayout
			*blitTargetImage.GetImage(),                     // dstImage
			vk::ImageLayout::eTransferDstOptimal,            // dstImageLayout
			blitRegion,                                      // region
			vk::Filter::eNearest                             // filter
		);

		// Transition blit target to eTransferSrcOptimal for buffer copy
		vk::ImageMemoryBarrier blitTargetToTransferSrc = vk::ImageMemoryBarrier(
			vk::AccessFlagBits::eTransferWrite,              // srcAccessMask
			vk::AccessFlagBits::eTransferRead,               // dstAccessMask
			vk::ImageLayout::eTransferDstOptimal,            // oldLayout
			vk::ImageLayout::eTransferSrcOptimal,            // newLayout
			VK_QUEUE_FAMILY_IGNORED,                         // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                         // dstQueueFamilyIndex
			*blitTargetImage.GetImage(),                     // image
			subresourceRange                                 // subresourceRange
		);
		commandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eTransfer,
			{}, {}, {}, blitTargetToTransferSrc);

		// Copy blit target → staging buffer
		vk::BufferImageCopy copyRegion = vk::BufferImageCopy(
			0,                                              // bufferOffset
			0,                                              // bufferRowLength
			0,                                              // bufferImageHeight
			vk::ImageSubresourceLayers(
				vk::ImageAspectFlagBits::eColor,            // aspectMask
				0,                                          // mipLevel
				0,                                          // baseArrayLayer
				1                                           // layerCount
			),
			vk::Offset3D(0, 0, 0),                          // imageOffset
			vk::Extent3D(extent.width, extent.height, 1)    // imageExtent
		);
		commandBuffer.copyImageToBuffer(
			*blitTargetImage.GetImage(),                    // srcImage
			vk::ImageLayout::eTransferSrcOptimal,            // srcImageLayout
			stagingBuffer->GetBuffer(),                     // dstBuffer
			copyRegion                                      // region
		);

		// Restore source layout if we changed it
		if (sourceLayout != vk::ImageLayout::eTransferSrcOptimal)
		{
			vk::ImageMemoryBarrier sourceRestore = vk::ImageMemoryBarrier(
				vk::AccessFlagBits::eTransferRead,                                      // srcAccessMask
				vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,     // dstAccessMask
				vk::ImageLayout::eTransferSrcOptimal,                                   // oldLayout
				sourceLayout,                                                           // newLayout
				VK_QUEUE_FAMILY_IGNORED,                                                // srcQueueFamilyIndex
				VK_QUEUE_FAMILY_IGNORED,                                                // dstQueueFamilyIndex
				sourceImage,                                                            // image
				subresourceRange                                                        // subresourceRange
			);
			commandBuffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eAllCommands,
				{}, {}, {}, sourceRestore);
		}

		deviceContext.EndSingleTimeCommands(std::move(commandBuffer),
			deviceContext.GetGraphicsCommandPool(), deviceContext.GetGraphicsQueue());

		// Read pixel data from staging buffer
		std::vector<uint8_t> pixelData(bufferSize);
		void* mappedData = stagingBuffer->GetVmaAllocationInfo().pMappedData;
		if (mappedData)
		{
			memcpy(pixelData.data(), mappedData, bufferSize);
		}
		else
		{
			LOG_ERROR("Frame capture: failed to map staging buffer");
			return;
		}

		int writeResult = stbi_write_png(outputPath.c_str(),
			static_cast<int>(extent.width),
			static_cast<int>(extent.height),
			4,										// channels (RGBA)
			pixelData.data(),
			static_cast<int>(extent.width * 4));	// stride

		if (writeResult)
		{
			LOG_INFO("Frame captured to: {} ({}x{}, source format {})",
				outputPath, extent.width, extent.height, vk::to_string(format));
		}
		else
		{
			LOG_ERROR("Frame capture: failed to write PNG to {}", outputPath);
		}
	}
}
