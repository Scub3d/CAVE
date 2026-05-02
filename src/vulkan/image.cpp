#include "image.h"
#include "../common/logger.h"

namespace Cave
{

Image::Image(DeviceContext& deviceContext, vk::Format format, int arrayCount, int mipLevels, vk::Extent3D extent, vk::ImageTiling imageTiling, vk::ImageUsageFlags imageUsageFlags,
			 vk::MemoryPropertyFlags memoryPropertyFlags, vk::ImageCreateFlags imageCreateFlags, vk::ImageAspectFlags imageAspectFlags,
			 vk::ImageViewType imageViewType, vk::SharingMode sharingMode, std::vector<uint32_t> queueFamilyIndices, vk::ImageType imageType, void* pNext)
	: _deviceContext{deviceContext}, _format{format}, _arrayCount{arrayCount}, _mipLevels{mipLevels}, _extent{extent}, _imageTiling{imageTiling}, _imageUsageFlags{imageUsageFlags},
	  _memoryPropertyFlags{memoryPropertyFlags}, _imageCreateFlags{imageCreateFlags}, _imageAspectFlags{imageAspectFlags},
	  _imageViewType{imageViewType}, _imageType{imageType}, _sharingMode{sharingMode}, _queueFamilyIndices{queueFamilyIndices}, _pNext{pNext}
{
	CreateImage();
	CreateImageMemory();
	CreateImageView();
}

Image::Image(DeviceContext& deviceContext, vk::Image image, vk::Format format, vk::Extent2D extent)
	: _deviceContext{deviceContext}, _image{image}, _format{format}, _extent{extent}
{

	_imageAspectFlags = vk::ImageAspectFlagBits::eColor;
	_imageType = vk::ImageType::e2D;
	_mipLevels = 0;
	_arrayCount = 1;
	_sharingMode = vk::SharingMode::eExclusive;

	CreateImageView();
}

Image::~Image()
{
	_deviceContext.GetDevice().destroyImage(_image, {}, {});
	_deviceContext.GetDevice().freeMemory(_deviceMemory, {}, {});
	_deviceContext.GetDevice().destroyImageView(_imageView, {}, {});
}

void Image::CreateImage()
{
	LOG_DEBUG("Creating Vulkan Image");

	vk::ImageCreateInfo imageCreateInfo = vk::ImageCreateInfo(
		vk::ImageCreateFlags() | _imageCreateFlags,		   // flags
		_imageType,										   // imageType
		_format,										   // format
		_extent,										   // extent
		_mipLevels,										   // mipLevels
		_arrayCount,									   // arrayLayers
		vk::SampleCountFlagBits::e1,					   // samples
		_imageTiling,									   // tiling
		_imageUsageFlags,								   // usage
		_sharingMode,									   // sharingMode
		static_cast<uint32_t>(_queueFamilyIndices.size()), // queueFamilyIndexCount
		_queueFamilyIndices.data(),						   // pQueueFamilyIndices
		vk::ImageLayout::eUndefined,					   // initialLayout
		_pNext											   // pNext
	);

	try
	{
		_image = _deviceContext.GetDevice().createImage(imageCreateInfo);
	}
	catch (vk::SystemError& error)
	{
		LOG_ERROR("Failed to create Vulkan Image: {}", error.what());
		throw;
	}
}

void Image::CreateImageMemory()
{
	LOG_DEBUG("Allocating Memory For Vulkan Image AND Binding Vulkan Image To Memory");

	vk::MemoryRequirements memoryRequirements = _deviceContext.GetDevice().getImageMemoryRequirements(_image);

	uint32_t memoryTypeIndex = FindMemoryTypeIndex(memoryRequirements.memoryTypeBits);

	vk::MemoryAllocateInfo memoryAllocateInfo = vk::MemoryAllocateInfo(
		memoryRequirements.size, // allocationSize
		memoryTypeIndex			 // memoryTypeIndex
	);

	try
	{
		_deviceMemory = _deviceContext.GetDevice().allocateMemory(memoryAllocateInfo);
		_deviceContext.GetDevice().bindImageMemory(_image, _deviceMemory, 0);
	}
	catch (vk::SystemError& error)
	{
		// Log the request size + driver-reported alignment so OOMs at scale identify
		// which allocation pushed past the budget. memoryRequirements.size includes the
		// driver's tile/alignment overhead which can be 5-15% above the naive
		// width*height*depth*bpp calc for large 3D images.
		LOG_ERROR("Failed allocating memory for Vulkan Image: {} (requested {} MiB, alignment {} bytes, memoryTypeIndex {})",
			error.what(),
			memoryRequirements.size / (1024 * 1024),
			memoryRequirements.alignment,
			memoryTypeIndex);
		throw;
	}
}

void Image::CreateImageView()
{
	LOG_DEBUG("Creating Vulkan Image View");

	vk::ImageSubresourceRange imageSubresourceRange = vk::ImageSubresourceRange(
		_imageAspectFlags, // flags
		0,				   // baseMipLevel
		1,				   // levelCount
		0,				   // baseArrayLevel
		_arrayCount		   // layerCount
	);

	vk::ComponentMapping imageComponentMapping = vk::ComponentMapping(
		vk::ComponentSwizzle::eIdentity, // r
		vk::ComponentSwizzle::eIdentity, // g
		vk::ComponentSwizzle::eIdentity, // b
		vk::ComponentSwizzle::eIdentity	 // a
	);

	vk::ImageViewCreateInfo imageViewCreateInfo = vk::ImageViewCreateInfo(
		vk::ImageViewCreateFlags(), // flags
		_image,						// image
		_imageViewType,				// viewType
		_format,					// format
		imageComponentMapping,		// components
		imageSubresourceRange		// subresourceRange
	);

	_imageView = _deviceContext.GetDevice().createImageView(imageViewCreateInfo);
}

vk::Format Image::FindSupportedFormat(vk::PhysicalDevice physicalDevice, const std::vector<vk::Format> &candidates, vk::ImageTiling imageTiling, vk::FormatFeatureFlags formatFeatureFlags)
{
	for (vk::Format format : candidates)
	{
		vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(format);

		if (imageTiling == vk::ImageTiling::eLinear && (formatProperties.linearTilingFeatures & formatFeatureFlags) == formatFeatureFlags)
		{
			return format;
		}
		else if (imageTiling == vk::ImageTiling::eOptimal && (formatProperties.optimalTilingFeatures & formatFeatureFlags) == formatFeatureFlags)
		{
			return format;
		}
	}

	throw std::runtime_error("Failed to find supported format :(");
}

uint32_t Image::FindMemoryTypeIndex(uint32_t supportedMemoryIndices)
{
	vk::PhysicalDeviceMemoryProperties physicalDeviceMemoryProperties = _deviceContext.GetPhysicalDevice().getMemoryProperties();

	for (uint32_t memoryTypeIndex = 0; memoryTypeIndex < physicalDeviceMemoryProperties.memoryTypeCount; memoryTypeIndex++)
	{
		if (supportedMemoryIndices & (1 << memoryTypeIndex) && (physicalDeviceMemoryProperties.memoryTypes[memoryTypeIndex].propertyFlags & _memoryPropertyFlags) == _memoryPropertyFlags)
		{
			return memoryTypeIndex;
		}
	}

	throw std::runtime_error("Failed to find suitable memory type :(");
}

void Image::ComputeMemoryBarrier(vk::CommandBuffer commandBuffer, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::ImageSubresourceRange imageSubresourceRange, vk::AccessFlagBits srcAccessMask, vk::AccessFlagBits dstAccessMask)
{
	vk::ImageMemoryBarrier imageMemoryBarrier = vk::ImageMemoryBarrier(
		srcAccessMask,			 // srcAccessMask
		dstAccessMask,			 // dstAccessMask
		oldLayout,				 // oldLayout
		newLayout,				 // newLayout
		VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
		VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
		_image,					 // image
		imageSubresourceRange	 // subresourceRange
	);

	commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlagBits::eByRegion, nullptr, nullptr, imageMemoryBarrier);
}

} // namespace Cave