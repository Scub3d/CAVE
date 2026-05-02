#include "texture.h"
#include "../common/logger.h"

namespace Cave
{

void Texture::CreateTexture()
{
	vk::ImageCreateInfo imageCreateInfo = vk::ImageCreateInfo(
		_imageCreateFlags,								   // flags
		vk::ImageType::e2D,								   // imageType
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
		nullptr											   // pNext
	);

	VmaAllocationCreateInfo vmaAllocationCreateInfo{};
	vmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
	vmaAllocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

	VkImage image;

	VkImageCreateInfo vkImageCreateInfo = static_cast<VkImageCreateInfo>(imageCreateInfo);
	VkResult result = vmaCreateImage(_deviceContext.GetVmaAllocator(), &vkImageCreateInfo, &vmaAllocationCreateInfo, &image, &_vmaAllocation, &_vmaAllocationInfo);

	if (result != VK_SUCCESS)
	{
		LOG_ERROR("Failed to create image, VkResult: {}", static_cast<int>(result));
		throw std::runtime_error("failed to create image");
	}

	_image = vk::Image(image);

	vmaBindImageMemory(_deviceContext.GetVmaAllocator(), _vmaAllocation, image);
}

} // namespace Cave