#pragma once

#include <vector>
#include "vk_mem_alloc.h"
#include "deviceContext.h"

namespace Cave
{

class Image : public std::enable_shared_from_this<Image>
{
private:
	DeviceContext &_deviceContext;

	vk::Image _image;
	vk::ImageView _imageView;
	vk::DeviceMemory _deviceMemory;

	vk::Format _format;
	int _arrayCount, _mipLevels;
	vk::Extent3D _extent;
	vk::ImageTiling _imageTiling;
	vk::ImageUsageFlags _imageUsageFlags;
	vk::MemoryPropertyFlags _memoryPropertyFlags;
	vk::ImageCreateFlags _imageCreateFlags;
	vk::ImageAspectFlags _imageAspectFlags;
	vk::SharingMode _sharingMode;
	vk::ImageViewType _imageViewType;
	vk::ImageType _imageType;
	void* _pNext = nullptr;

	std::vector<uint32_t> _queueFamilyIndices;

	void CreateImage();
	void CreateImageMemory();
	void CreateImageView();

	uint32_t FindMemoryTypeIndex(uint32_t supportedMemoryIndices);

public:
	Image(DeviceContext &deviceContext, vk::Format format, int arrayCount, int mipLevels, vk::Extent3D extent, vk::ImageTiling imageTiling, vk::ImageUsageFlags imageUsageFlags,
		  vk::MemoryPropertyFlags memoryPropertyFlags, vk::ImageCreateFlags imageCreateFlags, vk::ImageAspectFlags imageAspectFlags,
		  vk::ImageViewType imageViewType, vk::SharingMode sharingMode, std::vector<uint32_t> queueFamilyIndices,
		  vk::ImageType imageType = vk::ImageType::e2D, void* pNext = nullptr);
	Image(DeviceContext &deviceContext, vk::Image image, vk::Format format, vk::Extent2D extent);
	~Image();

	Image(const Image &) = delete;
	Image(Image &&) = delete;
	Image &operator=(const Image &) = delete;
	Image &operator=(Image &&) = delete;

	void ComputeMemoryBarrier(vk::CommandBuffer commandBuffer, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::ImageSubresourceRange imageSubresourceRange, vk::AccessFlagBits srcAccessMask, vk::AccessFlagBits dstAccessMask);

	static vk::Format FindSupportedFormat(vk::PhysicalDevice physicalDevice, const std::vector<vk::Format> &candidates, vk::ImageTiling imageTiling, vk::FormatFeatureFlags formatFeatureFlags);

	vk::Image *GetImage() { return &_image; }
	vk::ImageView *GetImageView() { return &_imageView; }
};

} // namespace Cave
