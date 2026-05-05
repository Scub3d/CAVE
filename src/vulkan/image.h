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
	vk::DeviceSize _allocatedMemorySize = 0; // size of _deviceMemory after allocation; needed by GL importers (glImportMemoryWin32HandleEXT requires the byte size)

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
	void* _pNext = nullptr;        // chain head for VkImageCreateInfo.pNext (e.g. VkExternalMemoryImageCreateInfo)
	void* _allocPNext = nullptr;   // chain head for VkMemoryAllocateInfo.pNext (e.g. VkExportMemoryAllocateInfo + VkMemoryDedicatedAllocateInfo)

	std::vector<uint32_t> _queueFamilyIndices;

	void CreateImage();
	void CreateImageMemory();
	void CreateImageView();

	uint32_t FindMemoryTypeIndex(uint32_t supportedMemoryIndices);

public:
	Image(DeviceContext &deviceContext, vk::Format format, int arrayCount, int mipLevels, vk::Extent3D extent, vk::ImageTiling imageTiling, vk::ImageUsageFlags imageUsageFlags,
		  vk::MemoryPropertyFlags memoryPropertyFlags, vk::ImageCreateFlags imageCreateFlags, vk::ImageAspectFlags imageAspectFlags,
		  vk::ImageViewType imageViewType, vk::SharingMode sharingMode, std::vector<uint32_t> queueFamilyIndices,
		  vk::ImageType imageType = vk::ImageType::e2D, void* pNext = nullptr, void* allocPNext = nullptr);
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

	// Accessors used by Vulkan-OpenGL interop. The device memory + size are needed
	// to call glImportMemoryWin32HandleEXT after a HANDLE has been exported via
	// vkGetMemoryWin32HandleKHR. Format is exposed so the GL side can pick a
	// matching glTextureStorageMem2DEXT internal format.
	vk::DeviceMemory GetDeviceMemory() const { return _deviceMemory; }
	vk::DeviceSize GetAllocatedMemorySize() const { return _allocatedMemorySize; }
	vk::Format GetFormat() const { return _format; }
	vk::Extent3D GetExtent() const { return _extent; }

	// --- Vulkan-OpenGL interop helpers ---------------------------------------
	// CreateExportable allocates a 2D image with VkExternalMemoryImageCreateInfo +
	// VkExportMemoryAllocateInfo + VkMemoryDedicatedAllocateInfo all chained for
	// Opaque Win32 HANDLE export. The resulting image's underlying VkDeviceMemory
	// can be exported via GetMemoryWin32Handle (Windows-only).
	static std::shared_ptr<Image> CreateExportable(DeviceContext& deviceContext,
												   vk::Format format,
												   vk::Extent3D extent,
												   vk::ImageUsageFlags usage);

	// Returns a duplicated Win32 HANDLE referencing this image's VkDeviceMemory.
	// Caller owns the HANDLE and must CloseHandle (unless transferred to GL).
	// Returns nullptr on non-Windows or if the image was not created exportable.
	void* GetMemoryWin32Handle();
};

} // namespace Cave
