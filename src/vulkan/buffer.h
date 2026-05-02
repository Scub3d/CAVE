#pragma once

#include <cstdint>
#include <memory>

#include "vk_mem_alloc.h"
#include "deviceContext.h"

namespace Cave
{

class Descriptor;

class Buffer : public std::enable_shared_from_this<Buffer>
{
private:
	DeviceContext& _deviceContext;

	vk::DeviceSize _size;
	vk::BufferUsageFlags _bufferUsageFlags;
	uint64_t _alignment;
	bool _shared;

	vk::Buffer _buffer;
	VmaAllocation _vmaAllocation;
	VmaAllocationInfo _vmaAllocationInfo;

	VmaMemoryUsage _vmaMemoryUsage;
	VmaAllocationCreateFlags _vmaAllocationCreateFlags;

	std::string _debugName;
	void* _pNext = nullptr;
	std::vector<std::tuple<std::weak_ptr<Descriptor>, uint32_t, uint32_t, vk::DescriptorType>> _boundDescriptors;

	void Alloc();

public:
	Buffer(DeviceContext& deviceContext, vk::DeviceSize size, vk::BufferUsageFlags bufferUsageFlags, VmaMemoryUsage vmaMemoryUsage, VmaAllocationCreateFlags vmaAllocationCreateFlags,
		   bool concurrentSharing = false, VkDeviceSize alignment = 0, std::string debugName = "Unnamed", void* pNext = nullptr);
	~Buffer();

	Buffer(const Buffer &) = delete;
	Buffer(Buffer &&) = delete;
	Buffer &operator=(const Buffer &) = delete;
	Buffer &operator=(Buffer &&) = delete;

	void Realloc(uint64_t newSize);
	void BoundToDescriptor(std::weak_ptr<Descriptor> descriptor, uint32_t set, uint32_t binding, vk::DescriptorType type);

	void Upload(const void *data, uint32_t size, uint32_t offset = 0);
	void UploadFrom(std::shared_ptr<Buffer> buffer);
	std::vector<char> Download();
	void DownloadTo(std::shared_ptr<Buffer> buffer, vk::DeviceSize srcOffset = 0, vk::DeviceSize dstOffset = 0);

	void ComputeWriteReadBarrier(vk::CommandBuffer commandBuffer);
	void ComputeReadWriteBarrier(vk::CommandBuffer commandBuffer);
	void ComputeWriteWriteBarrier(vk::CommandBuffer commandBuffer);

	void AssertEquals(char *data, size_t length);

	template <typename T>
	T ReadOne(vk::DeviceSize offset = 0)
	{
		if (_vmaMemoryUsage == VMA_MEMORY_USAGE_GPU_ONLY || _vmaMemoryUsage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)
		{
			const auto stagingBuffer = Buffer::Staging(_deviceContext, sizeof(T));
			DownloadTo(stagingBuffer, offset, 0);
			return *static_cast<T *>(stagingBuffer->GetVmaAllocationInfo().pMappedData);
		}
		else if (_vmaAllocationCreateFlags & VMA_ALLOCATION_CREATE_MAPPED_BIT)
		{
			return *(static_cast<T *>(_vmaAllocationInfo.pMappedData) + offset / sizeof(T));
		}
		else
		{
			throw std::runtime_error("Buffer is not mappable");
		}
	}

	VmaAllocationInfo &GetVmaAllocationInfo() { return _vmaAllocationInfo; }
	VmaAllocation GetVmaAllocation() const { return _vmaAllocation; }
	vk::Buffer GetBuffer() const { return _buffer; }
	vk::DeviceSize GetSize() const { return _size; }

	vk::DeviceAddress GetDeviceAddress() const
	{
		vk::BufferDeviceAddressInfo bufferDeviceAddressInfo = vk::BufferDeviceAddressInfo(
			_buffer // buffer
		);
		return _deviceContext.GetDevice().getBufferAddress(bufferDeviceAddressInfo);
	}

	static std::shared_ptr<Buffer> Uniform(DeviceContext& deviceContext, uint32_t size, bool concurrentSharing = false);
	static std::shared_ptr<Buffer> Staging(DeviceContext& deviceContext, unsigned long size);
	static std::shared_ptr<Buffer> Storage(DeviceContext& deviceContext, uint64_t size, bool concurrentSharing = false, vk::DeviceSize alignment = 0, std::string debugName = "Unnamed Storage Buffer");
};

} // namespace Cave
