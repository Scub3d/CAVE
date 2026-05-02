#include "buffer.h"
#include "descriptors.h"

#include <iostream>
#include <utility>

#include "../common/logger.h"

namespace Cave
{

Buffer::Buffer(DeviceContext& deviceContext, vk::DeviceSize size, vk::BufferUsageFlags bufferUsageFlags, VmaMemoryUsage vmaMemoryUsage,
			   VmaAllocationCreateFlags vmaAllocationCreateFlags, bool shared, vk::DeviceSize alignment, std::string debugName, void* pNext)
	: _deviceContext{deviceContext}, _size{size}, _alignment{alignment}, _shared{shared}, _bufferUsageFlags{bufferUsageFlags},
	  _vmaMemoryUsage{vmaMemoryUsage}, _vmaAllocationCreateFlags{vmaAllocationCreateFlags}, _vmaAllocation{nullptr}, _debugName{debugName}, _pNext{pNext}
{
	Alloc();
}

Buffer::~Buffer()
{
	LOG_DEBUG("Destroying Vulkan Buffer");
	vmaDestroyBuffer(_deviceContext.GetVmaAllocator(), static_cast<VkBuffer>(_buffer), _vmaAllocation);
}

void Buffer::Alloc()
{
	LOG_DEBUG("Allocating Vulkan Buffer");

	uint32_t queueFamilyIndices[] = {0, 0};

	vk::BufferCreateInfo bufferCreateInfo = vk::BufferCreateInfo(
		vk::BufferCreateFlags(),	 // flags
		_size,						 // size
		_bufferUsageFlags,			 // usage
		vk::SharingMode::eExclusive, // sharingMode
		{},							 // queueFamilyIndexCount
		{},							 // pQueueFamilyIndices
		nullptr						 // pNext
	);

	if (_shared)
	{
		queueFamilyIndices[0] = _deviceContext.GetQueueFamilies().GraphicsFamily.Index.value();
		queueFamilyIndices[1] = _deviceContext.GetQueueFamilies().ComputeFamily.Index.value();
		bufferCreateInfo.sharingMode = vk::SharingMode::eConcurrent;
		bufferCreateInfo.queueFamilyIndexCount = 2;
		bufferCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
	}

	auto vkBufferInfo = static_cast<VkBufferCreateInfo>(bufferCreateInfo);
	vkBufferInfo.pNext = _pNext;

	VmaAllocationCreateInfo vmaAllocationCreateInfo = {};
	vmaAllocationCreateInfo.usage = _vmaMemoryUsage;
	vmaAllocationCreateInfo.flags = _vmaAllocationCreateFlags;

	VkBuffer vkBuffer = VK_NULL_HANDLE;

	VkResult allocationResult;
	if (_alignment != 0)
	{
		allocationResult = vmaCreateBufferWithAlignment(_deviceContext.GetVmaAllocator(), &vkBufferInfo, &vmaAllocationCreateInfo, _alignment, &vkBuffer, &_vmaAllocation, &_vmaAllocationInfo);
	}
	else
	{
		allocationResult = vmaCreateBuffer(_deviceContext.GetVmaAllocator(), &vkBufferInfo, &vmaAllocationCreateInfo, &vkBuffer, &_vmaAllocation, &_vmaAllocationInfo);
	}
	if (allocationResult != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create buffer");
	}

	_buffer = vk::Buffer(vkBuffer);

	if (_deviceContext.EnableValidationLayers)
	{
		_deviceContext.GetDevice().setDebugUtilsObjectNameEXT(
			vk::DebugUtilsObjectNameInfoEXT{
				vk::ObjectType::eBuffer,
				reinterpret_cast<uint64_t>(static_cast<VkBuffer>(_buffer)),
				_debugName.c_str()},
			_deviceContext.GetDispatchLoaderDynamic());
	}
}

void Buffer::Realloc(uint64_t newSize)
{
	vmaDestroyBuffer(_deviceContext.GetVmaAllocator(), static_cast<VkBuffer>(_buffer), _vmaAllocation);

	_size = newSize;
	Alloc();

	vk::DescriptorBufferInfo bufferInfo = vk::DescriptorBufferInfo(
		_buffer,                   // buffer
		_vmaAllocationInfo.offset, // offset
		_size                      // range
	);

	std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
	for (auto &tuple : _boundDescriptors)
	{
		auto descriptor = std::get<0>(tuple);
		auto shared = descriptor.lock();
		if (shared)
		{
			writeDescriptorSets.emplace_back(
				shared->GetDescriptorSet(std::get<1>(tuple)), // dstSet
				std::get<2>(tuple),                           // dstBinding
				0,                                            // dstArrayElement
				1,                                            // descriptorCount
				std::get<3>(tuple),                           // descriptorType
				nullptr,                                      // pImageInfo
				&bufferInfo                                   // pBufferInfo
			);
		}
	}
	if (!writeDescriptorSets.empty())
	{
		_deviceContext.GetDevice().updateDescriptorSets(writeDescriptorSets, nullptr);
	}
}

void Buffer::Upload(const void *data, uint32_t size, uint32_t offset)
{
	if (size + offset > this->_size)
	{
		throw std::runtime_error("Buffer overflow");
	}

	if (_vmaMemoryUsage == VMA_MEMORY_USAGE_GPU_ONLY || _vmaMemoryUsage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)
	{
		auto stagingBuffer = Buffer::Staging(_deviceContext, size);

		memcpy(stagingBuffer->GetVmaAllocationInfo().pMappedData, ((char *)data) + offset, size);

		vk::CommandBuffer commandBuffer = _deviceContext.BeginSingleTimeCommands(_deviceContext.GetGraphicsCommandPool());

		vk::BufferCopy copyRegion = {};
		copyRegion.setSize(size);

		commandBuffer.copyBuffer(stagingBuffer->GetBuffer(), _buffer, 1, &copyRegion);

		// Pipeline barrier: make the transfer write visible to all subsequent GPU reads.
		// Without this, the draw indirect / shader read / vertex attribute read hardware
		// may not see the uploaded data due to GPU cache coherence timing.
		vk::BufferMemoryBarrier bufferMemoryBarrier = vk::BufferMemoryBarrier(
			vk::AccessFlagBits::eTransferWrite,                                                      // srcAccessMask
			vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |                     // dstAccessMask
				vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eVertexAttributeRead,
			VK_QUEUE_FAMILY_IGNORED,                                                                 // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED,                                                                 // dstQueueFamilyIndex
			_buffer,                                                                                 // buffer
			0,                                                                                       // offset
			size                                                                                     // size
		);
		commandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer,                                                    // srcStageMask
			vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eDrawIndirect |   // dstStageMask
				vk::PipelineStageFlagBits::eVertexInput,
			{}, {}, bufferMemoryBarrier, {});

		_deviceContext.EndSingleTimeCommands(std::move(commandBuffer), _deviceContext.GetGraphicsCommandPool(), _deviceContext.GetGraphicsQueue());
	}
	else if (_vmaAllocationCreateFlags & VMA_ALLOCATION_CREATE_MAPPED_BIT)
	{
		memcpy(_vmaAllocationInfo.pMappedData, ((char *)data) + offset, size);
	}
	else
	{
		throw std::runtime_error("Buffer is not mappable");
	}
}

void Buffer::UploadFrom(std::shared_ptr<Buffer> buffer)
{
	if (buffer->GetSize() > _size)
	{
		throw std::runtime_error("Buffer overflow");
	}

	if (_vmaMemoryUsage == VMA_MEMORY_USAGE_GPU_ONLY || _vmaMemoryUsage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)
	{
		vk::CommandBuffer commandBuffer = _deviceContext.BeginSingleTimeCommands(_deviceContext.GetGraphicsCommandPool());

		vk::BufferCopy copyRegion = {};
		copyRegion.setSize(buffer->GetSize());

		commandBuffer.copyBuffer(buffer->GetBuffer(), this->_buffer, 1, &copyRegion);

		_deviceContext.EndSingleTimeCommands(std::move(commandBuffer), _deviceContext.GetGraphicsCommandPool(), _deviceContext.GetGraphicsQueue());
	}
	else if (_vmaAllocationCreateFlags & VMA_ALLOCATION_CREATE_MAPPED_BIT)
	{
		memcpy(_vmaAllocationInfo.pMappedData, buffer->GetVmaAllocationInfo().pMappedData, buffer->GetSize());
	}
	else
	{
		throw std::runtime_error("Buffer is not mappable");
	}
}

std::vector<char> Buffer::Download()
{
	auto stagingBuffer = Buffer::Staging(_deviceContext, _size);
	DownloadTo(stagingBuffer);
	return {
		(char *)stagingBuffer->GetVmaAllocationInfo().pMappedData,
		((char *)stagingBuffer->GetVmaAllocationInfo().pMappedData) + _size};
}

void Buffer::DownloadTo(std::shared_ptr<Buffer> buffer, vk::DeviceSize srcOffset, vk::DeviceSize dstOffset)
{
	if (_vmaMemoryUsage == VMA_MEMORY_USAGE_GPU_ONLY || _vmaMemoryUsage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)
	{
		vk::CommandBuffer commandBuffer = _deviceContext.BeginSingleTimeCommands(_deviceContext.GetGraphicsCommandPool());

		vk::BufferCopy copyRegion = {};
		copyRegion.srcOffset = srcOffset;
		copyRegion.dstOffset = dstOffset;
		copyRegion.setSize(buffer->_size);

		commandBuffer.copyBuffer(this->_buffer, buffer->GetBuffer(), 1, &copyRegion);

		_deviceContext.EndSingleTimeCommands(std::move(commandBuffer), _deviceContext.GetGraphicsCommandPool(), _deviceContext.GetGraphicsQueue());
	}
	else if (_vmaAllocationCreateFlags & VMA_ALLOCATION_CREATE_MAPPED_BIT)
	{
		memcpy(buffer->GetVmaAllocationInfo().pMappedData, _vmaAllocationInfo.pMappedData, buffer->GetSize());
	}
	else
	{
		throw std::runtime_error("Buffer is not mappable");
	}
}

void Buffer::BoundToDescriptor(std::weak_ptr<Descriptor> descriptor, uint32_t set, uint32_t binding, vk::DescriptorType type)
{
	_boundDescriptors.push_back({descriptor, set, binding, type});
}

std::shared_ptr<Buffer> Buffer::Uniform(DeviceContext& deviceContext, uint32_t size, bool concurrentSharing)
{
	return std::make_shared<Buffer>(deviceContext, size, vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
									VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT, concurrentSharing);
}

std::shared_ptr<Buffer> Buffer::Staging(DeviceContext& deviceContext, unsigned long size)
{
	return std::make_shared<Buffer>(deviceContext, size, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
									VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
									false);
}

std::shared_ptr<Buffer> Buffer::Storage(DeviceContext& deviceContext, uint64_t size, bool concurrentSharing, vk::DeviceSize alignment, std::string debugName)
{
	return std::make_shared<Buffer>(deviceContext, size, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
									concurrentSharing, alignment, debugName);
}

void Buffer::ComputeWriteReadBarrier(vk::CommandBuffer commandBuffer)
{
	vk::BufferMemoryBarrier bufferMemoryBarrier = vk::BufferMemoryBarrier(
		vk::AccessFlagBits::eShaderWrite,								// srcAccessMask
		vk::AccessFlagBits::eShaderRead,								// dstAccessMask
		_deviceContext.GetQueueFamilies().ComputeFamily.Index.value(), // srcQueueFamilyIndex
		_deviceContext.GetQueueFamilies().ComputeFamily.Index.value(), // dstQueueFamilyIndex
		_buffer,														// buffer
		0,																// offset
		_size															// size
	);

	commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags(), nullptr, bufferMemoryBarrier, nullptr);
}

void Buffer::ComputeReadWriteBarrier(vk::CommandBuffer commandBuffer)
{
	vk::BufferMemoryBarrier bufferMemoryBarrier = vk::BufferMemoryBarrier(
		vk::AccessFlagBits::eShaderRead,								// srcAccessMask
		vk::AccessFlagBits::eShaderWrite,								// dstAccessMask
		_deviceContext.GetQueueFamilies().ComputeFamily.Index.value(), // srcQueueFamilyIndex
		_deviceContext.GetQueueFamilies().ComputeFamily.Index.value(), // dstQueueFamilyIndex
		_buffer,														// buffer
		0,																// offset
		_size															// size
	);

	commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags(), nullptr, bufferMemoryBarrier, nullptr);
}

void Buffer::ComputeWriteWriteBarrier(vk::CommandBuffer commandBuffer)
{
	vk::BufferMemoryBarrier bufferMemoryBarrier = vk::BufferMemoryBarrier(
		vk::AccessFlagBits::eShaderWrite,								// srcAccessMask
		vk::AccessFlagBits::eShaderWrite,								// dstAccessMask
		_deviceContext.GetQueueFamilies().ComputeFamily.Index.value(), // srcQueueFamilyIndex
		_deviceContext.GetQueueFamilies().ComputeFamily.Index.value(), // dstQueueFamilyIndex
		_buffer,														// buffer
		0,																// offset
		_size															// size
	);

	commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags(), nullptr, bufferMemoryBarrier, nullptr);
}

void Buffer::AssertEquals(char *data, size_t length)
{
	if (length > _size)
	{
		throw std::runtime_error("Buffer overflow");
	}

	if (_vmaMemoryUsage == VMA_MEMORY_USAGE_GPU_ONLY || _vmaMemoryUsage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)
	{
		auto stagingBuffer = Buffer::Staging(_deviceContext, length);
		DownloadTo(stagingBuffer);
		if (memcmp(data, stagingBuffer->GetVmaAllocationInfo().pMappedData, length) != 0)
		{
			throw std::runtime_error("Buffer content does not match");
		}
	}
	else if (_vmaAllocationCreateFlags & VMA_ALLOCATION_CREATE_MAPPED_BIT)
	{
		if (memcmp(data, _vmaAllocationInfo.pMappedData, length) != 0)
		{
			throw std::runtime_error("Buffer content does not match");
		}
	}
	else
	{
		throw std::runtime_error("Buffer is not mappable");
	}
}

} // namespace Cave