#pragma once

#include <vulkan/vulkan.hpp>

#include <vector>
#include <optional>
#include <set>
#include <cstdint>
#include <string>

#include "../common/logger.h"
#include "vk_mem_alloc.h"

#include "../common/structs.h"

namespace Cave
{

	class DeviceContext
	{
	public:
		DeviceContext() = default;
		~DeviceContext() = default;

		DeviceContext(const DeviceContext &) = delete;
		DeviceContext &operator=(const DeviceContext &) = delete;

	private: // Variables
		vk::Device _device;
		vk::PhysicalDevice _physicalDevice;

		QueueFamilies _queueFamilies;

		vk::Queue _graphicsQueue, _computeQueue, _presentQueue, _videoEncodeQueue;

		vk::Format _colorFormat;
		vk::Format _depthFormat;

		vk::CommandPool _graphicsCommandPool, _computeCommandPool;
		vk::CommandBuffer _commandBuffer;
		std::vector<vk::CommandBuffer> _commandBuffers;

		std::vector<vk::Fence> _computeFences, _graphicsFences;
		std::vector<vk::Semaphore> _computeFinishedSemaphores;

		vk::QueryPool _queryPool;
		float _timestampPeriod = 1.0f;
		vk::DescriptorPool _descriptorPool;

		VmaAllocator _vmaAllocator;

		std::vector<const char *> _activeDeviceLayer{};
		std::vector<const char *> _activeDeviceExtensions{};

		// Mutated in Initialize() — Looking Glass mode appends 6 extra extensions when
		// requested (external memory/semaphore + Win32 variants, dedicated allocation,
		// multiview). See _lookingGlassRequested below.
		std::vector<const char *> _deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
													   VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME, VK_KHR_SPIRV_1_4_EXTENSION_NAME, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
													   VK_KHR_VIDEO_QUEUE_EXTENSION_NAME, VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME, VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME};

		bool _debugMode = false;
		bool _lookingGlassRequested = false; // Set before Initialize() to opt into the LG ext bundle.

		vk::DispatchLoaderDynamic* _dldi = nullptr;
		uint32_t _framesInFlight = 2;

	public: // Variables
#ifdef NDEBUG
		const bool EnableValidationLayers = false;
#else
		const bool EnableValidationLayers = true;
#endif

	private: // Methods
		void CreateLogicalDevice();
		void CreateQueues();
		void CreateQueryPool();
		void SetupVMA(vk::Instance instance);

		void FindQueueFamilies(vk::SurfaceKHR surface);

		void CreateDescriptorPool(uint32_t framesInFlight);

	public: // Methods
		// Must be called before Initialize() to enable Vulkan-OpenGL interop extensions
		// for Looking Glass mode (external memory/semaphore + Win32 variants, dedicated
		// allocation, multiview). When false, the device is created with the original
		// extension set unchanged.
		void SetLookingGlassRequested(bool requested) { _lookingGlassRequested = requested; }
		bool IsLookingGlassRequested() const { return _lookingGlassRequested; }

		void Initialize(vk::Instance instance, vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface, bool debugMode, vk::DispatchLoaderDynamic* dldi);
		void FinishSetup(uint32_t framesInFlight);
		void CreateResources(uint32_t framesInFlight);
		void Cleanup();

		vk::Semaphore CreateSemaphore();
		vk::Semaphore CreateTimelineSemaphore(uint64_t initialValue = 0);
		vk::Fence CreateFence();

		vk::CommandBuffer CreateCommandBuffer(vk::CommandPool commandPool, vk::CommandBufferLevel commandBufferLevel = vk::CommandBufferLevel::ePrimary);

		vk::Format GetColorFormat() { return _colorFormat; }

		vk::Device GetDevice() { return _device; }
		vk::PhysicalDevice GetPhysicalDevice() { return _physicalDevice; }

		void *GetDevicePointer() { return (void *)&_device; }

		vk::CommandBuffer BeginSingleTimeCommands(vk::CommandPool commandPool);
		void EndSingleTimeCommands(vk::CommandBuffer commandBuffer, vk::CommandPool commandPool, vk::Queue queue);

		void CopyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size, vk::CommandPool commandPool, vk::Queue queue);

		vk::CommandPool CreateCommandPool(uint32_t queueFamilyIndex);

		vk::Queue GetGraphicsQueue() { return _graphicsQueue; }
		vk::Queue GetComputeQueue() { return _computeQueue; }
		vk::Queue GetPresentQueue() { return _presentQueue; }
		vk::Queue GetVideoEncodeQueue() { return _videoEncodeQueue; }

		vk::CommandPool GetComputeCommandPool() { return _computeCommandPool; }
		vk::CommandPool GetGraphicsCommandPool() { return _graphicsCommandPool; }

		vk::QueryPool GetQueryPool() { return _queryPool; }
		float GetTimestampPeriod() const { return _timestampPeriod; }
		vk::DescriptorPool GetDescriptorPool() { return _descriptorPool; }
		QueueFamilies GetQueueFamilies() { return _queueFamilies; }

		VmaAllocator &GetVmaAllocator() { return _vmaAllocator; }
		vk::DispatchLoaderDynamic &GetDispatchLoaderDynamic() { return *_dldi; }

		uint32_t GetFramesInFlight() const { return _framesInFlight; }

		vk::CommandBuffer GetCommandBuffer(uint32_t imageIndex) { return _commandBuffers[imageIndex]; }

		bool CheckDeviceExtensionSupport(vk::PhysicalDevice device);
		int RateDeviceSuitability(vk::PhysicalDevice physicalDevice);
	};

} // namespace Cave
