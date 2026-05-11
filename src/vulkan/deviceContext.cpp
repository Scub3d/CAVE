// Win32 platform support must be requested BEFORE the vulkan.hpp include (via
// deviceContext.h) so the dispatch loader's init() resolves the Win32 entry
// points (vkGetMemoryWin32HandleKHR / vkGetSemaphoreWin32HandleKHR / etc.).
// Without this define, those function pointers stay null in the dldi struct
// from this TU's perspective, even after init() runs.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define VK_USE_PLATFORM_WIN32_KHR
// Windows.h defines CreateSemaphore / CreateFence as macros that expand to the A/W
// variants. Our DeviceContext has methods of the same names — undef so the methods
// keep their identifiers. Same pattern used in vulkanInstance.h.
#ifdef CreateSemaphore
#undef CreateSemaphore
#endif
#ifdef CreateFence
#undef CreateFence
#endif
#endif

#include "deviceContext.h"
#include <common/logger.h>

namespace Cave
{

	void DeviceContext::Initialize(vk::Instance instance, vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface, bool debugMode, vk::DispatchLoaderDynamic* dldi)
	{
		_physicalDevice = physicalDevice;
		_debugMode = debugMode;
		_dldi = dldi;

		vk::PhysicalDeviceProperties selectedDeviceProperties = _physicalDevice.getProperties();
		_timestampPeriod = selectedDeviceProperties.limits.timestampPeriod;

		const auto& deviceLimits = selectedDeviceProperties.limits;
		LOG_INFO("Device caps: maxImageDim2D={}, maxImageDim3D={}, maxStorageBufferRange={} GiB, maxComputeWorkGroupCount=({},{},{}), maxComputeWorkGroupInvocations={}",
			deviceLimits.maxImageDimension2D,
			deviceLimits.maxImageDimension3D,
			deviceLimits.maxStorageBufferRange / (1024ULL * 1024 * 1024),
			deviceLimits.maxComputeWorkGroupCount[0],
			deviceLimits.maxComputeWorkGroupCount[1],
			deviceLimits.maxComputeWorkGroupCount[2],
			deviceLimits.maxComputeWorkGroupInvocations);

		// Log device-local heap size per GPU so we know the true VRAM ceiling.
		vk::PhysicalDeviceMemoryProperties memoryProperties = _physicalDevice.getMemoryProperties();
		for (uint32_t heapIndex = 0; heapIndex < memoryProperties.memoryHeapCount; heapIndex++)
		{
			const auto& heap = memoryProperties.memoryHeaps[heapIndex];
			bool deviceLocal = static_cast<bool>(heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
			LOG_INFO("Memory heap {}: size={} GiB, deviceLocal={}",
				heapIndex, heap.size / (1024ULL * 1024 * 1024), deviceLocal);
		}

		// Looking Glass mode needs Vulkan-OpenGL interop extensions on top of the base
		// set. Append before CreateLogicalDevice so they're enabled at device creation.
		// Validation step happens implicitly: vkCreateDevice fails loudly if any are
		// unsupported on the chosen GPU. Enabled only when explicitly requested so we
		// don't refuse otherwise-suitable GPUs that lack one of these extensions.
		if (_lookingGlassRequested)
		{
			_deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
			_deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
			_deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
			_deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
			_deviceExtensions.push_back(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
			_deviceExtensions.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);
			LOG_INFO("Looking Glass mode: enabling 6 additional device extensions for Vulkan-OpenGL interop + multiview");
		}

		FindQueueFamilies(surface);
		CreateLogicalDevice();
		CreateQueues();
		SetupVMA(instance);
		_colorFormat = vk::Format::eR8G8B8A8Unorm;

		// After the logical device is created, extend the dispatch loader to also
		// resolve device-level extension functions (vkGetMemoryWin32HandleKHR,
		// vkGetSemaphoreWin32HandleKHR, etc.). Without this, Win32 export calls
		// dereference null function pointers and segfault. Only matters for LG
		// mode but cheap to do unconditionally.
		if (_dldi)
		{
			_dldi->init(instance, vkGetInstanceProcAddr, _device, vkGetDeviceProcAddr);
		}
	}

	void DeviceContext::FinishSetup(uint32_t framesInFlight)
	{
		_framesInFlight = framesInFlight;
		CreateResources(framesInFlight);
		CreateQueryPool();
		CreateDescriptorPool(framesInFlight);
	}

	void DeviceContext::CreateResources(uint32_t framesInFlight)
	{
		LOG_DEBUG("Creating Vulkan Resources");

		_computeCommandPool = CreateCommandPool(_queueFamilies.ComputeFamily.Index.value());
		_graphicsCommandPool = CreateCommandPool(_queueFamilies.GraphicsFamily.Index.value());

		_commandBuffer = CreateCommandBuffer(_graphicsCommandPool);
		_commandBuffers.resize(framesInFlight);

		_computeFences.resize(framesInFlight);
		_graphicsFences.resize(framesInFlight);

		_computeFinishedSemaphores.resize(framesInFlight);

		for (uint32_t imageIndex = 0; imageIndex < framesInFlight; imageIndex++)
		{
			_commandBuffers[imageIndex] = CreateCommandBuffer(_graphicsCommandPool);

			_computeFences[imageIndex] = CreateFence();
			_graphicsFences[imageIndex] = CreateFence();

			_computeFinishedSemaphores[imageIndex] = CreateSemaphore();
		}
	}

	void DeviceContext::Cleanup()
	{
		vmaDestroyAllocator(_vmaAllocator);
	}

	void DeviceContext::CreateLogicalDevice()
	{
		LOG_DEBUG("Creating Vulkan Logical Device");

		std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilyIndices;
		std::vector<QueueFamily> uniqueQueueFamilies;

		for (const QueueFamily *family : {&_queueFamilies.ComputeFamily, &_queueFamilies.GraphicsFamily, &_queueFamilies.PresentFamily, &_queueFamilies.VideoEncodeFamily})
		{
			if (family->Index.has_value() && uniqueQueueFamilyIndices.insert(family->Index.value()).second)
			{
				uniqueQueueFamilies.push_back(*family);
			}
		}

		for (QueueFamily queueFamily : uniqueQueueFamilies)
		{
			float *const queuePriorities = new float[queueFamily.QueueCount];
			std::fill_n(queuePriorities, queueFamily.QueueCount, 1.0f);

			queueCreateInfos.push_back(vk::DeviceQueueCreateInfo(
				vk::DeviceQueueCreateFlags(),
				queueFamily.Index.value(),
				queueFamily.QueueCount,
				queuePriorities,
				nullptr));
		}

		vk::PhysicalDeviceFeatures deviceFeatures = _physicalDevice.getFeatures();
		deviceFeatures.setMultiDrawIndirect(true);
		deviceFeatures.setShaderInt64(true);
		deviceFeatures.setMultiViewport(true);
		deviceFeatures.setSamplerAnisotropy(true);
		// Required so the search-mode Slang shaders can index storage-buffer arrays
		// via gl_GlobalInvocationID.y (descriptor arrays per slot, single 2D dispatch).
		deviceFeatures.setShaderStorageBufferArrayDynamicIndexing(true);

		vk::PhysicalDeviceSynchronization2Features synchronization2Features = vk::PhysicalDeviceSynchronization2Features(
			vk::True // synchronization2
					 // pNext
		);

		vk::PhysicalDeviceVulkan12Features vulkan12Features{};
		vulkan12Features.timelineSemaphore = vk::True;
		vulkan12Features.bufferDeviceAddress = vk::True;
		vulkan12Features.hostQueryReset = vk::True;
		vulkan12Features.pNext = &synchronization2Features;

		vk::PhysicalDeviceDynamicRenderingFeaturesKHR dynamicRenderingFeatures = vk::PhysicalDeviceDynamicRenderingFeaturesKHR(
			vk::True,			  // dynamicRendering
			&vulkan12Features	  // pNext
		);

		// Looking Glass renders 48 views per frame; the quilt renderer uses
		// VK_KHR_multiview to amortize draw-call overhead across views via gl_ViewIndex.
		// Feature chain only extended when LG is requested — keeps unrelated runs lean.
		vk::PhysicalDeviceMultiviewFeatures multiviewFeatures{};
		if (_lookingGlassRequested)
		{
			multiviewFeatures.multiview = vk::True;
			multiviewFeatures.pNext = &dynamicRenderingFeatures;
		}
		void* featuresChainHead = _lookingGlassRequested
			? static_cast<void*>(&multiviewFeatures)
			: static_cast<void*>(&dynamicRenderingFeatures);

		vk::DeviceCreateInfo deviceInfo = vk::DeviceCreateInfo(
			vk::DeviceCreateFlags(),							   // flags
			static_cast<uint32_t>(queueCreateInfos.size()),		   // queueCreateInfoCount
			queueCreateInfos.data(),							   // pQueueCreateInfo
			0,													   // enabledLayerCount
			nullptr,											   // ppEnabledLayers
			static_cast<uint32_t>(_deviceExtensions.size()),	   // enabledExtensionCount
			_deviceExtensions.data(),							   // ppEnabledExtension
			&deviceFeatures,									   // pEnabledFeatures
			featuresChainHead									   // pNext (multiview before dynamicRendering when LG; else dynamicRendering directly)
		);

		try
		{
			_device = _physicalDevice.createDevice(deviceInfo);
			LOG_DEBUG("Created Vulkan Logical Device");
		}
		catch (vk::SystemError error)
		{
			LOG_ERROR("Failed To Create Vulkan Logical Device: {}", error.what());
		}
	}

	int DeviceContext::RateDeviceSuitability(vk::PhysicalDevice physicalDevice)
	{
		LOG_DEBUG("Rating device suitablility");

		vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();
		vk::PhysicalDeviceFeatures deviceFeatures = physicalDevice.getFeatures();

		if (CheckDeviceExtensionSupport(physicalDevice))
		{
			LOG_DEBUG("The device supports the requested extensions");
		}
		else
		{
			LOG_ERROR("The device does not support the requested extensions");
			return 0;
		}

		if (!deviceFeatures.samplerAnisotropy)
		{
			LOG_ERROR("The device does not support sampler anisotropy");
			return 0;
		}

		int score = 0;

		if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
		{
			score += 1000;
		}

		score += deviceProperties.limits.maxImageDimension2D;

		return score;
	}

	bool DeviceContext::CheckDeviceExtensionSupport(vk::PhysicalDevice device)
	{
		vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
		std::vector<vk::ExtensionProperties> supportedDeviceExtensions = device.enumerateDeviceExtensionProperties();
		std::set<std::string> requiredExtensions(_deviceExtensions.begin(), _deviceExtensions.end());

		LOG_DEBUG("Requesting the following extensions:");

		for (auto requiredExtension : requiredExtensions)
		{
			LOG_DEBUG("\t\"{}\"", requiredExtension);
		}

		for (const vk::ExtensionProperties &extension : supportedDeviceExtensions)
		{
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	void DeviceContext::FindQueueFamilies(vk::SurfaceKHR surface)
	{
		LOG_DEBUG("Finding Vulkan Queue Families");

		if (_physicalDevice == nullptr)
		{
			LOG_ERROR("Unable to find queue families because there is no physical device");
		}

		std::vector<vk::QueueFamilyProperties> queueFamiliesProperties = _physicalDevice.getQueueFamilyProperties();

		int queueFamilyIndex = 0;
		for (const vk::QueueFamilyProperties &queueFamilyProperties : queueFamiliesProperties)
		{
			if (queueFamilyProperties.queueFlags & vk::QueueFlagBits::eGraphics && !_queueFamilies.GraphicsFamily.Index.has_value())
			{
				_queueFamilies.GraphicsFamily.Index = queueFamilyIndex;
				_queueFamilies.GraphicsFamily.QueueCount = 1;
				_queueFamilies.GraphicsFamily.Priority = 1.0f;
			}

			if (queueFamilyProperties.queueFlags & vk::QueueFlagBits::eCompute)
			{
				if (!(queueFamilyProperties.queueFlags & vk::QueueFlagBits::eGraphics))
				{
					if (!_queueFamilies.ComputeFamily.Index.has_value())
					{
						_queueFamilies.ComputeFamily.Index = queueFamilyIndex;
						_queueFamilies.ComputeFamily.QueueCount = queueFamilyProperties.queueCount;
						_queueFamilies.ComputeFamily.Priority = 0.9f;
					}

					if (queueFamilyProperties.queueCount > _queueFamilies.ComputeFamily.QueueCount)
					{
						_queueFamilies.ComputeFamily.Index = queueFamilyIndex;
						_queueFamilies.ComputeFamily.QueueCount = queueFamilyProperties.queueCount;
					}
				}
			}

			if (queueFamilyProperties.queueFlags & vk::QueueFlagBits::eVideoEncodeKHR && !_queueFamilies.VideoEncodeFamily.Index.has_value())
			{
				_queueFamilies.VideoEncodeFamily.Index = queueFamilyIndex;
				_queueFamilies.VideoEncodeFamily.QueueCount = 1;
				_queueFamilies.VideoEncodeFamily.Priority = 0.8f;
			}

			if (_physicalDevice.getSurfaceSupportKHR(queueFamilyIndex, surface))
			{
				if (!_queueFamilies.PresentFamily.Index.has_value())
				{
					_queueFamilies.PresentFamily.Index = queueFamilyIndex;
					_queueFamilies.PresentFamily.QueueCount = 1;
					_queueFamilies.PresentFamily.Priority = 1.0f;
				}

				// Prefer a queue family that supports both graphics and present
				if (queueFamilyProperties.queueFlags & vk::QueueFlagBits::eGraphics)
				{
					_queueFamilies.PresentFamily.Index = queueFamilyIndex;
					_queueFamilies.PresentFamily.QueueCount = 1;
					_queueFamilies.PresentFamily.Priority = 1.0f;
				}
			}

			queueFamilyIndex++;
		}
	}

	void DeviceContext::CreateQueues()
	{
		LOG_DEBUG("Creating Vulkan Queues");

		_computeQueue = _device.getQueue(_queueFamilies.ComputeFamily.Index.value(), 0);
		_graphicsQueue = _device.getQueue(_queueFamilies.GraphicsFamily.Index.value(), 0);

		if (_queueFamilies.PresentFamily.Index.has_value())
		{
			_presentQueue = _device.getQueue(_queueFamilies.PresentFamily.Index.value(), 0);
		}

		if (_queueFamilies.VideoEncodeFamily.Index.has_value())
		{
			_videoEncodeQueue = _device.getQueue(_queueFamilies.VideoEncodeFamily.Index.value(), 0);
		}
	}

	void DeviceContext::SetupVMA(vk::Instance instance)
	{
		LOG_DEBUG("Setting Up VMA");

		VmaAllocatorCreateInfo allocatorInfo = {};
		allocatorInfo.physicalDevice = _physicalDevice;
		allocatorInfo.device = (VkDevice)_device;
		allocatorInfo.instance = (VkInstance)instance;
		allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
		vmaCreateAllocator(&allocatorInfo, &_vmaAllocator);
	}

	vk::CommandPool DeviceContext::CreateCommandPool(uint32_t queueFamilyIndex)
	{
		LOG_DEBUG("Creating Vulkan Command Pool");

		vk::CommandPoolCreateInfo commandPoolCreateInfo = vk::CommandPoolCreateInfo(
			vk::CommandPoolCreateFlags() | vk::CommandPoolCreateFlagBits::eResetCommandBuffer, // flags
			queueFamilyIndex																   // queueFamilyIndex
		);

		try
		{
			return _device.createCommandPool(commandPoolCreateInfo);
		}
		catch (vk::SystemError error)
		{
			LOG_ERROR("Failed To Create Vulkan Command Pool");
			return nullptr;
		}
	}

	vk::CommandBuffer DeviceContext::CreateCommandBuffer(vk::CommandPool commandPool, vk::CommandBufferLevel commandBufferLevel)
	{
		LOG_DEBUG("Creating Vulkan Command Buffer");

		vk::CommandBufferAllocateInfo commandBufferAllocateInfo = vk::CommandBufferAllocateInfo(
			commandPool,		// commandPool
			commandBufferLevel, // level
			1					// commandBufferCount
		);

		try
		{
			return _device.allocateCommandBuffers(commandBufferAllocateInfo)[0];
		}
		catch (vk::SystemError error)
		{
			LOG_FATAL("Failed to allocate command buffer for swapchain frame");
			return nullptr;
		}
	}

	vk::Semaphore DeviceContext::CreateSemaphore()
	{
		LOG_DEBUG("Creating Vulkan Semaphore");

		vk::SemaphoreCreateInfo semaphoreCreateInfo = vk::SemaphoreCreateInfo(
			vk::SemaphoreCreateFlags() // flags
		);

		try
		{
			return _device.createSemaphore(semaphoreCreateInfo);
		}
		catch (vk::SystemError error)
		{
			LOG_ERROR("Failed to create semaphore");
			return nullptr;
		}
	}

	vk::Semaphore DeviceContext::CreateTimelineSemaphore(uint64_t initialValue)
	{
		LOG_DEBUG("Creating Vulkan Timeline Semaphore with initial value {}", initialValue);

		vk::SemaphoreTypeCreateInfo semaphoreTypeCreateInfo = vk::SemaphoreTypeCreateInfo(
			vk::SemaphoreType::eTimeline, // semaphoreType
			initialValue                  // initialValue
		);

		vk::SemaphoreCreateInfo semaphoreCreateInfo = vk::SemaphoreCreateInfo(
			vk::SemaphoreCreateFlags() // flags
		);
		semaphoreCreateInfo.pNext = &semaphoreTypeCreateInfo;

		try
		{
			return _device.createSemaphore(semaphoreCreateInfo);
		}
		catch (vk::SystemError error)
		{
			LOG_ERROR("Failed to create timeline semaphore");
			return nullptr;
		}
	}

	vk::Fence DeviceContext::CreateFence()
	{
		LOG_DEBUG("Creating Vulkan Fence");
		vk::FenceCreateInfo fenceCreateInfo = vk::FenceCreateInfo(
			vk::FenceCreateFlags() | vk::FenceCreateFlagBits::eSignaled // flags
		);

		try
		{
			return _device.createFence(fenceCreateInfo);
		}
		catch (vk::SystemError error)
		{
			LOG_ERROR("Failed to create fence");
			return nullptr;
		}
	}

	void DeviceContext::CreateDescriptorPool(uint32_t framesInFlight)
	{
		// Sized to handle very large search-mode chunk counts. Each SearchSystem at
		// gridCount=2 with framesInFlight=2 allocates 8 descriptor sets + 52 storage
		// buffer descriptors, so p=512 split across two GPUs (256 chunks per GPU)
		// needs ~2K sets and ~13K storage buffers per GPU. Headroom on top covers
		// rendering / video / GUI consumers.
		// Increased to 65536 sets / 262144 buffers to accommodate searches with full
		// FECW neighborhoods (7 subsets × multiple maxCS × wrap = 70+ configs); the
		// previous 16K cap overflowed at FECW K=2 ticks=100 CS=2-6.
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			{vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(framesInFlight * 10)},
			{vk::DescriptorType::eStorageBuffer, 262144},
			{vk::DescriptorType::eStorageImage, static_cast<uint32_t>(framesInFlight * 32)}};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo = vk::DescriptorPoolCreateInfo(
			vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, // flags
			65536,												  // maxSets
			static_cast<uint32_t>(poolSizes.size()),			  // poolSizeCount
			poolSizes.data()									  // pPoolSizes
		);

		try
		{
			_descriptorPool = _device.createDescriptorPool(descriptorPoolInfo);
		}
		catch (vk::SystemError error)
		{
			LOG_ERROR("Failed To Create Vulkan Descriptor Pool");
		}
	}

	void DeviceContext::CreateQueryPool()
	{
		vk::QueryPoolCreateInfo queryPoolCreateInfo = vk::QueryPoolCreateInfo(
			vk::QueryPoolCreateFlags(), // flags
			vk::QueryType::eTimestamp,	// queryType
			32							// queryCount
										// pipelineStatistics
										// pNext
		);

		_queryPool = _device.createQueryPool(queryPoolCreateInfo);

		vk::CommandBuffer commandBuffer = BeginSingleTimeCommands(_graphicsCommandPool);
		commandBuffer.resetQueryPool(_queryPool, 0, 32);
		EndSingleTimeCommands(std::move(commandBuffer), _graphicsCommandPool, _graphicsQueue);
	}

	vk::CommandBuffer DeviceContext::BeginSingleTimeCommands(vk::CommandPool commandPool)
	{
		LOG_DEBUG("Begin Vulkan Single Time Command");

		vk::CommandBufferAllocateInfo commandBufferAllocateInfo = vk::CommandBufferAllocateInfo(
			commandPool,					  // commandPool
			vk::CommandBufferLevel::ePrimary, // level
			1								  // commandBufferCount
		);

		vk::CommandBuffer commandBuffer = _device.allocateCommandBuffers(commandBufferAllocateInfo)[0];

		vk::CommandBufferBeginInfo commandBufferBeginInfo = vk::CommandBufferBeginInfo(
			vk::CommandBufferUsageFlagBits::eOneTimeSubmit // flags
		);

		commandBuffer.begin(commandBufferBeginInfo);

		return commandBuffer;
	}

	void DeviceContext::EndSingleTimeCommands(vk::CommandBuffer commandBuffer, vk::CommandPool commandPool, vk::Queue queue)
	{
		commandBuffer.end();

		vk::SubmitInfo submitInfo = vk::SubmitInfo(
			0,				// waitSemaphoreCount
			{},				// pWaitSemaphores
			{},				// pWaitDstStageMask
			1,				// commandBufferCount
			&commandBuffer, // pCommandBuffers
			0,				// signalSemaphoreCount
			{}				// pSignalSemaphores
		);

		queue.submit(submitInfo);
		queue.waitIdle();

		_device.freeCommandBuffers(commandPool, 1, &commandBuffer);
		LOG_DEBUG("Finished Vulkan Single Time Command");
	}

	void DeviceContext::CopyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size, vk::CommandPool commandPool, vk::Queue queue)
	{
		vk::CommandBuffer commandBuffer = BeginSingleTimeCommands(commandPool);

		vk::BufferCopy bufferCopy = vk::BufferCopy(
			0,	 // srcOffset
			0,	 // dstOffset
			size // size
		);

		commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &bufferCopy);

		EndSingleTimeCommands(commandBuffer, commandPool, queue);
	}

} // namespace Cave
