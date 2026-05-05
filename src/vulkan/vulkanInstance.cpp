#include "vulkanInstance.h"
#include <common/logger.h>

namespace Cave
{

	void VulkanInstance::Initialize()
	{
		CreateWindow();
		CreateInstance();
		CreateDebugMessenger();
		CreateWindowSurface();
		PickPhysicalDevice();
	}

	std::unique_ptr<DeviceContext> VulkanInstance::CreateDeviceContext()
	{
		return CreateDeviceContext(_selectedPhysicalDevice);
	}

	std::unique_ptr<DeviceContext> VulkanInstance::CreateDeviceContext(vk::PhysicalDevice physicalDevice)
	{
		auto deviceContext = std::make_unique<DeviceContext>();
		deviceContext->SetLookingGlassRequested(_lookingGlassRequested);
		deviceContext->Initialize(_instance, physicalDevice, _surface, _debugMode, &_dldi);
		return deviceContext;
	}

	std::vector<vk::PhysicalDevice> VulkanInstance::GetAvailablePhysicalDevices()
	{
		return _instance.enumeratePhysicalDevices();
	}

	void VulkanInstance::CreateWindow()
	{
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
		if (_headless)
			glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

		LOG_DEBUG("Creating GLFW Vulkan Window{}", _headless ? " (hidden, headless mode)" : "");
		if (_window = glfwCreateWindow(_windowWidth, _windowHeight, _windowName.c_str(), nullptr, nullptr))
		{
			LOG_DEBUG("Successfully created a glfw vulkan window with:\n - Title: {} \n - Width: {} \n - Height: {}", _windowName.c_str(), _windowWidth, _windowHeight);
		}
		else
		{
			LOG_ERROR("GLFW vulkan window creation failed");
		}

		glfwSetWindowUserPointer(_window, this);
		glfwSetFramebufferSizeCallback(_window, FramebufferResizeCallback);
	}

	void VulkanInstance::CreateWindowSurface()
	{
		VkSurfaceKHR temporarySurface;

		if (glfwCreateWindowSurface(_instance, _window, nullptr, &temporarySurface) != VK_SUCCESS)
		{
			LOG_ERROR("GLFW Window Surface creation failed");
			throw std::runtime_error("Failed to Create GLFW Window Surface");
		}
		else
		{
			LOG_DEBUG("Successfully created a glfw vulkan surface with:\n - Title: {} \n - Width: {} \n - Height: {}", _windowName, _windowWidth, _windowHeight);
			_surface = temporarySurface;
		}
	}

	void VulkanInstance::FramebufferResizeCallback(GLFWwindow *window, int width, int height)
	{
		auto context = reinterpret_cast<VulkanInstance *>(glfwGetWindowUserPointer(window));
		context->_framebufferResized = true;
		context->_windowWidth = width;
		context->_windowHeight = height;
	}

	void VulkanInstance::CreateInstance()
	{
		LOG_DEBUG("Creating Vulkan Instance");

		uint32_t version;
		vkEnumerateInstanceVersion(&version);

		LOG_DEBUG("System can support vulkan variant: {}, Major: {}, Minor: {}, Patch: {}",
				  VK_API_VERSION_VARIANT(version), VK_API_VERSION_MAJOR(version),
				  VK_API_VERSION_MINOR(version), VK_API_VERSION_PATCH(version));

		vk::ApplicationInfo applicationInfo = vk::ApplicationInfo(
			_applicationName, // pApplicationName
			0,				  // applicationVersion
			"CAVE",			  // pEngineName
			0,				  // engineVersion
			version			  // apiVersion
		);

		std::vector<const char *> requiredExtensions;

		requiredExtensions = GetRequiredExtensions();

		std::vector<const char *> layers;

		if (_debugMode)
		{
			for (const char *validationLayer : _validationLayers)
			{
				layers.push_back(validationLayer);
			}
		}

		if (!HasGLFWRequiredInstanceExtensions() || !CheckValidationLayerSupport())
		{
			_instance = nullptr;
			LOG_ERROR("Unable to create instance. Issue with supported extensions/layers");
			return;
		}

		vk::InstanceCreateInfo createInfo = vk::InstanceCreateInfo(
			vk::InstanceCreateFlags(),						  // flags
			&applicationInfo,								  // pApplicationInfo
			static_cast<uint32_t>(layers.size()),			  // enabledLayerCount
			layers.data(),									  // ppEnabledLayerNames
			static_cast<uint32_t>(requiredExtensions.size()), // enabledExtensionCount
			requiredExtensions.data()						  // ppEnabledExtensionNames
		);

		try
		{
			_instance = vk::createInstance(createInfo, nullptr);
			LOG_DEBUG("Created Vulkan Instance");
		}
		catch (vk::SystemError error)
		{
			_instance = nullptr;
			LOG_ERROR("Failed To Create Vulkan Instance: {}", error.what());
		}
	}

	void VulkanInstance::CreateDebugMessenger()
	{
		_dldi = vk::DispatchLoaderDynamic(_instance, vkGetInstanceProcAddr);
		if (!EnableValidationLayers)
			return;
		_debugMessenger = LogHandler::CreateVulkanDebugMessenger(_instance, _dldi);
	}

	void VulkanInstance::PickPhysicalDevice()
	{
		LOG_DEBUG("Picking physical device");

		std::vector<vk::PhysicalDevice> availableDevices = _instance.enumeratePhysicalDevices();
		LOG_INFO("Found {} physical device(s):", availableDevices.size());

		// Temporary DeviceContext used only for device suitability rating.
		// RateDeviceSuitability and CheckDeviceExtensionSupport only read the
		// const _deviceExtensions member and the passed physical device properties.
		DeviceContext deviceRater;

		for (uint32_t deviceIndex = 0; deviceIndex < availableDevices.size(); deviceIndex++)
		{
			vk::PhysicalDeviceProperties deviceProperties = availableDevices[deviceIndex].getProperties();
			int score = deviceRater.RateDeviceSuitability(availableDevices[deviceIndex]);
			LOG_INFO("  [{}] {} — score: {}, deviceID: 0x{:04X}, vendorID: 0x{:04X}",
				deviceIndex,
				std::string(deviceProperties.deviceName.data()),
				score,
				deviceProperties.deviceID,
				deviceProperties.vendorID);
		}

		if (_forcedGpuIndex >= 0)
		{
			if (_forcedGpuIndex < static_cast<int>(availableDevices.size()))
			{
				_selectedPhysicalDevice = availableDevices[_forcedGpuIndex];
				LOG_INFO("Forced GPU selection: index {}", _forcedGpuIndex);
			}
			else
			{
				LOG_ERROR("Forced GPU index {} is out of range (only {} device(s) available)", _forcedGpuIndex, availableDevices.size());
				throw std::runtime_error("Invalid forced GPU index");
			}
		}
		else
		{
			std::multimap<int, std::pair<uint32_t, VkPhysicalDevice>> deviceCandidates;

			for (uint32_t deviceIndex = 0; deviceIndex < availableDevices.size(); deviceIndex++)
			{
				int score = deviceRater.RateDeviceSuitability(availableDevices[deviceIndex]);
				deviceCandidates.insert(std::make_pair(score, std::make_pair(deviceIndex, static_cast<VkPhysicalDevice>(availableDevices[deviceIndex]))));
			}

			if (deviceCandidates.rbegin()->first > 0)
			{
				_selectedPhysicalDevice = deviceCandidates.rbegin()->second.second;
				LOG_INFO("Auto-selected GPU index: {}", deviceCandidates.rbegin()->second.first);
			}
			else
			{
				LOG_ERROR("Failed to find suitable device");
				throw std::runtime_error("No suitable GPU found");
			}
		}

		vk::PhysicalDeviceProperties selectedDeviceProperties = _selectedPhysicalDevice.getProperties();
		LOG_INFO("Selected GPU: {} (timestamp period: {} ns/tick)", std::string(selectedDeviceProperties.deviceName.data()), selectedDeviceProperties.limits.timestampPeriod);
	}

	std::vector<const char *> VulkanInstance::GetRequiredExtensions()
	{
		uint32_t glfwExtensionCount;
		const char **glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (_debugMode)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		// Looking Glass needs the capability extensions to query the physical device's
		// support for exporting Win32 handles for memory and semaphores. Enabled only
		// when the user actually selected LG mode so we don't refuse devices that lack
		// these capabilities for unrelated runs.
		if (_lookingGlassRequested)
		{
			extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
			extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
		}

		if (_debugMode)
		{
			LOG_DEBUG("Required extensions:");

			for (const char *extensionName : extensions)
			{
				LOG_DEBUG("\t\"{}\"", extensionName);
			}
		}

		return extensions;
	}

	bool VulkanInstance::CheckValidationLayerSupport()
	{
		std::vector<const char *> layers;

		if (_debugMode)
		{
			for (const char *validationLayer : _validationLayers)
			{
				layers.push_back(validationLayer);
			}
		}

		std::vector<vk::LayerProperties> supportedLayers = vk::enumerateInstanceLayerProperties();

		LOG_DEBUG("Device can support the following layers:");

		if (_debugMode)
		{
			for (vk::LayerProperties supportedLayer : supportedLayers)
			{
				LOG_DEBUG("\t\"{}\"", static_cast<const char*>(supportedLayer.layerName));
			}
		}

		for (const char *layer : layers)
		{
			bool found = false;

			for (vk::LayerProperties supportedLayer : supportedLayers)
			{
				if (strcmp(layer, supportedLayer.layerName) == 0)
				{
					found = true;
					LOG_DEBUG("Layer \"{}\" is supported", layer);
				}
			}

			if (!found)
			{
				LOG_DEBUG("Layer \"{}\" is not supported", layer);
				return false;
			}
		}

		return true;
	}

	bool VulkanInstance::HasGLFWRequiredInstanceExtensions()
	{
		std::vector<vk::ExtensionProperties> availableExtensionsProperties = vk::enumerateInstanceExtensionProperties();
		std::vector<const char *> requiredExtensions = GetRequiredExtensions();

		LOG_DEBUG("Device can support the following extensions:");

		if (_debugMode)
		{
			for (vk::ExtensionProperties availableExtensionProperties : availableExtensionsProperties)
			{
				LOG_DEBUG("\t{}", static_cast<const char*>(availableExtensionProperties.extensionName));
			}
		}

		for (const char *requiredExtension : requiredExtensions)
		{
			bool found = false;

			for (vk::ExtensionProperties availableExtensionProperties : availableExtensionsProperties)
			{
				if (strcmp(requiredExtension, availableExtensionProperties.extensionName) == 0)
				{
					LOG_DEBUG("Extension \"{}\" is supported", requiredExtension);
					found = true;
				}
			}

			if (!found)
			{
				LOG_DEBUG("Extension \"{}\" is not supported", requiredExtension);
				return false;
			}
		}

		return true;
	}

} // namespace Cave
