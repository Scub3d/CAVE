#pragma once

#include <vulkan/vulkan.hpp>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <GLFW/glfw3.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <map>
#include <optional>
#include <set>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <array>
#include <memory>
#include <unordered_map>
#include <string>

#include "../common/logger.h"
#include "vk_mem_alloc.h"

#include "../common/structs.h"
#include "../common/utils.h"

#include "deviceContext.h"

#define VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

// Windows.h macros that collide with method names
#ifdef CreateWindow
#undef CreateWindow
#endif
#ifdef CreateSemaphore
#undef CreateSemaphore
#endif

namespace Cave
{

	class VulkanInstance
	{
	public:
		VulkanInstance() = default;
		~VulkanInstance() = default;

		VulkanInstance(const VulkanInstance &) = delete;
		VulkanInstance &operator=(const VulkanInstance &) = delete;

	private: // Variables
		bool _headless = false;
		int _forcedGpuIndex = -1; // -1 = auto-select, 0+ = force specific GPU
		bool _lookingGlassRequested = false; // enables Vulkan-OpenGL interop ext bundle
		const char *_applicationName = "3D Cellular Automata Vulkan Engine";

		std::vector<const char *> _activeInstanceLayers{};
		std::vector<const char *> _activeInstanceExtensions{};

		vk::SurfaceKHR _surface;

		vk::Instance _instance;
		vk::DebugUtilsMessengerEXT _debugMessenger;
		vk::DispatchLoaderDynamic _dldi;

		vk::Extent2D _extent;

		uint32_t _framesInFlight = 2; // Default to 2

#ifdef NDEBUG
		bool _debugMode = false;
#else
		bool _debugMode = true;
#endif

		GLFWwindow *_window;
		int _windowWidth = 1920;
		int _windowHeight = 1080;
		std::string _windowName = "CAVE";
		bool _framebufferResized = false;

		const std::vector<const char *> _validationLayers = {"VK_LAYER_KHRONOS_validation"};

		vk::PhysicalDevice _selectedPhysicalDevice;

	public: // Variables
#ifdef NDEBUG
		const bool EnableValidationLayers = false;
#else
		const bool EnableValidationLayers = true;
#endif

	private: // Methods
		void CreateWindow();
		void CreateWindowSurface();
		static void FramebufferResizeCallback(GLFWwindow *window, int width, int height);
		void CreateInstance();
		void CreateDebugMessenger();
		void PickPhysicalDevice();
		std::vector<const char *> GetRequiredExtensions();
		bool CheckValidationLayerSupport();
		bool HasGLFWRequiredInstanceExtensions();

	public: // Methods
		void Initialize();

		std::unique_ptr<DeviceContext> CreateDeviceContext();
		std::unique_ptr<DeviceContext> CreateDeviceContext(vk::PhysicalDevice physicalDevice);

		vk::PhysicalDevice GetSelectedPhysicalDevice() const { return _selectedPhysicalDevice; }
		std::vector<vk::PhysicalDevice> GetAvailablePhysicalDevices();

		vk::Instance GetInstance() { return _instance; }

		void SetFramesInFlight(uint32_t framesInFlight) { _framesInFlight = framesInFlight; }
		uint32_t GetFramesInFlight() { return _framesInFlight; }

		void SetHeadless(bool headless) { _headless = headless; }
		bool IsHeadless() const { return _headless; }
		void SetForcedGpuIndex(int gpuIndex) { _forcedGpuIndex = gpuIndex; }

		// Enable Vulkan-OpenGL interop instance/device extensions for the Looking Glass
		// mode. Must be called before Initialize(). When true, GetRequiredExtensions()
		// also requests VK_KHR_external_memory_capabilities and
		// VK_KHR_external_semaphore_capabilities; CreateDeviceContext propagates the
		// flag so the device side enables the matching device extensions + multiview.
		void SetLookingGlassRequested(bool requested) { _lookingGlassRequested = requested; }
		bool IsLookingGlassRequested() const { return _lookingGlassRequested; }

		// ---- Instance-level accessors ----

		vk::SurfaceKHR GetSurface() { return _surface; }
		vk::DispatchLoaderDynamic &GetDispatchLoaderDynamic() { return _dldi; }

		std::string GetWindowName() const { return _windowName; }
		GLFWwindow *GetVulkanGLFWWindow() const { return _window; }
		VkExtent2D GetWindowExtent() { return {static_cast<uint32_t>(_windowWidth), static_cast<uint32_t>(_windowHeight)}; }
		bool ShouldCloseVulkanWindow() { return glfwWindowShouldClose(_window); }
		bool WasWindowResized() { return _framebufferResized; }
		void ResetWindowResizedFlag() { _framebufferResized = false; }
		const char *GetApplicationName() const { return _windowName.c_str(); }
	};

} // namespace Cave
