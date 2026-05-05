// VK_USE_PLATFORM_WIN32_KHR must be defined BEFORE the vulkan.hpp include
// (transitively pulled in via externalSemaphore.h → vulkan/vulkan.hpp).
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define VK_USE_PLATFORM_WIN32_KHR
#ifdef CreateSemaphore
#undef CreateSemaphore
#endif
#endif

#include "externalSemaphore.h"
#include "deviceContext.h"
#include "../common/logger.h"

namespace Cave
{
	ExternalBinarySemaphore::ExternalBinarySemaphore(DeviceContext& deviceContext)
		: _deviceContext{deviceContext}
	{
#ifdef _WIN32
		LOG_DEBUG("ExternalBinarySemaphore: ctor entry");
		// Step 1: declare we want this semaphore to be exportable as a Win32 HANDLE.
		vk::ExportSemaphoreCreateInfo exportInfo{};
		exportInfo.handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;

		vk::SemaphoreCreateInfo createInfo{};
		createInfo.pNext = &exportInfo;

		LOG_DEBUG("ExternalBinarySemaphore: createSemaphore");
		_semaphore = _deviceContext.GetDevice().createSemaphore(createInfo);
		LOG_DEBUG("ExternalBinarySemaphore: semaphore handle = {:x}",
			reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(_semaphore)));

		// Step 2: pull the Win32 HANDLE for the just-created semaphore.
		// We use the C API directly (vkGetSemaphoreWin32HandleKHR loaded via
		// vkGetDeviceProcAddr) instead of the vk-hpp wrapper to avoid any
		// dispatch-loader templated-overload resolution surprises.
		auto& dldi = _deviceContext.GetDispatchLoaderDynamic();
		auto pfn = dldi.vkGetSemaphoreWin32HandleKHR;
		if (!pfn)
		{
			LOG_ERROR("ExternalBinarySemaphore: vkGetSemaphoreWin32HandleKHR not loaded in dldi");
			throw std::runtime_error("vkGetSemaphoreWin32HandleKHR not loaded");
		}
		LOG_DEBUG("ExternalBinarySemaphore: pfn loaded at {:x}", reinterpret_cast<uint64_t>(pfn));

		VkSemaphoreGetWin32HandleInfoKHR getInfo{};
		getInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
		getInfo.semaphore = static_cast<VkSemaphore>(_semaphore);
		getInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

		HANDLE hnd = nullptr;
		VkResult res = pfn(static_cast<VkDevice>(_deviceContext.GetDevice()), &getInfo, &hnd);
		LOG_DEBUG("ExternalBinarySemaphore: vkGetSemaphoreWin32HandleKHR returned res={}, handle={:x}",
			static_cast<int>(res), reinterpret_cast<uint64_t>(hnd));

		if (res != VK_SUCCESS || !hnd || hnd == INVALID_HANDLE_VALUE)
		{
			LOG_ERROR("ExternalBinarySemaphore: vkGetSemaphoreWin32HandleKHR failed (res={})", static_cast<int>(res));
			throw std::runtime_error("Failed to export semaphore Win32 handle");
		}
		_win32Handle = hnd;
		LOG_DEBUG("ExternalBinarySemaphore: created semaphore + exported Win32 HANDLE");
#else
		throw std::runtime_error("ExternalBinarySemaphore is Windows-only");
#endif
	}

	ExternalBinarySemaphore::~ExternalBinarySemaphore()
	{
#ifdef _WIN32
		if (_win32Handle && !_handleTransferredToGl)
		{
			CloseHandle(static_cast<HANDLE>(_win32Handle));
		}
#endif
		if (_semaphore)
		{
			_deviceContext.GetDevice().destroySemaphore(_semaphore);
		}
	}
} // namespace Cave
