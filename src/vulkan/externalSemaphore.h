#pragma once

// Binary external semaphore wrapper for Vulkan-OpenGL interop.
//
// Each frame the LookingGlass mode hands a binary semaphore back and forth
// between Vulkan and OpenGL: Vulkan signals after rendering the quilt, GL
// waits on it before drawing, GL signals after Bridge consumes the texture,
// Vulkan waits before reusing the quilt next frame.
//
// GL has no notion of timeline semaphores, hence binary. The Win32 export
// HANDLE is owned by this wrapper until imported into GL via
// glImportSemaphoreWin32HandleEXT — at which point ownership transfers to GL
// (we set a flag and skip CloseHandle in the destructor).
//
// Windows-only as of M2; gated by LOOKING_GLASS_BUILD in the .cpp.

#include <vulkan/vulkan.hpp>

namespace Cave
{
	class DeviceContext;

	class ExternalBinarySemaphore
	{
	public:
		// Allocates a vk::Semaphore with VkExportSemaphoreCreateInfo + handleType
		// = OPAQUE_WIN32, then immediately exports it to a Win32 HANDLE via
		// vkGetSemaphoreWin32HandleKHR. Throws on failure.
		explicit ExternalBinarySemaphore(DeviceContext& deviceContext);
		~ExternalBinarySemaphore();

		ExternalBinarySemaphore(const ExternalBinarySemaphore&) = delete;
		ExternalBinarySemaphore& operator=(const ExternalBinarySemaphore&) = delete;

		// The Vulkan-side handle, used for vkQueueSubmit signal/wait chains.
		vk::Semaphore GetSemaphore() const { return _semaphore; }

		// The Win32 HANDLE for export to OpenGL via glImportSemaphoreWin32HandleEXT.
		// Returned as void* so consumers don't need to include windows.h. The HANDLE
		// is owned by this wrapper until MarkHandleTransferredToGl() is called.
		void* GetWin32Handle() const { return _win32Handle; }

		// Marks the HANDLE as transferred to OpenGL ownership. After this call the
		// destructor will NOT call CloseHandle (GL keeps a reference). The vk::Semaphore
		// itself is still owned by Vulkan and is destroyed in the destructor.
		void MarkHandleTransferredToGl() { _handleTransferredToGl = true; }

	private:
		DeviceContext& _deviceContext;
		vk::Semaphore _semaphore;
		void* _win32Handle = nullptr;
		bool _handleTransferredToGl = false;
	};

} // namespace Cave
