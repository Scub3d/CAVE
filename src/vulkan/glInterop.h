#pragma once

// Vulkan-OpenGL interop helpers for the Looking Glass Portrait output path.
//
// The Bridge SDK only exposes an OpenGL submission entry point, so Cave renders
// the 48-view "quilt" texture in Vulkan and exports it via Win32 external memory
// handles. This module owns:
//   - A hidden GLFW window that hosts the OpenGL 4.5 context (Bridge will create
//     its own visible window on the holographic display separately).
//   - A minimal hand-typed loader for the ~10 GL symbols we need, no GLAD.
//   - Helpers to import a Vulkan-allocated image and binary external semaphores
//     into GL by their Win32 HANDLEs.
//
// Lifetime / threading rules:
//   - GLFW must be initialized before constructing GlInteropContext (it is —
//     VulkanInstance::CreateWindow does that). The hidden GL window shares the
//     process-global GLFW state.
//   - GL contexts are thread-affine. MakeCurrent() / ReleaseCurrent() bracket
//     any GL call. The smoke tests run on the main thread; LookingGlassMode
//     will MakeCurrent the GL context inside its per-frame submit.
//
// Windows-only as of M1. The CMake LOOKING_GLASS_BUILD flag gates the entire
// module compile; on non-Windows platforms it stays out of the build.

#include <cstdint>
#include <string>

struct GLFWwindow;

namespace Cave
{

class GlInteropContext
{
public:
	GlInteropContext() = default;
	~GlInteropContext();

	GlInteropContext(const GlInteropContext&) = delete;
	GlInteropContext& operator=(const GlInteropContext&) = delete;

	// Creates the hidden GLFW window with GLFW_CLIENT_API = GLFW_OPENGL_API,
	// requests a 4.5 core profile, makes the context current on the calling
	// thread, and loads the GL function pointers we need. Returns false (and
	// logs a clear error) if GL 4.5 / required extensions / function loading
	// fails on this driver. Idempotent — calling twice is a no-op.
	bool Initialize();

	// Releases the GL context and destroys the GLFW window. Safe to call even
	// if Initialize() was never called or failed.
	void Shutdown();

	// Bind / unbind the GL context on the calling thread. GL calls are only
	// valid between MakeCurrent and ReleaseCurrent. Re-entrant on the same
	// thread (subsequent MakeCurrent calls are cheap; nesting requires the
	// caller to balance Release calls).
	void MakeCurrent();
	void ReleaseCurrent();

	// Returns the version / vendor / renderer reported by GL after Initialize.
	// Empty before Initialize succeeds.
	const std::string& GlVersion() const { return _glVersion; }
	const std::string& GlVendor()  const { return _glVendor; }
	const std::string& GlRenderer() const { return _glRenderer; }

	bool IsInitialized() const { return _initialized; }

	// Imports a Vulkan-allocated image's Win32 HANDLE as a GL texture. The HANDLE
	// must be obtained via vkGetMemoryWin32HandleKHR; memorySize is the byte size
	// of the underlying VkDeviceMemory (NOT just width*height*bpp — the driver
	// may add tile padding). glInternalFormat is e.g. GL_RGBA8 (use the constant
	// GlInteropConstants::RGBA8 below to avoid pulling in <gl.h>). Returns 0 on
	// failure; the texture is otherwise owned by GL.
	//
	// Layout-handshake note: the imported texture must be transitioned to GL's
	// expected layout via the layout array passed to glWaitSemaphoreEXT. This
	// module assumes VK_IMAGE_LAYOUT_GENERAL ↔ GL_LAYOUT_GENERAL_EXT throughout
	// (sidesteps the dominant interop bug class).
	uint32_t ImportImageHandleAsTexture(void* win32Handle, uint64_t memorySize,
	                                    uint32_t width, uint32_t height,
	                                    uint32_t glInternalFormat);

	// Imports a Vulkan-exported binary semaphore HANDLE as a GL semaphore.
	// Returns the GL semaphore ID, or 0 on failure.
	uint32_t ImportSemaphoreHandle(void* win32Handle);

	// Per-frame handshake helpers. Both treat the texture as residing in
	// VK_IMAGE_LAYOUT_GENERAL on the Vulkan side.
	void WaitSemaphoreOnTexture(uint32_t glSemaphore, uint32_t glTexture);
	void SignalSemaphoreOnTexture(uint32_t glSemaphore, uint32_t glTexture);

	// Cleanup helpers (avoid leaking GL objects when the mode tears down).
	void DeleteTexture(uint32_t glTexture);
	void DeleteSemaphore(uint32_t glSemaphore);

	// Read the entire 2D RGBA8 texture's pixels back to host memory. Used by
	// the round-trip smoke test (M2) and the save-quilt PNG path (M3). Output
	// buffer must be at least width*height*4 bytes.
	void ReadTextureRGBA8(uint32_t glTexture, uint32_t width, uint32_t height, void* outPixels);

	// Synchronous flush — call before yielding the context.
	void Flush();

private:
	GLFWwindow* _window = nullptr;
	bool _initialized = false;
	bool _hasGlfwInit = false;

	std::string _glVersion;
	std::string _glVendor;
	std::string _glRenderer;

	bool LoadGlFunctions();
	bool VerifyRequiredExtensions();
};

// Top-level smoke-test driver. Called from main / Engine when --lg-test=NAME is
// passed on the CLI. Returns 0 on success, non-zero on failure (matching
// process exit-code convention). NAME is one of: "context", "roundtrip",
// "save-quilt". Unknown names log a warning and return 0.
//
// "context" needs no Vulkan setup — just verifies GL coexistence.
// "roundtrip" needs an initialized DeviceContext (passed via the second arg).
// "save-quilt" needs a full simulation + render pipeline (M3 wires that path).
class DeviceContext;
int RunLookingGlassSmokeTest(const std::string& testName, DeviceContext* deviceContext = nullptr);

} // namespace Cave
