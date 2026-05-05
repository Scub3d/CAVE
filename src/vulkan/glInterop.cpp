#include "glInterop.h"
#include "../common/logger.h"
#include "deviceContext.h"
#include "image.h"
#include "externalSemaphore.h"
#include "systems/quiltRenderSystem.h"

#include <cstdint>
#include <cstring>
#include <vector>

// stb_image_write — single-header PNG writer. Implementation is provided by
// frameCaptureUtility.cpp (the only TU that defines STB_IMAGE_WRITE_IMPLEMENTATION).
#include <stb_image_write.h>

// Vulkan-Hpp is needed for the round-trip smoke test (cmd buffer clear/submit).
// We include it WITHOUT pulling windows.h to avoid the gl.h ↔ constexpr
// collision; the Win32-specific helpers live in image.cpp / externalSemaphore.cpp
// where windows.h is properly isolated.
#include <vulkan/vulkan.hpp>

// Deliberately NOT including <windows.h> here — it pulls in <gl/gl.h> which
// #defines GL_VERSION / GL_VENDOR / GL_EXTENSIONS / etc. as macros and conflicts
// with the constexpr declarations below. Win32 HANDLE is exchanged as void* at
// the API boundary (cheap, type-safe enough for our purposes); concrete Win32
// calls happen only inside externalSemaphore.cpp / image.cpp where we control
// the include order.

// GLFW pulls in <GL/gl.h> by default on Windows, which collides with our
// constexpr GL constants below. GLFW_INCLUDE_NONE suppresses that.
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cstring>
#include <vector>

// --- Minimal hand-typed GL function pointer loader --------------------------
//
// We only need a handful of GL symbols for Vulkan-OpenGL interop. Avoiding GLAD
// keeps the build dependency-free; the trade-off is hand-typing the function
// pointer typedefs and constants. All declarations follow the Khronos GL spec
// (https://registry.khronos.org/OpenGL/extensions/EXT/EXT_external_objects.txt
// and EXT_external_objects_win32.txt).

#ifndef APIENTRY
#define APIENTRY __stdcall
#endif

using GLenum   = unsigned int;
using GLuint   = unsigned int;
using GLint    = int;
using GLsizei  = int;
using GLubyte  = unsigned char;
using GLuint64 = unsigned long long;
using GLboolean = unsigned char;

// GL constants (subset — only the ones we use in interop)
constexpr GLenum GL_NUM_EXTENSIONS                        = 0x821D;
constexpr GLenum GL_EXTENSIONS                            = 0x1F03;
constexpr GLenum GL_VERSION                               = 0x1F02;
constexpr GLenum GL_VENDOR                                = 0x1F00;
constexpr GLenum GL_RENDERER                              = 0x1F01;
constexpr GLenum GL_NO_ERROR                              = 0;
constexpr GLenum GL_HANDLE_TYPE_OPAQUE_WIN32_EXT          = 0x9587;
constexpr GLenum GL_LAYOUT_GENERAL_EXT                    = 0x958D;
constexpr GLenum GL_LAYOUT_COLOR_ATTACHMENT_EXT           = 0x958E;
constexpr GLenum GL_LAYOUT_DEPTH_STENCIL_ATTACHMENT_EXT   = 0x958F;
constexpr GLenum GL_LAYOUT_DEPTH_STENCIL_READ_ONLY_EXT    = 0x9590;
constexpr GLenum GL_LAYOUT_SHADER_READ_ONLY_EXT           = 0x9591;
constexpr GLenum GL_LAYOUT_TRANSFER_SRC_EXT               = 0x9592;
constexpr GLenum GL_LAYOUT_TRANSFER_DST_EXT               = 0x9593;
constexpr GLenum GL_RGBA8                                 = 0x8058;
constexpr GLenum GL_RGBA                                  = 0x1908;
constexpr GLenum GL_UNSIGNED_BYTE                         = 0x1401;

// Function pointer typedefs — declared here so we don't need glext.h.
using PFNGLGETSTRINGPROC                  = const GLubyte* (APIENTRY*)(GLenum);
using PFNGLGETSTRINGIPROC                 = const GLubyte* (APIENTRY*)(GLenum, GLuint);
using PFNGLGETINTEGERVPROC                = void (APIENTRY*)(GLenum, GLint*);
using PFNGLGETERRORPROC                   = GLenum (APIENTRY*)(void);
using PFNGLCREATEMEMORYOBJECTSEXTPROC     = void (APIENTRY*)(GLsizei, GLuint*);
using PFNGLDELETEMEMORYOBJECTSEXTPROC     = void (APIENTRY*)(GLsizei, const GLuint*);
using PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC = void (APIENTRY*)(GLuint, GLuint64, GLenum, void*);
using PFNGLCREATETEXTURESPROC             = void (APIENTRY*)(GLenum, GLsizei, GLuint*);
using PFNGLTEXTURESTORAGEMEM2DEXTPROC     = void (APIENTRY*)(GLuint, GLsizei, GLenum, GLsizei, GLsizei, GLuint, GLuint64);
using PFNGLDELETETEXTURESPROC             = void (APIENTRY*)(GLsizei, const GLuint*);
using PFNGLGETTEXTUREIMAGEPROC            = void (APIENTRY*)(GLuint, GLint, GLenum, GLenum, GLsizei, void*);
using PFNGLTEXTUREPARAMETERIPROC          = void (APIENTRY*)(GLuint, GLenum, GLint);
using PFNGLGENSEMAPHORESEXTPROC           = void (APIENTRY*)(GLsizei, GLuint*);
using PFNGLDELETESEMAPHORESEXTPROC        = void (APIENTRY*)(GLsizei, const GLuint*);
using PFNGLIMPORTSEMAPHOREWIN32HANDLEEXTPROC = void (APIENTRY*)(GLuint, GLenum, void*);
using PFNGLWAITSEMAPHOREEXTPROC           = void (APIENTRY*)(GLuint, GLuint, const GLuint*, GLuint, const GLuint*, const GLenum*);
using PFNGLSIGNALSEMAPHOREEXTPROC         = void (APIENTRY*)(GLuint, GLuint, const GLuint*, GLuint, const GLuint*, const GLenum*);
using PFNGLFLUSHPROC                      = void (APIENTRY*)(void);

// Module-local function pointers. Loaded once in LoadGlFunctions.
namespace
{
	PFNGLGETSTRINGPROC                  glGetString_fp = nullptr;
	PFNGLGETSTRINGIPROC                 glGetStringi_fp = nullptr;
	PFNGLGETINTEGERVPROC                glGetIntegerv_fp = nullptr;
	PFNGLGETERRORPROC                   glGetError_fp = nullptr;
	PFNGLCREATEMEMORYOBJECTSEXTPROC     glCreateMemoryObjectsEXT_fp = nullptr;
	PFNGLDELETEMEMORYOBJECTSEXTPROC     glDeleteMemoryObjectsEXT_fp = nullptr;
	PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC glImportMemoryWin32HandleEXT_fp = nullptr;
	PFNGLCREATETEXTURESPROC             glCreateTextures_fp = nullptr;
	PFNGLTEXTURESTORAGEMEM2DEXTPROC     glTextureStorageMem2DEXT_fp = nullptr;
	PFNGLDELETETEXTURESPROC             glDeleteTextures_fp = nullptr;
	PFNGLGETTEXTUREIMAGEPROC            glGetTextureImage_fp = nullptr;
	PFNGLTEXTUREPARAMETERIPROC          glTextureParameteri_fp = nullptr;
	PFNGLGENSEMAPHORESEXTPROC           glGenSemaphoresEXT_fp = nullptr;
	PFNGLDELETESEMAPHORESEXTPROC        glDeleteSemaphoresEXT_fp = nullptr;
	PFNGLIMPORTSEMAPHOREWIN32HANDLEEXTPROC glImportSemaphoreWin32HandleEXT_fp = nullptr;
	PFNGLWAITSEMAPHOREEXTPROC           glWaitSemaphoreEXT_fp = nullptr;
	PFNGLSIGNALSEMAPHOREEXTPROC         glSignalSemaphoreEXT_fp = nullptr;
	PFNGLFLUSHPROC                      glFlush_fp = nullptr;

	// Fallback for the 1.1-era functions that wglGetProcAddress refuses to
	// load — those must come from opengl32.dll directly. glfwGetProcAddress
	// handles this internally on Windows but we double-check by linking
	// against opengl32.lib (CMake target_link_libraries opengl32).
	template <typename FnPtr>
	bool LoadGl(FnPtr& slot, const char* name)
	{
		slot = reinterpret_cast<FnPtr>(glfwGetProcAddress(name));
		if (!slot)
		{
			LOG_ERROR("GlInterop: failed to load GL function '{}'", name);
			return false;
		}
		return true;
	}
}

namespace Cave
{

GlInteropContext::~GlInteropContext()
{
	Shutdown();
}

bool GlInteropContext::Initialize()
{
	if (_initialized) return true;

	// GLFW is already initialized by VulkanInstance::CreateWindow. We just
	// create a second hidden window with the OpenGL client API. Both windows
	// coexist; the Vulkan window has GLFW_NO_API, the GL window has 4.5 core.
	if (!glfwInit())
	{
		LOG_ERROR("GlInterop: glfwInit failed (was it called by VulkanInstance first?)");
		return false;
	}
	_hasGlfwInit = true;

	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	_window = glfwCreateWindow(64, 64, "Cave LG GL context (hidden)", nullptr, nullptr);
	if (!_window)
	{
		LOG_ERROR("GlInterop: glfwCreateWindow failed for the hidden GL context window. "
		          "Driver may not support GL 4.5 core or GL contexts may be unavailable.");
		Shutdown();
		return false;
	}

	glfwMakeContextCurrent(_window);

	if (!LoadGlFunctions())
	{
		Shutdown();
		return false;
	}

	if (glGetString_fp)
	{
		const GLubyte* v = glGetString_fp(GL_VERSION);
		const GLubyte* vd = glGetString_fp(GL_VENDOR);
		const GLubyte* r = glGetString_fp(GL_RENDERER);
		_glVersion = v ? reinterpret_cast<const char*>(v) : "?";
		_glVendor = vd ? reinterpret_cast<const char*>(vd) : "?";
		_glRenderer = r ? reinterpret_cast<const char*>(r) : "?";
		LOG_INFO("GlInterop: GL_VERSION={} GL_VENDOR={} GL_RENDERER={}", _glVersion, _glVendor, _glRenderer);
	}

	if (!VerifyRequiredExtensions())
	{
		Shutdown();
		return false;
	}

	_initialized = true;
	LOG_INFO("GlInterop: GL 4.5 context + interop extensions ready on hidden GLFW window");
	return true;
}

void GlInteropContext::Shutdown()
{
	if (_window)
	{
		// Detach the context if it's current on this thread before destroying
		// the window — avoids GLFW warning about destroying a current context.
		if (glfwGetCurrentContext() == _window)
			glfwMakeContextCurrent(nullptr);
		glfwDestroyWindow(_window);
		_window = nullptr;
	}
	_initialized = false;
}

void GlInteropContext::MakeCurrent()
{
	if (_window) glfwMakeContextCurrent(_window);
}

void GlInteropContext::ReleaseCurrent()
{
	glfwMakeContextCurrent(nullptr);
}

bool GlInteropContext::LoadGlFunctions()
{
	bool ok = true;
	ok &= LoadGl(glGetString_fp, "glGetString");
	ok &= LoadGl(glGetStringi_fp, "glGetStringi");
	ok &= LoadGl(glGetIntegerv_fp, "glGetIntegerv");
	ok &= LoadGl(glGetError_fp, "glGetError");
	ok &= LoadGl(glFlush_fp, "glFlush");

	// EXT_external_objects + EXT_external_objects_win32 — interop core
	ok &= LoadGl(glCreateMemoryObjectsEXT_fp, "glCreateMemoryObjectsEXT");
	ok &= LoadGl(glDeleteMemoryObjectsEXT_fp, "glDeleteMemoryObjectsEXT");
	ok &= LoadGl(glImportMemoryWin32HandleEXT_fp, "glImportMemoryWin32HandleEXT");
	ok &= LoadGl(glTextureStorageMem2DEXT_fp, "glTextureStorageMem2DEXT");
	ok &= LoadGl(glGenSemaphoresEXT_fp, "glGenSemaphoresEXT");
	ok &= LoadGl(glDeleteSemaphoresEXT_fp, "glDeleteSemaphoresEXT");
	ok &= LoadGl(glImportSemaphoreWin32HandleEXT_fp, "glImportSemaphoreWin32HandleEXT");
	ok &= LoadGl(glWaitSemaphoreEXT_fp, "glWaitSemaphoreEXT");
	ok &= LoadGl(glSignalSemaphoreEXT_fp, "glSignalSemaphoreEXT");

	// Texture creation + readback — used in the M2 round-trip smoke test
	ok &= LoadGl(glCreateTextures_fp, "glCreateTextures");
	ok &= LoadGl(glDeleteTextures_fp, "glDeleteTextures");
	ok &= LoadGl(glGetTextureImage_fp, "glGetTextureImage");
	ok &= LoadGl(glTextureParameteri_fp, "glTextureParameteri");

	if (!ok)
	{
		LOG_ERROR("GlInterop: one or more required GL function pointers failed to load. "
		          "Likely cause: driver does not support GL 4.5 core or the EXT_memory_object / "
		          "EXT_semaphore extension family.");
	}
	return ok;
}

bool GlInteropContext::VerifyRequiredExtensions()
{
	if (!glGetIntegerv_fp || !glGetStringi_fp)
	{
		LOG_ERROR("GlInterop: extension query functions missing — fundamental GL load failed");
		return false;
	}

	GLint extensionCount = 0;
	glGetIntegerv_fp(GL_NUM_EXTENSIONS, &extensionCount);

	const std::vector<const char*> required = {
		"GL_EXT_memory_object",
		"GL_EXT_memory_object_win32",
		"GL_EXT_semaphore",
		"GL_EXT_semaphore_win32",
	};
	std::vector<bool> found(required.size(), false);

	for (GLint i = 0; i < extensionCount; i++)
	{
		const GLubyte* extName = glGetStringi_fp(GL_EXTENSIONS, static_cast<GLuint>(i));
		if (!extName) continue;
		const char* extStr = reinterpret_cast<const char*>(extName);
		for (size_t r = 0; r < required.size(); r++)
		{
			if (!found[r] && std::strcmp(extStr, required[r]) == 0)
				found[r] = true;
		}
	}

	bool allFound = true;
	for (size_t r = 0; r < required.size(); r++)
	{
		if (found[r])
		{
			LOG_DEBUG("GlInterop: extension '{}' present", required[r]);
		}
		else
		{
			LOG_ERROR("GlInterop: REQUIRED extension '{}' missing — Vulkan-OpenGL interop unavailable on this driver. "
			          "On NVIDIA, install latest Studio or Game Ready driver. On AMD, install latest Adrenalin.", required[r]);
			allFound = false;
		}
	}
	return allFound;
}

uint32_t GlInteropContext::ImportImageHandleAsTexture(void* win32Handle, uint64_t memorySize,
                                                      uint32_t width, uint32_t height,
                                                      uint32_t glInternalFormat)
{
	if (!_initialized)
	{
		LOG_ERROR("GlInterop::ImportImageHandleAsTexture called before Initialize");
		return 0;
	}
	if (!win32Handle)
	{
		LOG_ERROR("GlInterop::ImportImageHandleAsTexture got null handle");
		return 0;
	}

	// Drain any pre-existing GL error so the per-step checks below are clean.
	while (glGetError_fp() != GL_NO_ERROR) { /* drain */ }

	LOG_DEBUG("GlInterop::ImportImageHandleAsTexture: handle={} size={} bytes ({} MiB) {}x{} format=0x{:04X}",
		win32Handle, memorySize, memorySize >> 20, width, height, glInternalFormat);

	// 1) Allocate a GL memory object backed by the Win32 HANDLE.
	GLuint memoryObject = 0;
	glCreateMemoryObjectsEXT_fp(1, &memoryObject);
	if (GLenum e = glGetError_fp(); e != GL_NO_ERROR)
		LOG_ERROR("  glCreateMemoryObjectsEXT failed: 0x{:04X}", e);

	glImportMemoryWin32HandleEXT_fp(memoryObject, memorySize, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, win32Handle);
	if (GLenum e = glGetError_fp(); e != GL_NO_ERROR)
	{
		LOG_WARNING("  glImportMemoryWin32HandleEXT(size={}) failed 0x{:04X}; retrying with size=0", memorySize, e);
		while (glGetError_fp() != GL_NO_ERROR) {} // drain
		glDeleteMemoryObjectsEXT_fp(1, &memoryObject);
		glCreateMemoryObjectsEXT_fp(1, &memoryObject);
		glImportMemoryWin32HandleEXT_fp(memoryObject, 0, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, win32Handle);
		GLenum e2 = glGetError_fp();
		if (e2 != GL_NO_ERROR)
		{
			LOG_ERROR("  glImportMemoryWin32HandleEXT(size=0) also failed: 0x{:04X}", e2);
		}
		else
		{
			LOG_INFO("  glImportMemoryWin32HandleEXT(size=0) succeeded");
		}
	}

	// 2) Allocate the GL texture name.
	constexpr GLenum GL_TEXTURE_2D = 0x0DE1;
	GLuint texture = 0;
	glCreateTextures_fp(GL_TEXTURE_2D, 1, &texture);
	if (GLenum e = glGetError_fp(); e != GL_NO_ERROR)
		LOG_ERROR("  glCreateTextures failed: 0x{:04X}", e);

	// 3) Bind the texture's storage to the imported memory at offset 0.
	//    glTextureStorageMem2DEXT(texture, mipLevels=1, internalformat, w, h, memoryObject, offset=0)
	glTextureStorageMem2DEXT_fp(texture, 1, glInternalFormat, width, height, memoryObject, 0);
	if (GLenum e = glGetError_fp(); e != GL_NO_ERROR)
		LOG_ERROR("  glTextureStorageMem2DEXT failed: 0x{:04X} (texture={} format=0x{:04X} {}x{})", e, texture, glInternalFormat, width, height);

	// 4) NEAREST filtering — important for the LG quilt: when Bridge samples
	//    our quilt to compose the lenticular output it does its own subpixel
	//    re-sampling. GL_LINEAR magnification on top introduces a second
	//    blur pass that softens cell edges. GL_NEAREST keeps each cell sharp
	//    so the lenticular optics stay the only source of inter-view blend.
	constexpr GLenum GL_TEXTURE_MIN_FILTER = 0x2801;
	constexpr GLenum GL_TEXTURE_MAG_FILTER = 0x2800;
	constexpr GLint  GL_NEAREST            = 0x2600;
	glTextureParameteri_fp(texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTextureParameteri_fp(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	while (glGetError_fp() != GL_NO_ERROR) {} // drain — these are best-effort

	// 5) The memory object is owned by the texture binding; we can delete the
	//    name now (the GL driver keeps the underlying binding alive). This
	//    matches the EXT_external_objects spec idiom.
	glDeleteMemoryObjectsEXT_fp(1, &memoryObject);

	LOG_DEBUG("GlInterop: imported Win32 HANDLE as GL texture {} ({}x{}, format=0x{:04X})",
	          texture, width, height, glInternalFormat);
	return texture;
}

uint32_t GlInteropContext::ImportSemaphoreHandle(void* win32Handle)
{
	if (!_initialized || !win32Handle) return 0;

	GLuint glSem = 0;
	glGenSemaphoresEXT_fp(1, &glSem);
	glImportSemaphoreWin32HandleEXT_fp(glSem, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, win32Handle);

	GLenum err = glGetError_fp();
	if (err != GL_NO_ERROR)
	{
		LOG_ERROR("GlInterop::ImportSemaphoreHandle: GL error 0x{:04X}", err);
	}
	LOG_DEBUG("GlInterop: imported Win32 HANDLE as GL semaphore {}", glSem);
	return glSem;
}

void GlInteropContext::WaitSemaphoreOnTexture(uint32_t glSemaphore, uint32_t glTexture)
{
	// Wait for Vulkan to signal the semaphore, on a single texture in GENERAL layout.
	GLuint sem = glSemaphore;
	GLuint tex = glTexture;
	GLenum layout = GL_LAYOUT_GENERAL_EXT;
	glWaitSemaphoreEXT_fp(sem, 0, nullptr, 1, &tex, &layout);
}

void GlInteropContext::SignalSemaphoreOnTexture(uint32_t glSemaphore, uint32_t glTexture)
{
	GLuint sem = glSemaphore;
	GLuint tex = glTexture;
	GLenum layout = GL_LAYOUT_GENERAL_EXT;
	glSignalSemaphoreEXT_fp(sem, 0, nullptr, 1, &tex, &layout);
}

void GlInteropContext::DeleteTexture(uint32_t glTexture)
{
	if (glTexture && glDeleteTextures_fp)
	{
		GLuint t = glTexture;
		glDeleteTextures_fp(1, &t);
	}
}

void GlInteropContext::DeleteSemaphore(uint32_t glSemaphore)
{
	if (glSemaphore && glDeleteSemaphoresEXT_fp)
	{
		GLuint s = glSemaphore;
		glDeleteSemaphoresEXT_fp(1, &s);
	}
}

void GlInteropContext::ReadTextureRGBA8(uint32_t glTexture, uint32_t width, uint32_t height, void* outPixels)
{
	if (!_initialized || !glTexture || !outPixels) return;
	GLsizei bufSize = static_cast<GLsizei>(width * height * 4);
	glGetTextureImage_fp(glTexture, 0, GL_RGBA, GL_UNSIGNED_BYTE, bufSize, outPixels);
}

void GlInteropContext::Flush()
{
	if (glFlush_fp) glFlush_fp();
}

// ----- Smoke-test driver --------------------------------------------------

namespace
{
	// Round-trip smoke: Vulkan allocates an exportable RGBA8 image, clears it
	// to magenta, signals a binary external semaphore. GL imports both, waits
	// for the semaphore, reads the texture, asserts the read-back pixels are
	// magenta. Validates the entire interop primitive stack (image export,
	// memory HANDLE pass-through, semaphore HANDLE pass-through, layout
	// handshake, GL texture binding to imported memory).
	int RunRoundtripTest(Cave::DeviceContext& dc)
	{
		using namespace Cave;
		LOG_INFO("=== Looking Glass smoke test: Vulkan->GL round-trip ===");

		const uint32_t kWidth = 256;
		const uint32_t kHeight = 256;
		const uint8_t kMagentaR = 255, kMagentaG = 0, kMagentaB = 255, kMagentaA = 255;

		// 1. Vulkan-side: exportable image + binary external semaphore.
		LOG_DEBUG("Roundtrip: creating exportable image");
		auto exportImage = Image::CreateExportable(
			dc,
			vk::Format::eR8G8B8A8Unorm,
			vk::Extent3D{kWidth, kHeight, 1},
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
		LOG_DEBUG("Roundtrip: exportable image created, allocatedMemorySize={}", exportImage->GetAllocatedMemorySize());

		LOG_DEBUG("Roundtrip: creating external binary semaphore");
		ExternalBinarySemaphore vkToGlSem(dc);
		LOG_DEBUG("Roundtrip: semaphore created");

		// 2. Record + submit a clear-to-magenta to put magenta into the image
		//    AND transition it to VK_IMAGE_LAYOUT_GENERAL (so the GL side can
		//    use GL_LAYOUT_GENERAL_EXT in glWaitSemaphoreEXT).
		auto pool = dc.GetGraphicsCommandPool();
		auto cmd = dc.BeginSingleTimeCommands(pool);

		vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

		// Undefined -> TransferDst
		vk::ImageMemoryBarrier toTransferDst(
			{}, vk::AccessFlagBits::eTransferWrite,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			*exportImage->GetImage(), range);
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
			vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, toTransferDst);

		vk::ClearColorValue magenta;
		magenta.setFloat32({1.0f, 0.0f, 1.0f, 1.0f});
		cmd.clearColorImage(*exportImage->GetImage(), vk::ImageLayout::eTransferDstOptimal, magenta, range);

		// TransferDst -> General (so GL can read in GENERAL layout)
		vk::ImageMemoryBarrier toGeneral(
			vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
			vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			*exportImage->GetImage(), range);
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eAllCommands, {}, {}, {}, toGeneral);

		cmd.end();

		vk::SubmitInfo submit{};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &cmd;
		// Signal the binary external semaphore on submit-complete.
		auto sem = vkToGlSem.GetSemaphore();
		submit.signalSemaphoreCount = 1;
		submit.pSignalSemaphores = &sem;

		dc.GetGraphicsQueue().submit(submit, nullptr);
		// Note: we DON'T wait on the queue here — the GL side blocks on the
		// semaphore. That's the actual interop test.

		// 3. Export the image memory + semaphore HANDLEs.
		void* memHandle = exportImage->GetMemoryWin32Handle();
		void* semHandle = vkToGlSem.GetWin32Handle();
		if (!memHandle || !semHandle)
		{
			LOG_ERROR("Roundtrip: failed to export Win32 HANDLEs");
			dc.GetDevice().waitIdle();
			dc.GetDevice().freeCommandBuffers(pool, 1, &cmd);
			return 2;
		}

		// 4. GL-side: init context, import HANDLEs, wait on semaphore, readback.
		GlInteropContext ctx;
		if (!ctx.Initialize())
		{
			LOG_ERROR("Roundtrip: GL context init failed");
			dc.GetDevice().waitIdle();
			dc.GetDevice().freeCommandBuffers(pool, 1, &cmd);
			return 3;
		}
		// GLFW's Initialize re-set the current context implicitly; ensure it.
		ctx.MakeCurrent();

		// GL_RGBA8 = 0x8058 — defined as constexpr at top of file.
		uint32_t glTex = ctx.ImportImageHandleAsTexture(
			memHandle, exportImage->GetAllocatedMemorySize(),
			kWidth, kHeight, GL_RGBA8);
		uint32_t glSem = ctx.ImportSemaphoreHandle(semHandle);

		if (!glTex || !glSem)
		{
			LOG_ERROR("Roundtrip: GL import failed (tex={}, sem={})", glTex, glSem);
			ctx.Shutdown();
			dc.GetDevice().waitIdle();
			dc.GetDevice().freeCommandBuffers(pool, 1, &cmd);
			return 4;
		}
		// Mark the HANDLEs as transferred — GL keeps a reference now (no
		// CloseHandle in destructor).
		// NOTE: per GL_EXT_external_objects_win32 spec, glImportMemoryWin32HandleEXT
		// + glImportSemaphoreWin32HandleEXT do NOT consume the HANDLE on Windows
		// (unlike POSIX FD imports). However once imported, GL may keep the
		// HANDLE referenced internally; closing it from our side is undefined.
		// Safest: leave the HANDLEs alone (the wrappers' destructors will close
		// them when the wrappers go out of scope). For the smoke test that's
		// fine — wrappers live until function exit.

		// 5. GL waits for Vulkan to signal.
		ctx.WaitSemaphoreOnTexture(glSem, glTex);

		// 6. Read pixels back.
		std::vector<uint8_t> readback(kWidth * kHeight * 4, 0);
		ctx.ReadTextureRGBA8(glTex, kWidth, kHeight, readback.data());
		ctx.Flush();

		// 7. Verify.
		size_t mismatches = 0;
		for (size_t i = 0; i < readback.size(); i += 4)
		{
			if (readback[i + 0] != kMagentaR ||
			    readback[i + 1] != kMagentaG ||
			    readback[i + 2] != kMagentaB ||
			    readback[i + 3] != kMagentaA)
			{
				mismatches++;
			}
		}

		// 8. Cleanup.
		ctx.DeleteTexture(glTex);
		ctx.DeleteSemaphore(glSem);
		ctx.Shutdown();

		dc.GetDevice().waitIdle();
		dc.GetDevice().freeCommandBuffers(pool, 1, &cmd);

		if (mismatches > 0)
		{
			size_t total = kWidth * kHeight;
			LOG_ERROR("Roundtrip FAILED: {}/{} pixels did NOT match magenta. Sample first pixel = ({}, {}, {}, {})",
			          mismatches, total, readback[0], readback[1], readback[2], readback[3]);
			return 5;
		}

		LOG_INFO("Roundtrip PASS: all {} pixels are magenta — Vulkan-OpenGL interop end-to-end works on this driver",
		         kWidth * kHeight);
		return 0;
	}
}

// Defined in lookingGlassMode.cpp (where bridge.h is safely included).
int RunBridgeDisplaysTest();

int RunLookingGlassSmokeTest(const std::string& testName, DeviceContext* deviceContext)
{
	if (testName == "displays")
	{
		return RunBridgeDisplaysTest();
	}
	if (testName == "context")
	{
		LOG_INFO("=== Looking Glass smoke test: context bring-up ===");
		GlInteropContext ctx;
		if (!ctx.Initialize())
		{
			LOG_ERROR("Context smoke test FAILED to initialize");
			return 1;
		}
		LOG_INFO("Context smoke test PASS: GL_VERSION={} GL_VENDOR={} GL_RENDERER={}",
		         ctx.GlVersion(), ctx.GlVendor(), ctx.GlRenderer());
		ctx.Shutdown();
		return 0;
	}
	else if (testName == "roundtrip")
	{
		if (!deviceContext)
		{
			LOG_ERROR("Roundtrip smoke test requires a DeviceContext (engine must initialize Vulkan first)");
			return 1;
		}
		return RunRoundtripTest(*deviceContext);
	}
	else if (testName == "save-quilt")
	{
		if (!deviceContext)
		{
			LOG_ERROR("save-quilt smoke test requires a DeviceContext");
			return 1;
		}
		LOG_INFO("=== Looking Glass smoke test: save-quilt (test pattern) ===");

		QuiltRenderSystem quilt(*deviceContext);
		quilt.RenderTestPattern();

		std::vector<uint8_t> rgba8;
		quilt.ReadbackQuiltToHost(rgba8);

		const char* outPath = "build/Release/lg_test_quilt.png";
		const int strideBytes = static_cast<int>(QuiltRenderSystem::kQuiltWidth * 4);
		int writeRes = stbi_write_png(outPath,
			static_cast<int>(QuiltRenderSystem::kQuiltWidth),
			static_cast<int>(QuiltRenderSystem::kQuiltHeight),
			4, rgba8.data(), strideBytes);
		if (!writeRes)
		{
			LOG_ERROR("save-quilt: stbi_write_png failed for {}", outPath);
			return 2;
		}
		LOG_INFO("save-quilt PASS: wrote {}x{} RGBA8 quilt PNG to '{}' "
		         "(open in Looking Glass Studio app to verify 8x6=48 view layout)",
		         QuiltRenderSystem::kQuiltWidth, QuiltRenderSystem::kQuiltHeight, outPath);
		return 0;
	}
	else if (testName.empty())
	{
		return 0;
	}
	else
	{
		LOG_WARNING("Unknown --lg-test={} (valid: context | roundtrip | save-quilt | displays)", testName);
		return 0;
	}
}

} // namespace Cave
