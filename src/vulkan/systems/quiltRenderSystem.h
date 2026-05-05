#pragma once

#include <vulkan/vulkan.hpp>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "../deviceContext.h"
#include "../image.h"
#include "../buffer.h"
#include "../externalSemaphore.h"
#include "../descriptors.h"
#include "../pipelines/graphicsPipeline.h"
#include "../simulationRenderer.h"
#include "../../common/structs.h"

namespace Cave
{
	// Quilt-format render target for the Looking Glass Portrait.
	//
	// Owns a single exportable 3360×3360 RGBA8 image arranged as an 8×6 grid of
	// 420×560 per-view tiles. The image's VkDeviceMemory is allocated with
	// VkExportMemoryAllocateInfo so its Win32 HANDLE can be imported into
	// OpenGL via glImportMemoryWin32HandleEXT.
	//
	// Two operating modes:
	//   - RenderTestPattern() — host-side: fills the quilt with an 8×6 rainbow
	//     gradient. Used by the M3 save-quilt smoke test (validates the quilt
	//     format loads in the Looking Glass Studio app before linking Bridge).
	//   - RenderFrame(...) — real per-view ray-march of the cellular automata.
	//     48 viewport-tiled draws inside one dynamic-rendering scope, each with
	//     its own camera UBO whose eye is offset along cameraRight by
	//     (viewIndex / 47 - 0.5) × viewconeWidth.
	//
	// Threading: all methods run on the calling thread. Real-render submits to
	// the graphics queue with a wait on the compute timeline + signal on a
	// caller-provided binary external semaphore (the GL handshake primitive).
	class QuiltRenderSystem
	{
	public:
		// Looking Glass Portrait canonical quilt format — 3360×3360 RGBA8 with
		// 8×6 = 48 views per the LG Studio specification. Bridge consumes this
		// natively without resampling, preserving max optical quality.
		static constexpr uint32_t kQuiltWidth     = 3360;
		static constexpr uint32_t kQuiltHeight    = 3360;
		static constexpr uint32_t kQuiltColumns   = 8;
		static constexpr uint32_t kQuiltRows      = 6;
		static constexpr uint32_t kQuiltViewCount = kQuiltColumns * kQuiltRows; // 48
		static constexpr uint32_t kViewWidth      = kQuiltWidth / kQuiltColumns; // 420
		static constexpr uint32_t kViewHeight     = kQuiltHeight / kQuiltRows;   // 560

		explicit QuiltRenderSystem(DeviceContext& deviceContext);
		~QuiltRenderSystem();

		QuiltRenderSystem(const QuiltRenderSystem&) = delete;
		QuiltRenderSystem& operator=(const QuiltRenderSystem&) = delete;

		// Lazily builds the graphics pipeline + 48 camera UBOs + 48 descriptor
		// sets, bound to simulationRenderer's cell state buffers. Called by
		// RenderFrame on first invocation.
		void EnsureRealRenderPath(VulkanSimulationRenderer& simulationRenderer);

		// Real-render path: 48 per-view ray-marches into the quilt tiles. The
		// caller's `baseCameraData` provides the central camera; the system
		// generates 48 per-view variants by offsetting the eye along cameraRight.
		// `viewconeRadians` is the total swing angle of the viewing volume
		// (typical: ~35° for LG Portrait — equivalent to half-radians = 0.305).
		// `computeWaitSemaphore` (timeline) + `computeWaitValue` are inserted
		// as a wait on the submit, so the compute system's tick must complete
		// before the fragment shader reads cell state. `signalSem` is a binary
		// external semaphore that gets signaled when the quilt is fully written
		// and ready for OpenGL import; pass nullptr to skip signaling.
		void RenderFrame(VulkanSimulationRenderer& simulationRenderer,
		                 const CameraData& baseCameraData,
		                 const RayMarchPushConstants& pushConstants,
		                 float viewconeRadians,
		                 vk::Semaphore computeWaitSemaphore,
		                 uint64_t computeWaitValue,
		                 ExternalBinarySemaphore* signalSem,
		                 uint32_t renderFromFrameIndex);

		// Fills the 48 tiles with a distinct color per view. Used by the M3
		// save-quilt smoke test (validates quilt format independently of the
		// real-render path). Synchronous — blocks on a single-time command.
		void RenderTestPattern();

		// Reads the entire quilt back to host memory. Output buffer must be at
		// least kQuiltWidth * kQuiltHeight * 4 bytes (RGBA8). Image is left in
		// VK_IMAGE_LAYOUT_GENERAL on return.
		void ReadbackQuiltToHost(std::vector<uint8_t>& outRgba8);

		// Accessors for the LookingGlassMode integration:
		std::shared_ptr<Image> GetQuiltImage() const { return _quiltImage; }
		void* GetQuiltMemoryWin32Handle() const { return _quiltImage ? _quiltImage->GetMemoryWin32Handle() : nullptr; }
		vk::DeviceSize GetQuiltMemorySize() const { return _quiltImage ? _quiltImage->GetAllocatedMemorySize() : 0; }

	private:
		DeviceContext& _deviceContext;

		// Output: the exportable quilt image. Lives in eGeneral after first
		// transition (so the GL side can use GL_LAYOUT_GENERAL_EXT throughout
		// — sidesteps the dominant interop layout bug class).
		std::shared_ptr<Image> _quiltImage;
		bool _initialLayoutTransitioned = false;

		// Real-render path resources, built lazily on first RenderFrame call.
		bool _realRenderPathReady = false;
		std::unique_ptr<Image> _depthAttachment;
		std::vector<std::shared_ptr<Buffer>> _viewCameraBuffers; // size = kQuiltViewCount; one UBO per view
		// One descriptor set per (frameIndex, view). Indexed as
		// _viewDescriptorsPerFrame[frameIndex][view]. The frameIndex selects
		// which cell-state buffer the fragment shader reads — matched to the
		// buffer that the compute system just wrote to in this tick.
		std::vector<std::vector<std::shared_ptr<Descriptor>>> _viewDescriptorsPerFrame;
		std::unique_ptr<GraphicsPipeline> _graphicsPipeline;
		std::unique_ptr<Buffer> _boundingBoxVertexBuffer;
		std::unique_ptr<Buffer> _boundingBoxIndexBuffer;
		uint32_t _boundingBoxIndexCount = 0;
		vk::CommandPool _graphicsCommandPool{};
		vk::CommandBuffer _renderCommandBuffer{};
		vk::Fence _renderFence{};

		// Compute per-view CameraData given the base camera and a fractional
		// position along the viewcone. t = 0 is the leftmost view, t = 1 is the
		// rightmost. The eye is offset along cameraRight by tan(viewcone/2) ×
		// (2t - 1) × distanceToTarget. Other fields (view+projection matrices,
		// lighting) are recomputed/tweaked accordingly.
		CameraData ComputePerViewCamera(const CameraData& base, float t, float viewconeRadians) const;

		void EnsureGeneralLayout(vk::CommandBuffer cmd);
		void BuildBoundingBoxMesh();
		void BuildDepthAttachment();
		void BuildCameraUbosAndDescriptors(VulkanSimulationRenderer& simulationRenderer);
		void BuildGraphicsPipeline(VulkanSimulationRenderer& simulationRenderer);
	};
} // namespace Cave
