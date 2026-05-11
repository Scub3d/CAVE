#pragma once

#include <vulkan/vulkan.hpp>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../deviceContext.h"
#include "../simulationRenderer.h"
#include "../pipelines/graphicsPipeline.h"
#include "../descriptors.h"
#include "../buffer.h"
#include "../image.h"
#include "../video/videoEncoder.h"
#include "videoOverlay.h"
#include "../../common/queryManager.h"

namespace Cave { class VulkanInstance; }

namespace Cave
{
	// Runs its own internal render pass and H.264 encode pipeline.
	// Does NOT depend on RenderSystem — designed for Mode 3 (headless video export)
	// where skipping the screen render pass saves GPU resources.
	//
	// Data flow:
	//   ComputeSystem → (cell/draw-command buffers in VulkanSimulationRenderer)
	//       → VideoEncoderSystem internal render → H.264 encode → video file
	class VideoEncoderSystem
	{
	private:
		DeviceContext& _deviceContext;
		VulkanSimulationRenderer& _simulationRenderer;
		uint32_t _width;
		uint32_t _height;
		uint32_t _fps;

		// Internal render path (mirrors RayMarchRenderSystem but writes to encode-ready images)
		vk::CommandPool _graphicsCommandPool;
		std::vector<vk::CommandBuffer> _graphicsCommandBuffers;
		std::shared_ptr<Descriptor> _renderDescriptor;
		std::unique_ptr<GraphicsPipeline> _graphicsPipeline;
		std::vector<std::shared_ptr<Buffer>> _cameraUniformBuffers;
		std::vector<std::unique_ptr<Image>> _colorAttachments;
		std::unique_ptr<Image> _depthAttachment;
		vk::Format _depthFormat = vk::Format::eUndefined;
		std::vector<vk::Fence> _renderInFlightFences;

		// Bounding box mesh for ray march rendering
		std::unique_ptr<Buffer> _boundingBoxVertexBuffer;
		std::unique_ptr<Buffer> _boundingBoxIndexBuffer;
		uint32_t _boundingBoxIndexCount;

		// Encode pipeline
		std::unique_ptr<VideoEncoder> _videoEncoder;

		// Bitstream sink. When streaming is on, NALs are written straight to _bitstreamFile
		// per frame and _accumulatedBitstream stays empty. When streaming is off (legacy
		// path, retained for A/B benchmarking), every NAL is appended to _accumulatedBitstream
		// and the caller drains it via Finish().
		bool _bitstreamStreamingEnabled = false;
		std::ofstream _bitstreamFile;
		std::string _bitstreamFilePath;
		size_t _bitstreamBytesWritten = 0;
		std::vector<char> _accumulatedBitstream;

		// Metadata overlay drawn into the encoder's color attachment each frame.
		// Owns its own ImGui context (separate from GuiSystem) because Vulkan dynamic
		// rendering requires the pipeline format to match the runtime attachment, and
		// the encoder's R8G8B8A8Unorm differs from GuiSystem's swapchain-bound format.
		std::unique_ptr<VideoOverlay> _videoOverlay;
		bool _overlayEnabled = false;

	private:
		void BuildBoundingBoxMesh();
		void BuildCameraBuffers();
		void BuildColorAttachments();
		void BuildRenderDescriptors();
		void BuildRenderPipeline();
		void BuildRenderCommandBuffers();

		void UpdateCameraBuffer(uint32_t frameIndex, const CameraData& cameraData);

		// Routes one NAL/header chunk produced by VideoEncoder::FinishEncode either to the
		// open bitstream file (streaming) or to _accumulatedBitstream (legacy).
		void WriteBitstreamChunk(const char* data, size_t size);

	public:
		VideoEncoderSystem(VulkanInstance& vulkanInstance, DeviceContext& deviceContext,
		                   VulkanSimulationRenderer& simulationRenderer,
		                   uint32_t width, uint32_t height, uint32_t fps);
		~VideoEncoderSystem();

		VideoEncoderSystem(const VideoEncoderSystem&) = delete;
		VideoEncoderSystem& operator=(const VideoEncoderSystem&) = delete;

		// Resets for a new encoding job: rebuilds graphics pipeline with updated shaders
		// and creates a fresh H.264 encoder session. Reuses color/depth attachments and camera buffers.
		void ResetForNewJob();

		// Configures the bitstream sink for the next job. Call once before the first
		// encode of every job (including the first job, where ResetForNewJob is not
		// invoked). When streaming is true the H.264 file is opened here and each NAL
		// is written as it's produced. When streaming is false the legacy in-RAM path
		// is used and outputPath is recorded for diagnostic logging only.
		void OpenBitstreamForJob(const std::string& outputPath, bool streaming);

		// Renders the current simulation state internally, then queues it for encoding.
		// Should be called after ComputeSystem::Tick() for the same frameIndex.
		void RenderAndEncodeFrame(uint32_t frameIndex, vk::Semaphore computeCompletedSemaphore,
								  uint64_t computeCompletedSemaphoreWaitValue, const CameraData& cameraData,
								  const RayMarchPushConstants& rayMarchPushConstants,
								  std::shared_ptr<QueryManager> queryManager = nullptr, vk::QueryPool queryPool = {});

		// Blits an external image (e.g., dual-GPU compositor output) into the encoder's
		// color attachment and encodes it. Handles format conversion via vkCmdBlitImage
		// (e.g., R16G16B16A16Sfloat → B8G8R8A8Srgb). Use instead of RenderAndEncodeFrame
		// when rendering is handled externally.
		void BlitAndEncodeFrame(uint32_t frameIndex, vk::Image sourceImage, vk::Extent2D sourceExtent);

		// Blocks until all queued frames are encoded and finalizes the bitstream sink.
		// When streaming is enabled, closes _bitstreamFile and leaves outBitstream empty;
		// the caller should treat the on-disk file at the path passed to OpenBitstreamForJob
		// as the finished output. When streaming is disabled, moves the accumulated buffer
		// into outBitstream (legacy behavior).
		void Finish(std::vector<char>& outBitstream);

		// True if Finish() already wrote the bitstream to disk via streaming; the caller
		// should skip the legacy "open ofstream + write whole vector" step in that case.
		bool DidStreamBitstreamToDisk() const { return _bitstreamStreamingEnabled; }

		// Bytes written to _bitstreamFile during the most recent streamed job. Used for
		// logging at job end (and lets the mode print the same size line either path).
		size_t GetStreamedByteCount() const { return _bitstreamBytesWritten; }

		// Update overlay metadata (call before each RenderAndEncodeFrame/BlitAndEncodeFrame
		// to refresh per-tick fields like currentTick). Has no effect if overlay is disabled.
		void SetOverlayInfo(const OverlayInfo& info) { if (_videoOverlay) _videoOverlay->SetInfo(info); }
		void SetOverlayEnabled(bool enabled) { _overlayEnabled = enabled; }

		vk::Fence GetRenderFence(uint32_t frameIndex) const { return _renderInFlightFences[frameIndex]; }
		vk::Image GetColorAttachmentImage(uint32_t frameIndex) const { return *_colorAttachments[frameIndex]->GetImage(); }
	};
}
