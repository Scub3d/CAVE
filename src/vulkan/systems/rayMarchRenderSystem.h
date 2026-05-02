#pragma once

#include <vulkan/vulkan.hpp>
#include <memory>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../deviceContext.h"
#include "../simulationRenderer.h"
#include "../pipelines/graphicsPipeline.h"
#include "../descriptors.h"
#include "../buffer.h"
#include "../image.h"
#include "../../common/queryManager.h"

namespace Cave
{
	class RayMarchRenderSystem
	{
	private:
		DeviceContext& _deviceContext;
		VulkanSimulationRenderer& _simulationRenderer;
		uint32_t _framesInFlight;
		vk::Extent2D _renderExtent;
		bool _compositeEnabled = false; // when true, uses R16G16B16A16Sfloat + alpha=0 clear for compositing
		vk::Format _colorFormat;

		vk::CommandPool _graphicsCommandPool;
		std::vector<vk::CommandBuffer> _graphicsCommandBuffers;

		std::shared_ptr<Descriptor> _descriptor;
		std::unique_ptr<GraphicsPipeline> _graphicsPipeline;

		// Per-frame camera uniform buffers
		std::vector<std::shared_ptr<Buffer>> _cameraUniformBuffers;

		// Off-screen render targets (one per frame in flight)
		std::vector<std::shared_ptr<Image>> _colorAttachments;
		std::unique_ptr<Image> _depthAttachment;

		// Bounding box mesh
		std::unique_ptr<Buffer> _boundingBoxVertexBuffer;
		std::unique_ptr<Buffer> _boundingBoxIndexBuffer;
		uint32_t _boundingBoxIndexCount;

		// Shader modules
		vk::ShaderModule _vertexShaderModule;
		vk::ShaderModule _fragmentShaderModule;

		RayMarchPushConstants _rayMarchPushConstants;

		std::vector<vk::Fence> _inFlightFences;
		std::vector<vk::Semaphore> _renderFinishedSemaphores;
		std::vector<uint64_t> _renderFinishedSemaphoreSignalValues;

	private:
		void BuildBoundingBoxMesh();
		void BuildCameraBuffers();
		void BuildAttachments();
		void BuildDescriptors();
		void BuildShaders();
		void BuildPipeline();
		void BuildCommandBuffers();

		void UpdateCameraBuffer(uint32_t frameIndex, const CameraData& cameraData);

	public:
		RayMarchRenderSystem(DeviceContext& deviceContext, VulkanSimulationRenderer& simulationRenderer,
							 vk::Extent2D renderExtent, bool compositeEnabled = false);
		~RayMarchRenderSystem();

		void RebuildDescriptors();

		RayMarchRenderSystem(const RayMarchRenderSystem&) = delete;
		RayMarchRenderSystem& operator=(const RayMarchRenderSystem&) = delete;

		void RenderFrame(uint32_t frameIndex, vk::Semaphore computeCompletedSemaphore,
						 uint64_t computeCompletedSemaphoreWaitValue, const CameraData& cameraData,
						 const RayMarchPushConstants& rayMarchPushConstants,
						 std::shared_ptr<QueryManager> queryManager = nullptr, vk::QueryPool queryPool = {},
						 bool waitForCompute = true);

		vk::Image GetOutputImage(uint32_t frameIndex) const;
		vk::ImageView GetOutputImageView(uint32_t frameIndex) const;
		std::shared_ptr<Image> GetColorAttachment(uint32_t frameIndex) const { return _colorAttachments[frameIndex]; }

		vk::Fence GetInFlightFence(uint32_t frameIndex) const { return _inFlightFences[frameIndex]; }
		vk::Semaphore GetRenderFinishedSemaphore(uint32_t frameIndex) const { return _renderFinishedSemaphores[frameIndex]; }
		uint64_t GetRenderFinishedSemaphoreSignalValue(uint32_t frameIndex) const { return _renderFinishedSemaphoreSignalValues[frameIndex]; }
		vk::Extent2D GetRenderExtent() const { return _renderExtent; }
		vk::Format GetColorFormat() const { return _colorFormat; }
	};
}
