#pragma once

#include <memory>
#include <cstdint>
#include <vulkan/vulkan.hpp>

#include "deviceContext.h"
#include "buffer.h"
#include "image.h"
#include "slangCompiler.h"
#include "descriptors.h"
#include "pipelines/computePipeline.h"
#include "shader.h"
#include "../common/startupConfig.h"

namespace Cave
{

	struct GhostCopyPushConstants
	{
		int32_t zStart;
		int32_t imageWidth;
		int32_t gridDimY;
		int32_t numBlocksX;
		int32_t numBlocksY;
		int32_t numBlocksZ;
	};

	// Manages ghost layer exchange between two GPUs for Z-axis domain decomposition.
	// Supports two modes:
	//   Buffer mode: compute shaders extract/insert Z-slices from tiled Morton buffers
	//   Image mode:  vkCmdCopyImageToBuffer/vkCmdCopyBufferToImage on 3D storage images
	// Image mode avoids the 4GB maxStorageBufferRange limit and needs no compute shaders.
	class GhostExchange
	{
	public:
		GhostExchange(DeviceContext& gpu0, DeviceContext& gpu1);
		~GhostExchange();

		GhostExchange(const GhostExchange&) = delete;
		GhostExchange& operator=(const GhostExchange&) = delete;

		void Initialize(uint32_t gridDimensionX, uint32_t gridDimensionY, uint32_t gridDimensionZ,
						uint32_t zMid, uint32_t ghostDepth, bool wrapEnabled,
						CellStateStorageMode storageMode = CellStateStorageMode::Buffer);

		// Buffer mode: pre-builds descriptors and pipelines for compute shader ghost exchange.
		void SetupForBufferMode(
			std::shared_ptr<Buffer> gpu0CellBuffer0, std::shared_ptr<Buffer> gpu0CellBuffer1,
			std::shared_ptr<Buffer> gpu1CellBuffer0, std::shared_ptr<Buffer> gpu1CellBuffer1,
			glm::ivec3 gpu0NumBlocks, glm::ivec3 gpu1NumBlocks);

		// Image mode: stores image references for direct vkCmdCopy-based ghost exchange.
		void SetupForImageMode(
			std::shared_ptr<Image> gpu0CellImage0, std::shared_ptr<Image> gpu0CellImage1,
			std::shared_ptr<Image> gpu1CellImage0, std::shared_ptr<Image> gpu1CellImage1);

		// Performs a complete ghost exchange on the cell state for the given frame.
		void Exchange(uint32_t frameIndex);

	private:
		DeviceContext& _gpu0;
		DeviceContext& _gpu1;

		CellStateStorageMode _storageMode = CellStateStorageMode::Buffer;
		uint32_t _ghostDepth = 0;
		uint32_t _imageWidth = 0;
		uint32_t _gridDimensionY = 0;
		uint32_t _gridDimensionZ = 0;
		uint32_t _zMid = 0;
		bool _wrapEnabled = false;

		// Local Z-sizes for each domain (including ghost slices on both sides)
		uint32_t _gpu0LocalZSize = 0;
		uint32_t _gpu1LocalZSize = 0;

		// Staging buffers (host-visible, linear layout, one per GPU)
		std::shared_ptr<Buffer> _gpu0Staging;
		std::shared_ptr<Buffer> _gpu1Staging;
		uint64_t _stagingBufferSize = 0;

		// Command pools, fences, and pre-allocated command buffers
		vk::CommandPool _gpu0CommandPool;
		vk::CommandPool _gpu1CommandPool;
		vk::Fence _gpu0Fence;
		vk::Fence _gpu1Fence;
		vk::CommandBuffer _gpu0CommandBuffer;
		vk::CommandBuffer _gpu1CommandBuffer;

		// === Buffer mode resources ===
		SlangCompiler _slangCompiler;
		vk::ShaderModule _gpu0ExtractShaderModule;
		vk::ShaderModule _gpu0InsertShaderModule;
		vk::ShaderModule _gpu1ExtractShaderModule;
		vk::ShaderModule _gpu1InsertShaderModule;

		std::shared_ptr<Descriptor> _gpu0ExtractDescriptor;
		std::shared_ptr<Descriptor> _gpu0InsertDescriptor;
		std::shared_ptr<Descriptor> _gpu1ExtractDescriptor;
		std::shared_ptr<Descriptor> _gpu1InsertDescriptor;

		std::unique_ptr<ComputePipeline> _gpu0ExtractPipeline;
		std::unique_ptr<ComputePipeline> _gpu0InsertPipeline;
		std::unique_ptr<ComputePipeline> _gpu1ExtractPipeline;
		std::unique_ptr<ComputePipeline> _gpu1InsertPipeline;

		std::shared_ptr<Buffer> _gpu0CellBuffers[2];
		std::shared_ptr<Buffer> _gpu1CellBuffers[2];
		glm::ivec3 _gpu0NumBlocks;
		glm::ivec3 _gpu1NumBlocks;

		void CompileShaders();
		void DispatchAndWait(DeviceContext& gpu, vk::CommandPool commandPool, vk::Fence fence,
							ComputePipeline& pipeline, uint32_t descriptorOption,
							GhostCopyPushConstants pushConstants, uint32_t sliceCount);
		void BufferTransfer(DeviceContext& sourceGpu, vk::CommandPool sourceCommandPool, vk::Fence sourceFence,
							ComputePipeline& sourceExtractPipeline, uint32_t sourceDescriptorOption,
							glm::ivec3 sourceNumBlocks, int32_t sourceZStart,
							DeviceContext& destGpu, vk::CommandPool destCommandPool, vk::Fence destFence,
							ComputePipeline& destInsertPipeline, uint32_t destDescriptorOption,
							glm::ivec3 destNumBlocks, int32_t destZStart);

		// === Image mode resources ===
		std::shared_ptr<Image> _gpu0CellImages[2];
		std::shared_ptr<Image> _gpu1CellImages[2];

		// Copies ghostDepth Z-slices from source image to dest image via host staging.
		void ImageTransfer(DeviceContext& sourceGpu, vk::CommandPool sourceCommandPool, vk::Fence sourceFence,
						   std::shared_ptr<Image> sourceImage, int32_t sourceZStart,
						   DeviceContext& destGpu, vk::CommandPool destCommandPool, vk::Fence destFence,
						   std::shared_ptr<Image> destImage, int32_t destZStart);
	};

} // namespace Cave
