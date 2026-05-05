#pragma once

#include <vulkan/vulkan.hpp>

#include <memory>
#include <vector>
#include <sstream>
#include <string>

#include "../common/structs.h"
#include "../common/startupConfig.h"
#include "deviceContext.h"
#include "buffer.h"
#include "image.h"
#include "slangCompiler.h"
#include "shader.h"
#include "../simulation/simulation.h"

namespace Cave
{
	// Owns all Vulkan GPU resources required to render a Simulation.
	// Reads simulation state; never modifies it.
	class VulkanSimulationRenderer
	{
	private:
		DeviceContext &_deviceContext;
		Simulation &_simulation;
		uint32_t _framesInFlight;
		std::unique_ptr<SlangCompiler> _slangCompiler;
		CellStateStorageMode _storageMode;

		// Domain decomposition (null when single-GPU)
		std::unique_ptr<DomainConfig> _domainConfig;
		uint32_t _localZSize = 0;       // = gridZ when single-GPU, ghostDepth + ownedZ + ghostDepth when decomposed
		uint32_t _ownedZStart = 0;      // first owned Z index in local coordinates (0 or ghostDepth)
		uint32_t _ownedZEnd = 0;        // one past last owned Z in local coordinates (gridZ or ghostDepth + ownedZ)
		uint32_t _zDomainOffset = 0;    // global Z of first owned cell (0 when single-GPU)
		uint32_t _ghostDepth = 0;       // ghost slices per side (0 when single-GPU)

		// Storage buffers for cell state (tiled Morton layout, one per frame in flight, ping-pong)
		std::vector<std::shared_ptr<Buffer>> _cellStateBuffers;
		uint32_t _cellStateBufferElementCount = 0; // total uint elements in tiled layout

		// Skip grid storage buffers (tiled Morton layout, ping-pong pair)
		std::vector<std::shared_ptr<Buffer>> _skipGridBuffers;
		uint32_t _skipGridBufferElementCount = 0;

		// 3D storage images for ray marching (one per frame in flight, ping-pong)
		std::vector<std::shared_ptr<Image>> _cellStateImages;
		vk::Format _cellStateImageFormat = vk::Format::eUndefined;

		// Skip grid images (ping-pong pair)
		std::vector<std::shared_ptr<Image>> _skipGridImages;

		// Shared dimensions
		uint32_t _cellStateImageWidth = 0; // ceil(gridX/2) for packed 4-bit format
		uint32_t _skipGridChunkSize = 8;
		uint32_t _skipGridDimensionX = 0;
		uint32_t _skipGridDimensionY = 0;
		uint32_t _skipGridDimensionZ = 0;

		// Shader modules (generated from simulation parameters)
		vk::ShaderModule _rayMarchInitShaderModule;
		vk::ShaderModule _rayMarchComputeShaderModule;
		vk::ShaderModule _rayMarchVertexShaderModule;
		vk::ShaderModule _rayMarchFragmentShaderModule;

		std::string _rayMarchComputeShaderCode;

	private:
		void InitBuffers();
		void InitCellStateBuffersOnGPU();
		void InitImages();
		void InitCellStateImagesOnGPU();
		void GenerateShaders();
		void GenerateRayMarchComputeShader();
		void GenerateRayMarchVertexShader();
		void GenerateRayMarchFragmentShader();


		public:
		VulkanSimulationRenderer(DeviceContext &deviceContext, Simulation &simulation, uint32_t framesInFlight,
								 CellStateStorageMode storageMode, const DomainConfig* domainConfig = nullptr);
		~VulkanSimulationRenderer() = default;

		VulkanSimulationRenderer(const VulkanSimulationRenderer &) = delete;
		VulkanSimulationRenderer &operator=(const VulkanSimulationRenderer &) = delete;

		// Resets for a new simulation with the same grid dimensions.
		// Keeps GPU buffers, regenerates shaders and reinitializes cell state.
		void Reset();

		const glm::uvec4* GetGridDimensions() const { return _simulation.GetDimensions(); }
		const SimulationParameters& GetSimulationParameters() const { return *_simulation.GetSimulationParameters(); }
		uint32_t GetFramesInFlight() const { return _framesInFlight; }

		// Domain decomposition queries
		bool IsDomainDecomposed() const { return _domainConfig != nullptr; }
		uint32_t GetLocalZSize() const { return _localZSize; }
		uint32_t GetOwnedZSize() const { return _ownedZEnd - _ownedZStart; }
		uint32_t GetOwnedZStart() const { return _ownedZStart; }
		uint32_t GetOwnedZEnd() const { return _ownedZEnd; }
		uint32_t GetZDomainOffset() const { return _zDomainOffset; }
		uint32_t GetGhostDepth() const { return _ghostDepth; }

		vk::ShaderModule GetRayMarchComputeShaderModule() const { return _rayMarchComputeShaderModule; }
		vk::ShaderModule GetRayMarchVertexShaderModule() const { return _rayMarchVertexShaderModule; }
		vk::ShaderModule GetRayMarchFragmentShaderModule() const { return _rayMarchFragmentShaderModule; }

		// Storage mode query
		bool UsesImages() const { return _storageMode == CellStateStorageMode::Image; }
		CellStateStorageMode GetStorageMode() const { return _storageMode; }

		// Cell state buffer access (buffer mode)
		std::shared_ptr<Buffer> GetCellStateBuffer(uint32_t frameIndex) { return _cellStateBuffers[frameIndex]; }
		uint32_t GetCellStateImageWidth() const { return _cellStateImageWidth; }

		// Cell state image access (image mode)
		std::shared_ptr<Image> GetCellStateImage(uint32_t frameIndex) { return _cellStateImages[frameIndex]; }
		vk::Format GetCellStateImageFormat() const { return _cellStateImageFormat; }

		// Skip grid buffer access (buffer mode)
		std::shared_ptr<Buffer> GetSkipGridBuffer(uint32_t frameIndex) { return _skipGridBuffers[frameIndex]; }

		// Skip grid image access (image mode)
		std::shared_ptr<Image> GetSkipGridImage(uint32_t frameIndex) { return _skipGridImages[frameIndex]; }

		uint32_t GetSkipGridChunkSize() const { return _skipGridChunkSize; }
		glm::uvec3 GetSkipGridDimensions() const { return glm::uvec3(_skipGridDimensionX, _skipGridDimensionY, _skipGridDimensionZ); }

		// Tiled block counts for the cell state buffer (needed by GhostExchange)
		glm::ivec3 GetCellStateNumBlocks() const
		{
			return glm::ivec3(
				(_cellStateImageWidth + 7) / 8,
				(_simulation.GetDimensions()->y + 7) / 8,
				(_localZSize + 7) / 8);
		}
	};
}
