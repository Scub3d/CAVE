#pragma once

#include <vulkan/vulkan.hpp>
#include <memory>
#include <vector>

#include "../deviceContext.h"
#include "../simulationRenderer.h"
#include "../pipelines/computePipeline.h"
#include "../descriptors.h"
#include "../../common/queryManager.h"

namespace Cave
{
	// Runs the cellular automata simulation on the GPU each tick.
	// Writes to the cell state buffers and indirect draw command buffers
	// owned by VulkanSimulationRenderer. Signals a semaphore when done
	// so RenderSystem can wait before reading those buffers.
	class ComputeSystem
	{
	private:
		DeviceContext& _deviceContext;
		VulkanSimulationRenderer& _simulationRenderer;
		uint32_t _framesInFlight;

		vk::CommandPool _computeCommandPool;
		std::vector<vk::CommandBuffer> _computeCommandBuffers;

		std::unique_ptr<ComputePipeline> _rayMarchSimulationPipeline;
		std::shared_ptr<Descriptor> _rayMarchDescriptor;

		std::vector<vk::Fence> _inFlightFences;
		std::vector<vk::Semaphore> _completedSemaphores;
		std::vector<uint64_t> _completedSemaphoreSignalValues;

	private:
		void BuildDescriptors();
		void BuildPipeline();
		void BuildCommandBuffers();

	public:
		ComputeSystem(DeviceContext& deviceContext, VulkanSimulationRenderer& simulationRenderer);
		~ComputeSystem();

		ComputeSystem(const ComputeSystem&) = delete;
		ComputeSystem& operator=(const ComputeSystem&) = delete;

		// Rebuilds the compute pipeline after shader regeneration.
		void RebuildPipeline();

		// Dispatches the simulation compute shader for the given frame.
		// Blocks until the previous compute and render on this frame slot are complete,
		// then signals GetCompletedSemaphore(frameIndex) when done.
		void Tick(uint32_t frameIndex, vk::Fence renderFence, bool dispatchCompute = true,
		         bool enableComputeSkip = true,
		         std::shared_ptr<QueryManager> queryManager = nullptr, vk::QueryPool queryPool = {});

		vk::Semaphore GetCompletedSemaphore(uint32_t frameIndex) const { return _completedSemaphores[frameIndex]; }
		uint64_t GetCompletedSemaphoreSignalValue(uint32_t frameIndex) const { return _completedSemaphoreSignalValues[frameIndex]; }
		vk::Fence GetInFlightFence(uint32_t frameIndex) const { return _inFlightFences[frameIndex]; }
	};
}
