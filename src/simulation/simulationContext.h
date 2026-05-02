#pragma once

#include <memory>

#include "simulation.h"
#include "../common/structs.h"
#include "../vulkan/simulationRenderer.h"
#include "../vulkan/systems/computeSystem.h"
#include "../camera.h"

namespace Cave
{
	class VulkanInstance;
	struct GuiState;

	struct SimulationContext
	{
		Simulation simulation;
		std::unique_ptr<VulkanSimulationRenderer> simulationRenderer;
		std::unique_ptr<ComputeSystem> computeSystem;
		std::unique_ptr<Camera> camera;

		SimulationContext() = default;
		~SimulationContext();

		SimulationContext(SimulationContext&&) noexcept;
		SimulationContext& operator=(SimulationContext&&) noexcept;

		SimulationContext(const SimulationContext&) = delete;
		SimulationContext& operator=(const SimulationContext&) = delete;

		// Destroys GPU resources (compute, renderer) but keeps camera alive.
		// Camera is intentionally preserved across resets.
		void DestroyGpuResources();

		// Builds RayMarchPushConstants from simulation state + GUI state.
		// Consolidates the duplicated push constant construction across modes.
		RayMarchPushConstants BuildRayMarchPushConstants(const GuiState& guiState);

		// Builds CameraData with lighting parameters from GUI state.
		CameraData BuildCameraData(const GuiState& guiState);
	};

} // namespace Cave
