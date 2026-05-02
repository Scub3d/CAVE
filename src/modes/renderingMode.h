#pragma once

#include "applicationModeHandler.h"
#include "../simulation/simulationContext.h"
#include "../vulkan/systems/rayMarchRenderSystem.h"
#include "../vulkan/ghostExchange.h"
#include "../vulkan/dualGpuCompositor.h"

#include <memory>
#include <future>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace Cave
{

	class RenderingMode : public ApplicationModeHandler
	{
	public:
		RenderingMode() = default;
		~RenderingMode() override;

		RenderingMode(const RenderingMode&) = delete;
		RenderingMode& operator=(const RenderingMode&) = delete;

		void Enter(ModeServices& services) override;
		bool OnFrame(ModeServices& services, float deltaTimeInSeconds) override;
		void Exit(ModeServices& services) override;
		uint32_t GetSimulationTickCount() const override { return _simulationTickCount; }

		// Expose raw timings for test runner access
		const std::unordered_map<std::string, float>& GetLastRawTimings() const { return _lastRawTimings; }

	private:
		enum class State { Configuring, Loading, Running, Paused };
		State _state = State::Configuring;

		// Primary GPU simulation context (always used)
		SimulationContext _simulationContext;
		std::unique_ptr<RayMarchRenderSystem> _rayMarchRenderSystem;
		std::future<Simulation> _simulationFuture;

		// Secondary GPU resources (only when --dual-gpu and rendering mode)
		bool _dualGpuEnabled = false;
		SimulationContext _gpu1SimulationContext;
		std::unique_ptr<RayMarchRenderSystem> _gpu1RayMarchRenderSystem;
		std::unique_ptr<GhostExchange> _ghostExchange;
		std::unique_ptr<DualGpuCompositor> _compositor;
		uint32_t _ghostDepth = 0;
		uint32_t _zMid = 0;

		bool _sessionRunning = false;
		bool _simulationHasStarted = false;
		bool _enableComputeSkip = true;
		bool _stepOneTick = false;
		uint32_t _currentFrameIndex = 0;
		uint32_t _simulationTickCount = 0;

		std::unordered_map<std::string, float> _lastRawTimings;

		void StartSimulationAsync(ModeServices& services);
		void FinalizeSimulationInit(ModeServices& services);
		void InitializeGuiStateFromSimulation(ModeServices& services);
		void DestroyRenderingModeSystems();
		void RetrieveTimestamps(ModeServices& services);
		void BuildGui(ModeServices& services);
	};

} // namespace Cave
