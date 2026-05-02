#pragma once

#include "applicationModeHandler.h"
#include "../simulation/simulationContext.h"
#include "../vulkan/systems/videoEncoderSystem.h"
#include "../vulkan/systems/rayMarchRenderSystem.h"
#include "../vulkan/ghostExchange.h"
#include "../vulkan/dualGpuCompositor.h"

#include <memory>
#include <future>
#include <vector>
#include <string>
#include <cstdint>

namespace Cave
{
	class VideoEncodingMode : public ApplicationModeHandler
	{
	public:
		VideoEncodingMode() = default;
		~VideoEncodingMode() override;

		VideoEncodingMode(const VideoEncodingMode&) = delete;
		VideoEncodingMode& operator=(const VideoEncodingMode&) = delete;

		void Enter(ModeServices& services) override;
		bool OnFrame(ModeServices& services, float deltaTimeInSeconds) override;
		void Exit(ModeServices& services) override;

	private:
		enum class State { Configuring, Loading, Encoding, Complete };
		State _state = State::Configuring;

		SimulationContext _simulationContext;
		std::unique_ptr<VideoEncoderSystem> _videoEncoderSystem;
		std::future<Simulation> _simulationFuture;

		// Dual-GPU domain decomposition resources
		bool _dualGpuEnabled = false;
		SimulationContext _gpu1SimulationContext;
		std::unique_ptr<RayMarchRenderSystem> _gpu0RayMarchRenderSystem;
		std::unique_ptr<RayMarchRenderSystem> _gpu1RayMarchRenderSystem;
		std::unique_ptr<GhostExchange> _ghostExchange;
		std::unique_ptr<DualGpuCompositor> _compositor;
		uint32_t _ghostDepth = 0;
		uint32_t _zMid = 0;

		uint32_t _videoCurrentTick = 0;
		uint32_t _videoTotalTicks = 0;
		std::vector<char> _videoAccumulatedBitstream;
		struct VideoEncodingJob { SimulationParameters params; };
		std::vector<VideoEncodingJob> _videoBatchQueue;
		uint32_t _videoCurrentJobIndex = 0;

		bool _sessionRunning = false;
		bool _enableComputeSkip = true;
		uint32_t _currentFrameIndex = 0;

		void RetrieveTimestamps(ModeServices& services);
		void BuildGui(ModeServices& services);

		// Builds the OverlayInfo to bake into encoded frames from the active simulation
		// + GuiState. Called once per encoded tick before invoking the encoder. Non-const
		// because Simulation::GetSimulationParameters and GetDimensions return non-const.
		OverlayInfo BuildOverlayInfo(const GuiState& guiState);
	};

} // namespace Cave
