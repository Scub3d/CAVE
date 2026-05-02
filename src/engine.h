#pragma once

#include <memory>
#include <string>
#include <cstdint>

#include "vulkan/vulkanInstance.h"
#include "vulkan/systems/presentSystem.h"
#include "vulkan/systems/guiSystem.h"
#include "common/applicationMode.h"
#include "common/queryManager.h"
#include "common/startupConfig.h"
#include "modes/applicationModeHandler.h"

namespace Cave
{
	class Engine
	{
	private: // Variables
		// Always-alive systems
		VulkanInstance _vulkanInstance;
		std::unique_ptr<DeviceContext> _gpu0;
		std::unique_ptr<DeviceContext> _gpu1; // nullptr when single-GPU
		std::unique_ptr<PresentSystem> _presentSystem;
		std::unique_ptr<GuiSystem> _guiSystem;

		// Active mode handler (polymorphic — RenderingMode, SearchMode, VideoEncodingMode)
		std::unique_ptr<ApplicationModeHandler> _activeMode;
		ApplicationMode _currentMode = ApplicationMode::None;
		std::unique_ptr<ModeServices> _modeServices;

		std::shared_ptr<QueryManager> _queryManager = std::make_shared<QueryManager>();

		bool _headless = false;
		bool _applicationRunning = true;

		uint32_t _currentFrameIndex = 0;
		uint32_t _framesInFlight = 2;

		double _currentTime = 0;
		double _lastTime = 0;
		double _previousFrameTime = 0;
		int _numberOfFrames = 0;
		float _frameTime = 0;

		const StartupConfig* _startupConfig = nullptr;
		bool _autoStartPending = false;

	private: // Methods
		void CalculateFrameRate();

		void EnterMode(ApplicationMode mode);
		void ExitMode();
		void IdleModeFrame();

	public:
		void Init(const StartupConfig* config = nullptr);
		void Run();
		void Cleanup();
		ModeServices& GetModeServices() { return *_modeServices; }
	};

} // namespace Cave
