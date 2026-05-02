#include "engine.h"
#include "common/logger.h"
#include "modes/renderingMode.h"
#include "modes/searchMode.h"
#include "modes/videoEncodingMode.h"

#include <glm/glm.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <GLFW/glfw3.h>

namespace Cave
{

	void Engine::Init(const StartupConfig* config)
	{
		_startupConfig = config;
		if (_startupConfig)
		{
			_vulkanInstance.SetHeadless(_startupConfig->headless || !_startupConfig->testName.empty());
			_vulkanInstance.SetForcedGpuIndex(_startupConfig->forcedGpuIndex);
		}
		_vulkanInstance.Initialize();
		_gpu0 = _vulkanInstance.CreateDeviceContext();
		_gpu0->FinishSetup(_vulkanInstance.GetFramesInFlight());

		_framesInFlight = _gpu0->GetFramesInFlight();
		_queryManager->SetTimestampPeriod(_gpu0->GetTimestampPeriod());

		// Create secondary GPU if --dual-gpu requested and a second suitable device exists
		if (_startupConfig && _startupConfig->dualGpu)
		{
			auto availableDevices = _vulkanInstance.GetAvailablePhysicalDevices();
			vk::PhysicalDevice selectedDevice = _vulkanInstance.GetSelectedPhysicalDevice();
			DeviceContext deviceRater;

			for (auto& device : availableDevices)
			{
				if (device == selectedDevice)
					continue;
				if (deviceRater.RateDeviceSuitability(device) > 0)
				{
					_gpu1 = _vulkanInstance.CreateDeviceContext(device);
					_gpu1->FinishSetup(_vulkanInstance.GetFramesInFlight());
					vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
					LOG_INFO("Dual-GPU enabled: secondary GPU is {} ", std::string(deviceProperties.deviceName.data()));
					break;
				}
			}

			if (!_gpu1)
			{
				LOG_WARNING("--dual-gpu requested but no suitable second GPU found. Continuing with single GPU.");
			}
		}

		// Always-alive systems: present (swapchain) and GUI
		_presentSystem = std::make_unique<PresentSystem>(_vulkanInstance, *_gpu0);

		_guiSystem = std::make_unique<GuiSystem>(_vulkanInstance, *_gpu0, _presentSystem->GetSwapchainImageFormat());
		_presentSystem->SetGuiSystem(_guiSystem.get());

		_gpu0->GetDevice().waitIdle();

		_headless = _startupConfig && (_startupConfig->headless || !_startupConfig->testName.empty());

		_modeServices = std::make_unique<ModeServices>(ModeServices{
			_vulkanInstance,
			*_gpu0,
			_gpu1 ? _gpu1.get() : nullptr,
			*_presentSystem,
			*_guiSystem,
			_queryManager,
			_startupConfig,
			_headless,
			_framesInFlight
		});

		_currentMode = ApplicationMode::None;

		// If CLI args provided, pre-populate GUI state and auto-enter the mode
		if (_startupConfig && _startupConfig->autoStart)
		{
			GuiState& gs = _guiSystem->GetState();

			// Pre-populate shared params (grid/spawn set before EnterMode since it doesn't touch them)
			gs.gridDimensionX = gs.gridDimensionY = gs.gridDimensionZ = _startupConfig->gridSize;
			gs.spawnDimensionX = gs.spawnDimensionY = gs.spawnDimensionZ = _startupConfig->spawnSize;
			gs.spawnMode = _startupConfig->spawnMode;
			gs.spawnRandomDensity = _startupConfig->spawnRandomDensity;

			// Apply CLI rule overrides BEFORE mode-specific setup, since video encoding's
			// BuildSimulationParameters() reads these from the GUI state.
			gs.maxCellState = _startupConfig->maxCellState;
			gs.shape = _startupConfig->shape;
			gs.faceNeighbors = _startupConfig->neighborhood.find('F') != std::string::npos;
			gs.edgeNeighbors = _startupConfig->neighborhood.find('E') != std::string::npos;
			gs.cornerNeighbors = _startupConfig->neighborhood.find('C') != std::string::npos;
			gs.wrapAtBoundary = _startupConfig->neighborhood.find('W') != std::string::npos;
			if (!_startupConfig->birthRules.empty())
				gs.birthRulesText = _startupConfig->birthRules;
			if (!_startupConfig->survivalRules.empty())
				gs.survivalRulesText = _startupConfig->survivalRules;

			// Mode-specific params (pre-EnterMode)
			if (_startupConfig->mode == ApplicationMode::Search)
			{
				gs.searchMinMaxCellState = _startupConfig->minMaxCellState;
				gs.searchMaxMaxCellState = _startupConfig->maxMaxCellState;
				gs.searchMaxTicksToSurvive = _startupConfig->maxTicks;
				gs.searchGridDimensionX = gs.searchGridDimensionY = gs.searchGridDimensionZ = _startupConfig->gridSize;
				gs.searchSpawnDimensionX = gs.searchSpawnDimensionY = gs.searchSpawnDimensionZ = _startupConfig->spawnSize;
				gs.searchSpawnMode = _startupConfig->spawnMode;
				gs.searchSpawnRandomDensity = _startupConfig->spawnRandomDensity;
				gs.searchShape = _startupConfig->shape;
				gs.searchFaceNeighbors = _startupConfig->neighborhood.find('F') != std::string::npos;
				gs.searchEdgeNeighbors = _startupConfig->neighborhood.find('E') != std::string::npos;
				gs.searchCornerNeighbors = _startupConfig->neighborhood.find('C') != std::string::npos;
				gs.searchWrapAtBoundary = _startupConfig->neighborhood.find('W') != std::string::npos;
				gs.searchGridSizeSweepEnabled = _startupConfig->searchGridSizeSweepEnabled;
				gs.searchGridSizeSweepMin = _startupConfig->searchGridSizeSweepMin;
				gs.searchGridSizeSweepMax = _startupConfig->searchGridSizeSweepMax;
				gs.searchGridSizeSweepStep = _startupConfig->searchGridSizeSweepStep;
				gs.searchDensitySweepEnabled = _startupConfig->searchDensitySweepEnabled;
				gs.searchDensitySweepMin = _startupConfig->searchDensitySweepMin;
				gs.searchDensitySweepMax = _startupConfig->searchDensitySweepMax;
				gs.searchDensitySweepStep = _startupConfig->searchDensitySweepStep;
				gs.searchSeedSweepCount = _startupConfig->searchSeedSweepCount;
				gs.searchMaxRuleBits = _startupConfig->searchMaxRuleBits;
				gs.searchGpuVramBudgetGb = _startupConfig->searchGpuVramBudgetGb;
				gs.searchChunksPerConfig = _startupConfig->chunksPerConfig;
				gs.searchWorkgroupSize = _startupConfig->searchWorkgroupSize;
				if (!_startupConfig->searchOutputFolder.empty())
					gs.searchOutputFolderPath = _startupConfig->searchOutputFolder;
			}
			else if (_startupConfig->mode == ApplicationMode::VideoEncoding)
			{
				gs.videoSourceJsonPath = _startupConfig->jsonPath;
				gs.videoEncodeAll = _startupConfig->encodeAll;
				gs.videoSelectedPermutationIndex = _startupConfig->selectedIndex;
				gs.videoResolutionWidth = _startupConfig->videoWidth;
				gs.videoResolutionHeight = _startupConfig->videoHeight;
				gs.videoFps = _startupConfig->videoFps;
				gs.videoDurationTicks = _startupConfig->videoDurationTicks;
				if (_startupConfig->orbitSpeed >= 0.0f)
					gs.orbitAngularVelocity = _startupConfig->orbitSpeed;
				if (!_startupConfig->videoOutputFolder.empty())
					gs.videoOutputFolderPath = _startupConfig->videoOutputFolder;
				if (!_startupConfig->jsonPath.empty())
				{
					gs.videoSourceMode = 0;
					std::ifstream inputFile(_startupConfig->jsonPath);
					if (inputFile.is_open())
					{
						try
						{
							gs.videoImportedPermutations.clear();
							gs.videoImportedTicksSurvived.clear();

							auto ingestEntry = [&gs](const nlohmann::json& entry) {
								SimulationParameters params{};
								params.birthAndMaxCellStateRules = entry["birthAndMaxCellStateRules"].get<uint64_t>();
								params.survivalAndNeighborhoodRules = entry["survivalAndNeighborhoodRules"].get<uint64_t>();
								params.shapeAndGridConfiguration = entry.value("shapeAndGridConfiguration", uint64_t(0));
								gs.videoImportedPermutations.push_back(params);
								gs.videoImportedTicksSurvived.push_back(entry.value("ticksSurvived", uint32_t(0)));
							};

							// Peek to distinguish JSONL (one object per line) from legacy JSON array.
							int firstCharacter = inputFile.peek();
							while (firstCharacter != EOF && std::isspace(firstCharacter))
							{
								inputFile.get();
								firstCharacter = inputFile.peek();
							}

							if (firstCharacter == '[')
							{
								nlohmann::json jsonArray = nlohmann::json::parse(inputFile);
								for (const auto& entry : jsonArray) ingestEntry(entry);
							}
							else
							{
								std::string line;
								while (std::getline(inputFile, line))
								{
									if (line.empty()) continue;
									ingestEntry(nlohmann::json::parse(line));
								}
							}
							LOG_INFO("CLI pre-loaded {} permutations from {}", gs.videoImportedPermutations.size(), _startupConfig->jsonPath);
						}
						catch (const std::exception& e)
						{
							LOG_ERROR("Failed to parse JSON: {}", e.what());
						}
					}
				}
				else
				{
					SimulationParameters cliParams = _guiSystem->BuildSimulationParameters();
					cliParams.shapeAndGridConfiguration =
						(static_cast<uint64_t>(gs.gridDimensionX - 1) << 5) |
						(static_cast<uint64_t>(gs.gridDimensionY - 1) << 13) |
						(static_cast<uint64_t>(gs.gridDimensionZ - 1) << 21);
					gs.videoImportedPermutations.push_back(cliParams);
					gs.videoImportedTicksSurvived.push_back(0);
					gs.videoSourceMode = 0;
					gs.videoEncodeAll = true;
				}
			}

			gs.enableComputeSkip = _startupConfig->enableComputeSkip;
			_headless = _startupConfig->headless;

			EnterMode(_startupConfig->mode);
			_autoStartPending = true;

			// Re-apply CLI overrides AFTER EnterMode, since EnterMode populates GUI
			// with hardcoded defaults that would otherwise clobber CLI values.
			gs.maxCellState = _startupConfig->maxCellState;
			gs.shape = _startupConfig->shape;
			gs.searchShape = _startupConfig->shape;
			gs.faceNeighbors = _startupConfig->neighborhood.find('F') != std::string::npos;
			gs.edgeNeighbors = _startupConfig->neighborhood.find('E') != std::string::npos;
			gs.cornerNeighbors = _startupConfig->neighborhood.find('C') != std::string::npos;
			gs.wrapAtBoundary = _startupConfig->neighborhood.find('W') != std::string::npos;
			if (!_startupConfig->birthRules.empty())
				gs.birthRulesText = _startupConfig->birthRules;
			if (!_startupConfig->survivalRules.empty())
				gs.survivalRulesText = _startupConfig->survivalRules;

			LOG_INFO("CLI auto-start: mode={}, frames={}, autoStart=true",
				static_cast<int>(_startupConfig->mode), _startupConfig->maxFrames);
		}
	}

	void Engine::Run()
	{
		_previousFrameTime = glfwGetTime();

		while (_applicationRunning)
		{
			glfwPollEvents();
			if (glfwWindowShouldClose(_vulkanInstance.GetVulkanGLFWWindow()))
			{
				_applicationRunning = false;
				break;
			}

			double currentTime = glfwGetTime();
			float deltaTimeInSeconds = static_cast<float>(currentTime - _previousFrameTime);
			_previousFrameTime = currentTime;

			CalculateFrameRate();

			if (_activeMode)
			{
				bool continueRunning = _activeMode->OnFrame(*_modeServices, deltaTimeInSeconds);
				if (!continueRunning)
				{
					_applicationRunning = false;
					break;
				}
			}
			else
			{
				IdleModeFrame();
			}

			// Auto-exit after N simulation ticks (CLI --frames flag)
			uint32_t tickCount = _activeMode ? _activeMode->GetSimulationTickCount() : 0;
			if (_startupConfig && _startupConfig->maxFrames >= 0 &&
				tickCount >= static_cast<uint32_t>(_startupConfig->maxFrames))
			{
				LOG_INFO("Auto-exit: reached {} ticks (--frames {})", tickCount, _startupConfig->maxFrames);
				_applicationRunning = false;
				break;
			}

			// Auto-start from CLI args: inject Start/Encode request on the first frame
			if (_autoStartPending)
			{
				_autoStartPending = false;
				if (_currentMode == ApplicationMode::Rendering || _currentMode == ApplicationMode::Search)
					_guiSystem->InjectStartRequest();
				else if (_currentMode == ApplicationMode::VideoEncoding)
					_guiSystem->InjectEncodeRequest();

				LOG_INFO("CLI auto-start: injected start/encode request");
			}

			// Handle mode switching requests from GUI
			ApplicationMode requestedMode = _guiSystem->ConsumeEnterModeRequest();
			if (requestedMode != ApplicationMode::None)
				EnterMode(requestedMode);

			if (_guiSystem->ConsumeExitModeRequest())
				ExitMode();
		}

		_gpu0->GetDevice().waitIdle();
	}

	void Engine::EnterMode(ApplicationMode mode)
	{
		if (_currentMode != ApplicationMode::None)
			ExitMode();

		LOG_INFO("Entering mode: {}", static_cast<int>(mode));

		switch (mode)
		{
		case ApplicationMode::Rendering:
			_activeMode = std::make_unique<RenderingMode>();
			break;
		case ApplicationMode::Search:
			_activeMode = std::make_unique<SearchMode>();
			break;
		case ApplicationMode::VideoEncoding:
			_activeMode = std::make_unique<VideoEncodingMode>();
			break;
		default:
			return;
		}

		_activeMode->Enter(*_modeServices);
		_currentMode = mode;
	}

	void Engine::ExitMode()
	{
		LOG_INFO("Exiting mode: {}", static_cast<int>(_currentMode));
		_gpu0->GetDevice().waitIdle();

		if (_activeMode)
		{
			_activeMode->Exit(*_modeServices);
			_activeMode.reset();
		}

		_currentMode = ApplicationMode::None;
		_currentFrameIndex = 0;
	}

	void Engine::IdleModeFrame()
	{
		if (!_headless)
		{
			_guiSystem->BuildIdleFrame();
			_presentSystem->PresentGuiOnly(_currentFrameIndex);
		}
	}

	void Engine::CalculateFrameRate()
	{
		_currentTime = glfwGetTime();
		double delta = _currentTime - _lastTime;

		if (delta >= 1)
		{
			int framerate{std::max(1, int(_numberOfFrames / delta))};
			std::stringstream title;
			title << "Running at " << framerate << " fps.";

			glfwSetWindowTitle(_vulkanInstance.GetVulkanGLFWWindow(), title.str().c_str());
			_lastTime = _currentTime;
			_numberOfFrames = -1;
			_frameTime = float(1000.0 / framerate);
		}

		++_numberOfFrames;
	}

	void Engine::Cleanup()
	{
		if (_currentMode != ApplicationMode::None)
			ExitMode();

		_presentSystem->SetGuiSystem(nullptr);
		_guiSystem.reset();
		_presentSystem.reset();
		if (_gpu1)
			_gpu1->Cleanup();
		_gpu0->Cleanup();
	}

} // namespace Cave
