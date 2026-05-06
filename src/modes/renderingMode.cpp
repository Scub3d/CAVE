#include "renderingMode.h"

#include "../vulkan/vulkanInstance.h"
#include "../vulkan/simulationRenderer.h"
#include "../vulkan/systems/computeSystem.h"
#include "../vulkan/systems/rayMarchRenderSystem.h"
#include "../vulkan/systems/presentSystem.h"
#include "../vulkan/systems/guiSystem.h"
#include "../vulkan/frameCaptureUtility.h"
#include "../common/queryManager.h"
#include "../common/startupConfig.h"
#include "../common/logger.h"
#include "../camera.h"
#include "../shapes/cube.h"
#include "../shapes/elongatedRhombicDodecahedron.h"

#include <algorithm>
#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Cave
{

	RenderingMode::~RenderingMode() = default;

	void RenderingMode::Enter(ModeServices& services)
	{
		auto extent = services.vulkanInstance.GetWindowExtent();
		float aspectRatio = static_cast<float>(extent.width) / static_cast<float>(extent.height);
		_simulationContext.camera = std::make_unique<Camera>(
			services.vulkanInstance.GetVulkanGLFWWindow(), aspectRatio);

		// Apply CLI camera overrides. When any orbit pose is fixed, freeze the angular
		// velocity by default so the camera stays at the user-specified angle.
		if (services.startupConfig)
		{
			const StartupConfig& startupConfig = *services.startupConfig;
			bool fixedPoseRequested = false;
			if (startupConfig.orbitAngle < 1e29f)
			{
				_simulationContext.camera->SetOrbitAngle(startupConfig.orbitAngle);
				fixedPoseRequested = true;
			}
			if (startupConfig.orbitElevation < 1e29f)
			{
				_simulationContext.camera->SetOrbitElevation(startupConfig.orbitElevation);
				fixedPoseRequested = true;
			}
			if (startupConfig.orbitRadius > 0.0f)
			{
				_simulationContext.camera->SetOrbitRadius(startupConfig.orbitRadius);
				fixedPoseRequested = true;
			}
			if (startupConfig.orbitSpeed >= 0.0f)
				_simulationContext.camera->SetOrbitAngularVelocity(startupConfig.orbitSpeed);
			else if (fixedPoseRequested)
				_simulationContext.camera->SetOrbitAngularVelocity(0.0f);
		}

		// Populate GUI state with sensible defaults before the user clicks Start
		GuiState& guiState = services.guiSystem.GetState();
		guiState.birthRulesText = "1,4-6";
		guiState.survivalRulesText = "4";
		guiState.maxCellState = 5;
		guiState.faceNeighbors = true;
		guiState.edgeNeighbors = false;
		guiState.cornerNeighbors = false;
		guiState.wrapAtBoundary = false;

		LOG_INFO("Default rules — birth: '{}', survival: '{}', maxState: {}",
			guiState.birthRulesText, guiState.survivalRulesText, guiState.maxCellState);

		_state = State::Configuring;
		_sessionRunning = false;
		_simulationHasStarted = false;
		_currentFrameIndex = 0;
		_simulationTickCount = 0;
		guiState.simulationTickCount = 0;
	}

	bool RenderingMode::OnFrame(ModeServices& services, float deltaTimeInSeconds)
	{
		if (_simulationContext.camera)
			_simulationContext.camera->Update(deltaTimeInSeconds);

		bool isLoading = (_state == State::Loading);

		switch (_state)
		{
		case State::Configuring:
		{
			if (!services.headless)
			{
				BuildGui(services);
				services.presentSystem.PresentGuiOnly(_currentFrameIndex);
			}

			if (services.guiSystem.ConsumeStartRequest() || services.guiSystem.ConsumeResetRequest())
			{
				StartSimulationAsync(services);
			}
			break;
		}

		case State::Loading:
		{
			if (!services.headless)
			{
				BuildGui(services);
				services.presentSystem.PresentGuiOnly(_currentFrameIndex);
			}

			if (_simulationFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
			{
				FinalizeSimulationInit(services);
			}
			break;
		}

		case State::Paused:
		{
			const GuiState& guiState = services.guiSystem.GetState();
			CameraData cameraData = _simulationContext.BuildCameraData(guiState);
			RayMarchPushConstants pushConstants = _simulationContext.BuildRayMarchPushConstants(guiState);

			_rayMarchRenderSystem->RenderFrame(_currentFrameIndex,
				{}, 0, cameraData, pushConstants, nullptr, {}, false);

			if (!services.headless)
			{
				BuildGui(services);
				services.presentSystem.PresentPausedFrame(
					_currentFrameIndex,
					_rayMarchRenderSystem->GetOutputImage(_currentFrameIndex),
					_rayMarchRenderSystem->GetRenderExtent());
			}

			if (services.guiSystem.ConsumeStartRequest())
			{
				services.deviceContext.GetDevice().waitIdle();
				_sessionRunning = true;
				_simulationHasStarted = true;
				_state = State::Running;
			}
			if (services.guiSystem.ConsumeStepRequest())
			{
				services.deviceContext.GetDevice().waitIdle();
				_sessionRunning = true;
				_simulationHasStarted = true;
				_state = State::Running;
				_stepOneTick = true;
			}
			if (services.guiSystem.ConsumeResetRequest())
			{
				services.deviceContext.GetDevice().waitIdle();
				_simulationTickCount = 0;
				_simulationHasStarted = false;
				services.guiSystem.GetState().simulationTickCount = 0;
				DestroyRenderingModeSystems();
				StartSimulationAsync(services);
			}
			break;
		}

		case State::Running:
		{
			_enableComputeSkip = services.guiSystem.GetState().enableComputeSkip;

			vk::Fence renderFence = _rayMarchRenderSystem->GetInFlightFence(_currentFrameIndex);

			// GPU0 compute dispatch
			_simulationContext.computeSystem->Tick(_currentFrameIndex, renderFence, _sessionRunning,
				_enableComputeSkip, services.queryManager, services.deviceContext.GetQueryPool());

			if (_dualGpuEnabled && _sessionRunning)
			{
				// GPU1 compute dispatch (no render fence — GPU1 has no render system)
				_gpu1SimulationContext.computeSystem->Tick(_currentFrameIndex, {}, true,
					_enableComputeSkip);

				// Wait for both computes to finish before ghost exchange
				auto gpu0WaitResult = services.deviceContext.GetDevice().waitForFences(
					_simulationContext.computeSystem->GetInFlightFence(_currentFrameIndex), vk::True, UINT64_MAX);
				auto gpu1WaitResult = services.secondaryDeviceContext->GetDevice().waitForFences(
					_gpu1SimulationContext.computeSystem->GetInFlightFence(_currentFrameIndex), vk::True, UINT64_MAX);
				(void)gpu0WaitResult;
				(void)gpu1WaitResult;

				// Ghost exchange on the output buffer (next frame in ping-pong)
				uint32_t nextFrame = (_currentFrameIndex + 1) % services.framesInFlight;
				_ghostExchange->Exchange(nextFrame);
			}

			if (!services.headless)
				services.presentSystem.WaitForPreviousBlit(_currentFrameIndex);

			const GuiState& guiState = services.guiSystem.GetState();
			CameraData cameraData = _simulationContext.BuildCameraData(guiState);

			vk::Semaphore computeSemaphore = _simulationContext.computeSystem->GetCompletedSemaphore(_currentFrameIndex);
			uint64_t computeSemaphoreValue = _simulationContext.computeSystem->GetCompletedSemaphoreSignalValue(_currentFrameIndex);

			RayMarchPushConstants rayMarchPushConstants = _simulationContext.BuildRayMarchPushConstants(guiState);

			_rayMarchRenderSystem->RenderFrame(_currentFrameIndex,
				computeSemaphore, computeSemaphoreValue, cameraData, rayMarchPushConstants,
				services.queryManager, services.deviceContext.GetQueryPool());

			// Dual-GPU: render GPU1's domain and composite both halves
			vk::Image presentImage;
			vk::Extent2D presentExtent = _rayMarchRenderSystem->GetRenderExtent();
			if (_dualGpuEnabled && _gpu1RayMarchRenderSystem && _compositor)
			{
				// GPU1 render (same camera, different domain push constants)
				RayMarchPushConstants gpu1PushConstants = _gpu1SimulationContext.BuildRayMarchPushConstants(guiState);
				vk::Semaphore gpu1ComputeSemaphore = _gpu1SimulationContext.computeSystem->GetCompletedSemaphore(_currentFrameIndex);
				uint64_t gpu1ComputeSemaphoreValue = _gpu1SimulationContext.computeSystem->GetCompletedSemaphoreSignalValue(_currentFrameIndex);
				_gpu1RayMarchRenderSystem->RenderFrame(_currentFrameIndex,
					gpu1ComputeSemaphore, gpu1ComputeSemaphoreValue, cameraData, gpu1PushConstants);

				// Wait for both renders to complete
				services.deviceContext.GetDevice().waitForFences(
					_rayMarchRenderSystem->GetInFlightFence(_currentFrameIndex), vk::True, UINT64_MAX);
				services.secondaryDeviceContext->GetDevice().waitForFences(
					_gpu1RayMarchRenderSystem->GetInFlightFence(_currentFrameIndex), vk::True, UINT64_MAX);

				// Composite: transfer GPU1 image to GPU0, depth-compare per pixel
				presentImage = _compositor->Composite(_currentFrameIndex,
					_gpu1RayMarchRenderSystem->GetColorAttachment(_currentFrameIndex));
			}
			else
			{
				presentImage = _rayMarchRenderSystem->GetOutputImage(_currentFrameIndex);
			}

			if (!services.headless)
			{
				BuildGui(services);

				services.presentSystem.PresentFrame(
					_currentFrameIndex,
					_rayMarchRenderSystem->GetRenderFinishedSemaphore(_currentFrameIndex),
					_rayMarchRenderSystem->GetRenderFinishedSemaphoreSignalValue(_currentFrameIndex),
					presentImage,
					presentExtent,
					services.queryManager, services.deviceContext.GetQueryPool());
			}
			else
			{
				services.deviceContext.GetDevice().waitForFences(
					_rayMarchRenderSystem->GetInFlightFence(_currentFrameIndex), vk::True, UINT64_MAX);
			}

			RetrieveTimestamps(services);

			if (_sessionRunning)
			{
				_currentFrameIndex = (_currentFrameIndex + 1) % services.framesInFlight;
				_simulationTickCount++;
				services.guiSystem.GetState().simulationTickCount = _simulationTickCount;

				if (_stepOneTick)
				{
					_stepOneTick = false;
					_sessionRunning = false;
					_state = State::Paused;
				}

				// Frame capture at target tick (CLI --capture-frame flag)
				if (services.startupConfig && services.startupConfig->captureFrame >= 0 &&
					_simulationTickCount == static_cast<uint32_t>(services.startupConfig->captureFrame))
				{
					services.deviceContext.GetDevice().waitIdle();
					if (_dualGpuEnabled && services.secondaryDeviceContext)
						services.secondaryDeviceContext->GetDevice().waitIdle();

					uint32_t capturedFrameIndex = (_currentFrameIndex + services.framesInFlight - 1) % services.framesInFlight;

					// Dual-GPU captures the composited output (always R16G16B16A16Sfloat).
					// Single-GPU captures GPU0's color attachment (device default color format).
					vk::Image outputImage;
					vk::Format capturedFormat;
					if (_dualGpuEnabled && _compositor)
					{
						outputImage = _compositor->GetOutputImage(capturedFrameIndex);
						capturedFormat = _rayMarchRenderSystem->GetColorFormat();
					}
					else
					{
						outputImage = _rayMarchRenderSystem->GetOutputImage(capturedFrameIndex);
						capturedFormat = _rayMarchRenderSystem->GetColorFormat();
					}

					vk::Extent2D renderExtent = _rayMarchRenderSystem->GetRenderExtent();

					CaptureFrameToPNG(services.deviceContext, outputImage, renderExtent,
						capturedFormat, services.startupConfig->captureOutputPath,
						vk::ImageLayout::eTransferSrcOptimal);
				}
			}

			// Handle GUI requests
			if (services.guiSystem.ConsumeStopRequest())
			{
				LOG_INFO("Simulation paused at tick {}", _simulationTickCount);
				services.deviceContext.GetDevice().waitIdle();
				if (_dualGpuEnabled && services.secondaryDeviceContext)
					services.secondaryDeviceContext->GetDevice().waitIdle();
				_currentFrameIndex = 0;
				_sessionRunning = false;
				_state = State::Paused;
			}

			if (services.guiSystem.ConsumeStartRequest())
			{
				_sessionRunning = true;
				_simulationHasStarted = true;
				_state = State::Running;
			}

			if (services.guiSystem.ConsumeStepRequest())
			{
				_sessionRunning = true;
				_simulationHasStarted = true;
				_state = State::Running;
				_stepOneTick = true;
			}

			if (services.guiSystem.ConsumeResetRequest())
			{
				services.deviceContext.GetDevice().waitIdle();
				if (_dualGpuEnabled && services.secondaryDeviceContext)
					services.secondaryDeviceContext->GetDevice().waitIdle();
				_simulationTickCount = 0;
				_simulationHasStarted = false;
				services.guiSystem.GetState().simulationTickCount = 0;
				DestroyRenderingModeSystems();
				StartSimulationAsync(services);
			}

			break;
		}
		}

		return true;
	}

	void RenderingMode::Exit(ModeServices& services)
	{
		if (_dualGpuEnabled && services.secondaryDeviceContext)
			services.secondaryDeviceContext->GetDevice().waitIdle();
		DestroyRenderingModeSystems();
		_simulationContext.camera.reset();
	}

	static std::shared_ptr<Shape> CreateShapeFromGuiState(const GuiState& guiState)
	{
		switch (guiState.shape)
		{
		case 0:  return Cube::Create();
		case 1:  return ElongatedRhombicDodecahedron::Create();
		default: return Cube::Create();
		}
	}

	void RenderingMode::StartSimulationAsync(ModeServices& services)
	{
		const GuiState& guiState = services.guiSystem.GetState();

		// A simulation with zero neighborhood types is degenerate — no cells can be born or survive.
		// Reject here rather than let the user watch a silently-dead simulation.
		if (!guiState.faceNeighbors && !guiState.edgeNeighbors && !guiState.cornerNeighbors)
		{
			LOG_ERROR("Cannot start simulation: at least one of Faces / Edges / Corners must be selected.");
			return;
		}

		glm::uvec3 gridDimensions(guiState.gridDimensionX, guiState.gridDimensionY, guiState.gridDimensionZ);
		glm::uvec3 spawnDimensions(guiState.spawnDimensionX, guiState.spawnDimensionY, guiState.spawnDimensionZ);
		int spawnMode = guiState.spawnMode;
		float spawnRandomDensity = guiState.spawnRandomDensity;
		ColorRules colorRules = services.guiSystem.BuildColorRules();
		std::shared_ptr<Shape> shape = CreateShapeFromGuiState(guiState);

		SimulationParameters simulationParameters;
		if (guiState.rulesInputMode == 1)
		{
			simulationParameters.birthAndMaxCellStateRules = strtoull(guiState.rawBirthAndMaxCellStateRulesInput, nullptr, 10);
			simulationParameters.survivalAndNeighborhoodRules = strtoull(guiState.rawSurvivalAndNeighborhoodRulesInput, nullptr, 10);
			simulationParameters.shapeAndGridConfiguration = 0;
		}
		else
		{
			simulationParameters = services.guiSystem.BuildSimulationParameters();
		}

		LOG_INFO("Starting async simulation construction ({} x {} x {}), shape: {}, birth: {}, survival: {}",
			gridDimensions.x, gridDimensions.y, gridDimensions.z,
			shape->GetName(),
			simulationParameters.birthAndMaxCellStateRules,
			simulationParameters.survivalAndNeighborhoodRules);

		_simulationFuture = std::async(std::launch::async,
			[gridDimensions, spawnDimensions, spawnMode, simulationParameters, colorRules, shape, spawnRandomDensity]()
			{
				return Simulation(gridDimensions, spawnDimensions, spawnMode,
								 simulationParameters, colorRules, shape, spawnRandomDensity);
			});

		_state = State::Loading;
	}

	void RenderingMode::FinalizeSimulationInit(ModeServices& services)
	{
		_simulationContext.simulation = _simulationFuture.get();

		LOG_INFO("Simulation constructed. Creating GPU resources...");

		const glm::uvec4* globalDimensions = _simulationContext.simulation.GetDimensions();
		CellStateStorageMode storageMode = services.startupConfig
			? services.startupConfig->storageMode : CellStateStorageMode::Image;

		_dualGpuEnabled = services.secondaryDeviceContext != nullptr;

		if (_dualGpuEnabled)
		{

			// Compute ghost depth from max Z neighbor reach
			const auto& neighborDeltaConfigurations = _simulationContext.simulation.GetNeighborDeltaConfigurations();
			_ghostDepth = 0;
			for (const auto& configurationDeltas : neighborDeltaConfigurations)
				for (const auto& delta : configurationDeltas)
					_ghostDepth = std::max(_ghostDepth, static_cast<uint32_t>(std::abs(delta.z)));
			if (_ghostDepth == 0) _ghostDepth = 1;

			// Split Z in half
			_zMid = globalDimensions->z / 2;

			bool wrapEnabled = (_simulationContext.simulation.GetSimulationParameters()->survivalAndNeighborhoodRules
				>> Simulation::BIT_SHIFT) & Simulation::WRAP_NEIGHBORS_MASK;

			DomainConfig gpu0Domain{};
			gpu0Domain.globalGridDimensionZ = globalDimensions->z;
			gpu0Domain.zDomainOffset = 0;
			gpu0Domain.ownedZSize = _zMid;
			gpu0Domain.ghostDepth = _ghostDepth;

			DomainConfig gpu1Domain{};
			gpu1Domain.globalGridDimensionZ = globalDimensions->z;
			gpu1Domain.zDomainOffset = _zMid;
			gpu1Domain.ownedZSize = globalDimensions->z - _zMid;
			gpu1Domain.ghostDepth = _ghostDepth;

			LOG_INFO("Dual-GPU domain decomposition: globalZ={}, zMid={}, ghostDepth={}, wrap={}",
				globalDimensions->z, _zMid, _ghostDepth, wrapEnabled);
			LOG_INFO("  GPU0: offset=0, owned={}, localZ={}", gpu0Domain.ownedZSize, gpu0Domain.GetLocalZSize());
			LOG_INFO("  GPU1: offset={}, owned={}, localZ={}", _zMid, gpu1Domain.ownedZSize, gpu1Domain.GetLocalZSize());

			// GPU0 renderer + compute
			_simulationContext.simulationRenderer = std::make_unique<VulkanSimulationRenderer>(
				services.deviceContext, _simulationContext.simulation, services.framesInFlight,
				storageMode, &gpu0Domain);
			_simulationContext.computeSystem = std::make_unique<ComputeSystem>(
				services.deviceContext, *_simulationContext.simulationRenderer);

			// GPU1 renderer + compute (uses the same Simulation object for topology/rules)
			_gpu1SimulationContext.simulation = _simulationContext.simulation;
			_gpu1SimulationContext.simulationRenderer = std::make_unique<VulkanSimulationRenderer>(
				*services.secondaryDeviceContext, _gpu1SimulationContext.simulation, services.framesInFlight,
				storageMode, &gpu1Domain);
			_gpu1SimulationContext.computeSystem = std::make_unique<ComputeSystem>(
				*services.secondaryDeviceContext, *_gpu1SimulationContext.simulationRenderer);

			// Ghost exchange between GPU0 and GPU1
			_ghostExchange = std::make_unique<GhostExchange>(services.deviceContext, *services.secondaryDeviceContext);
			_ghostExchange->Initialize(globalDimensions->x, globalDimensions->y, globalDimensions->z,
				_zMid, _ghostDepth, wrapEnabled, storageMode);

			if (storageMode == CellStateStorageMode::Image)
			{
				_ghostExchange->SetupForImageMode(
					_simulationContext.simulationRenderer->GetCellStateImage(0),
					_simulationContext.simulationRenderer->GetCellStateImage(1),
					_gpu1SimulationContext.simulationRenderer->GetCellStateImage(0),
					_gpu1SimulationContext.simulationRenderer->GetCellStateImage(1));
			}
			else
			{
				_ghostExchange->SetupForBufferMode(
					_simulationContext.simulationRenderer->GetCellStateBuffer(0),
					_simulationContext.simulationRenderer->GetCellStateBuffer(1),
					_gpu1SimulationContext.simulationRenderer->GetCellStateBuffer(0),
					_gpu1SimulationContext.simulationRenderer->GetCellStateBuffer(1),
					_simulationContext.simulationRenderer->GetCellStateNumBlocks(),
					_gpu1SimulationContext.simulationRenderer->GetCellStateNumBlocks());
			}

			// Ray march render systems — both GPUs render their Z-domain with compositing enabled
			vk::Extent2D renderExtent{
				services.vulkanInstance.GetWindowExtent().width,
				services.vulkanInstance.GetWindowExtent().height};
			_rayMarchRenderSystem = std::make_unique<RayMarchRenderSystem>(
				services.deviceContext, *_simulationContext.simulationRenderer, renderExtent, true);
			_gpu1RayMarchRenderSystem = std::make_unique<RayMarchRenderSystem>(
				*services.secondaryDeviceContext, *_gpu1SimulationContext.simulationRenderer, renderExtent, true);

			// Compositor transfers GPU1's rendered image to GPU0 and depth-compares per pixel
			_compositor = std::make_unique<DualGpuCompositor>(services.deviceContext, *services.secondaryDeviceContext);
			_compositor->Initialize(renderExtent, _rayMarchRenderSystem->GetColorFormat(), services.framesInFlight);

			// Bind GPU0 color attachments for the pre-built composite descriptor
			std::vector<std::shared_ptr<Image>> gpu0ColorAttachments;
			for (uint32_t frameIndex = 0; frameIndex < services.framesInFlight; frameIndex++)
				gpu0ColorAttachments.push_back(_rayMarchRenderSystem->GetColorAttachment(frameIndex));
			_compositor->SetupDescriptors(gpu0ColorAttachments);

			services.deviceContext.GetDevice().waitIdle();
			services.secondaryDeviceContext->GetDevice().waitIdle();
		}
		else
		{
			// Single-GPU path (unchanged)
			_simulationContext.simulationRenderer = std::make_unique<VulkanSimulationRenderer>(
				services.deviceContext, _simulationContext.simulation, services.framesInFlight, storageMode);

			_simulationContext.computeSystem = std::make_unique<ComputeSystem>(
				services.deviceContext, *_simulationContext.simulationRenderer);

			vk::Extent2D renderExtent{
				services.vulkanInstance.GetWindowExtent().width,
				services.vulkanInstance.GetWindowExtent().height};
			_rayMarchRenderSystem = std::make_unique<RayMarchRenderSystem>(
				services.deviceContext, *_simulationContext.simulationRenderer, renderExtent);

			services.deviceContext.GetDevice().waitIdle();
		}

		_currentFrameIndex = 0;
		_sessionRunning = true;
		_simulationHasStarted = true;
		_state = State::Running;

		LOG_INFO("Simulation ready.{}", _dualGpuEnabled ? " (dual-GPU domain decomposition)" : "");
	}

	void RenderingMode::InitializeGuiStateFromSimulation(ModeServices& services)
	{
		GuiState& guiState = services.guiSystem.GetState();
		const SimulationParameters* simulationParameters = _simulationContext.simulation.GetSimulationParameters();

		uint64_t birthBitmask = simulationParameters->birthAndMaxCellStateRules & Simulation::EXISTENCE_PERMUTATION_BIT_MASK;
		uint64_t survivalBitmask = simulationParameters->survivalAndNeighborhoodRules & Simulation::EXISTENCE_PERMUTATION_BIT_MASK;
		uint64_t maxCellState = Simulation::DecodeMaxCellState(simulationParameters->birthAndMaxCellStateRules >> Simulation::BIT_SHIFT);
		uint64_t neighborhoodFlags = simulationParameters->survivalAndNeighborhoodRules >> Simulation::BIT_SHIFT;

		guiState.birthRulesText = GuiSystem::BitmaskToRuleString(birthBitmask);
		guiState.survivalRulesText = GuiSystem::BitmaskToRuleString(survivalBitmask);
		guiState.maxCellState = static_cast<int>(maxCellState);
		guiState.faceNeighbors = (neighborhoodFlags & Simulation::FACE_NEIGHBORS_MASK) != 0;
		guiState.edgeNeighbors = (neighborhoodFlags & Simulation::EDGE_NEIGHBORS_MASK) != 0;
		guiState.cornerNeighbors = (neighborhoodFlags & Simulation::CORNER_NEIGHBORS_MASK) != 0;
		guiState.wrapAtBoundary = (neighborhoodFlags & Simulation::WRAP_NEIGHBORS_MASK) != 0;

		const glm::uvec4* dimensions = _simulationContext.simulation.GetDimensions();
		guiState.gridDimensionX = dimensions->x;
		guiState.gridDimensionY = dimensions->y;
		guiState.gridDimensionZ = dimensions->z;
		guiState.uniformGridDimensions = (dimensions->x == dimensions->y && dimensions->y == dimensions->z);

		const glm::uvec4* spawnDimensions = _simulationContext.simulation.GetCenterSpawnAreaDimensions();
		guiState.spawnDimensionX = spawnDimensions->x;
		guiState.spawnDimensionY = spawnDimensions->y;
		guiState.spawnDimensionZ = spawnDimensions->z;

		const ColorRules* colorRules = _simulationContext.simulation.GetColorRules();
		guiState.aliveColor[0] = colorRules->aliveColor.r;
		guiState.aliveColor[1] = colorRules->aliveColor.g;
		guiState.aliveColor[2] = colorRules->aliveColor.b;
		guiState.deadColor[0] = colorRules->deadColor.r;
		guiState.deadColor[1] = colorRules->deadColor.g;
		guiState.deadColor[2] = colorRules->deadColor.b;
	}

	void RenderingMode::DestroyRenderingModeSystems()
	{
		_compositor.reset();
		_gpu1RayMarchRenderSystem.reset();
		_rayMarchRenderSystem.reset();
		_ghostExchange.reset();
		_gpu1SimulationContext.DestroyGpuResources();
		_simulationContext.DestroyGpuResources();
		_dualGpuEnabled = false;
	}

	void RenderingMode::RetrieveTimestamps(ModeServices& services)
	{
		uint32_t queryCount = static_cast<uint32_t>(services.queryManager->GetNextID());
		if (queryCount == 0)
			return;

		// Poll without eWait to avoid blocking on query slots that may not have
		// been written this frame (e.g., queries registered by other modes).
		std::vector<uint64_t> timestamps(queryCount);
		auto res = services.deviceContext.GetDevice().getQueryPoolResults(
			services.deviceContext.GetQueryPool(), 0, queryCount,
			timestamps.size() * sizeof(uint64_t), timestamps.data(), sizeof(uint64_t),
			vk::QueryResultFlagBits::e64);
		if (res != vk::Result::eSuccess)
			return;

		auto rawTimings = services.queryManager->GetTimingsMs(timestamps);
		_lastRawTimings = rawTimings;
		auto smoothed = services.queryManager->GetSmoothedTimings();

		// Per-tick timing for benchmark logging
		if (services.startupConfig && services.startupConfig->benchmark && _simulationTickCount <= 15)
		{
			auto computeIt = rawTimings.find("compute");
			auto renderIt = rawTimings.find("render");
			float rawComputeMs = (computeIt != rawTimings.end()) ? computeIt->second : 0.0f;
			float rawRenderMs = (renderIt != rawTimings.end()) ? renderIt->second : 0.0f;
			LOG_INFO("[per-tick] tick={} compute={:.3f}ms render={:.3f}ms", _simulationTickCount, rawComputeMs, rawRenderMs);
		}

		// Populate GUI performance timings from smoothed results
		GuiState& guiState = services.guiSystem.GetState();
		auto get = [&](const std::string& name) -> float {
			auto it = smoothed.find(name);
			return (it != smoothed.end()) ? it->second : 0.0f;
		};

		guiState.performanceTimings.computeMs = get("compute");
		guiState.performanceTimings.renderMs = get("render");
		guiState.performanceTimings.presentMs = get("present");

		// CLI benchmark output (once per second)
		if (services.startupConfig && services.startupConfig->benchmark)
		{
			static double lastBenchmarkTime = 0;
			double now = glfwGetTime();
			if (now - lastBenchmarkTime >= 1.0)
			{
				lastBenchmarkTime = now;
				float total = get("compute") + get("render") + get("present");
				LOG_INFO("[benchmark] compute={:.2f}ms render={:.2f}ms present={:.2f}ms total={:.2f}ms",
					get("compute"), get("render"), get("present"), total);
			}
		}
	}

	void RenderingMode::BuildGui(ModeServices& services)
	{
		GuiState& guiState = services.guiSystem.GetState();
		bool isLoading = (_state == State::Loading);
		const Simulation* simulation = (_simulationContext.simulationRenderer) ? &_simulationContext.simulation : nullptr;
		Camera* camera = _simulationContext.camera.get();

		services.guiSystem.BeginModeFrame();

		// --- Simulation Controls ---
		if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (isLoading)
			{
				ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Loading simulation...");
			}
			else if (_sessionRunning)
			{
				if (ImGui::Button("Pause"))
					services.guiSystem.InjectStopRequest();
				ImGui::SameLine();
				if (ImGui::Button("Apply & Reset"))
					services.guiSystem.InjectResetRequest();
			}
			else
			{
				if (ImGui::Button(_simulationHasStarted ? "Resume" : "Start"))
					services.guiSystem.InjectStartRequest();
				ImGui::SameLine();
				if (ImGui::Button("Step"))
					services.guiSystem.InjectStepRequest();
				ImGui::SameLine();
				if (ImGui::Button("Apply & Reset"))
					services.guiSystem.InjectResetRequest();
			}

			ImGui::Text("Tick: %u", guiState.simulationTickCount);
		}

		// --- Shape ---
		if (ImGui::CollapsingHeader("Shape", ImGuiTreeNodeFlags_DefaultOpen))
		{
			const char* shapes[] = {"Cube", "Elongated Dodecahedron"};
			ImGui::Combo("Shape##render", &guiState.shape, shapes, IM_ARRAYSIZE(shapes));
		}

		// --- Simulation Rules ---
		if (ImGui::CollapsingHeader("Rules", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::RadioButton("Rules Editor", &guiState.rulesInputMode, 0);
			ImGui::SameLine();
			ImGui::RadioButton("Raw Parameters", &guiState.rulesInputMode, 1);

			if (guiState.rulesInputMode == 0)
			{
				char birthRulesBuffer[256] = {};
				char survivalRulesBuffer[256] = {};
				strncpy(birthRulesBuffer, guiState.birthRulesText.c_str(), sizeof(birthRulesBuffer) - 1);
				strncpy(survivalRulesBuffer, guiState.survivalRulesText.c_str(), sizeof(survivalRulesBuffer) - 1);

				if (ImGui::InputText("Birth Rules", birthRulesBuffer, sizeof(birthRulesBuffer)))
					guiState.birthRulesText = birthRulesBuffer;

				if (ImGui::InputText("Survival Rules", survivalRulesBuffer, sizeof(survivalRulesBuffer)))
					guiState.survivalRulesText = survivalRulesBuffer;

				ImGui::SliderInt("Max Cell State", &guiState.maxCellState, 1, 16);

				ImGui::Text("Neighborhood:");
				ImGui::Checkbox("Faces", &guiState.faceNeighbors);
				ImGui::SameLine();
				ImGui::Checkbox("Edges", &guiState.edgeNeighbors);
				ImGui::SameLine();
				ImGui::Checkbox("Corners", &guiState.cornerNeighbors);
				ImGui::Checkbox("Wrap at Boundary", &guiState.wrapAtBoundary);
			}
			else
			{
				ImGui::InputText("birthAndMaxCellStateRules", guiState.rawBirthAndMaxCellStateRulesInput,
					sizeof(guiState.rawBirthAndMaxCellStateRulesInput), ImGuiInputTextFlags_CharsDecimal);
				ImGui::InputText("survivalAndNeighborhoodRules", guiState.rawSurvivalAndNeighborhoodRulesInput,
					sizeof(guiState.rawSurvivalAndNeighborhoodRulesInput), ImGuiInputTextFlags_CharsDecimal);

				// Decode and display a human-readable preview
				uint64_t birthVal = strtoull(guiState.rawBirthAndMaxCellStateRulesInput, nullptr, 10);
				uint64_t survivalVal = strtoull(guiState.rawSurvivalAndNeighborhoodRulesInput, nullptr, 10);

				uint64_t maxCellState = Simulation::DecodeMaxCellState(birthVal >> Simulation::BIT_SHIFT);
				uint64_t neighborhoodFlags = survivalVal >> Simulation::BIT_SHIFT;

				std::string birthRulesDecoded = GuiSystem::BitmaskToRuleString(birthVal & Simulation::EXISTENCE_PERMUTATION_BIT_MASK);
				std::string survivalRulesDecoded = GuiSystem::BitmaskToRuleString(survivalVal & Simulation::EXISTENCE_PERMUTATION_BIT_MASK);

				std::string neighborhoodStr;
				if (neighborhoodFlags & Simulation::FACE_NEIGHBORS_MASK) neighborhoodStr += "F";
				if (neighborhoodFlags & Simulation::EDGE_NEIGHBORS_MASK) { if (!neighborhoodStr.empty()) neighborhoodStr += ","; neighborhoodStr += "E"; }
				if (neighborhoodFlags & Simulation::CORNER_NEIGHBORS_MASK) { if (!neighborhoodStr.empty()) neighborhoodStr += ","; neighborhoodStr += "C"; }
				if (neighborhoodFlags & Simulation::WRAP_NEIGHBORS_MASK) { if (!neighborhoodStr.empty()) neighborhoodStr += ","; neighborhoodStr += "W"; }
				if (neighborhoodStr.empty()) neighborhoodStr = "-";

				ImGui::Separator();
				ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Decoded:");
				ImGui::Text("  Max Cell State: %llu", maxCellState);
				ImGui::Text("  Neighborhood: %s", neighborhoodStr.c_str());
				ImGui::Text("  Birth Rules: %s", birthRulesDecoded.c_str());
				ImGui::Text("  Survival Rules: %s", survivalRulesDecoded.c_str());
			}
		}

		// --- Grid & Spawn ---
		if (ImGui::CollapsingHeader("Grid & Spawn"))
		{
			ImGui::Checkbox("Uniform Dimensions", &guiState.uniformGridDimensions);

			if (guiState.uniformGridDimensions)
			{
				if (ImGui::DragInt("Grid Size", &guiState.gridDimensionX, 1.0f, 1, 4096))
				{
					guiState.gridDimensionY = guiState.gridDimensionX;
					guiState.gridDimensionZ = guiState.gridDimensionX;
				}
			}
			else
			{
				ImGui::DragInt("Grid X", &guiState.gridDimensionX, 1.0f, 1, 4096);
				ImGui::DragInt("Grid Y", &guiState.gridDimensionY, 1.0f, 1, 4096);
				ImGui::DragInt("Grid Z", &guiState.gridDimensionZ, 1.0f, 1, 4096);
			}

			ImGui::Separator();
			ImGui::SliderInt("Spawn X", &guiState.spawnDimensionX, 1, guiState.gridDimensionX);
			ImGui::SliderInt("Spawn Y", &guiState.spawnDimensionY, 1, guiState.gridDimensionY);
			ImGui::SliderInt("Spawn Z", &guiState.spawnDimensionZ, 1, guiState.gridDimensionZ);

			const char* spawnModes[] = {"Random", "Filled"};
			ImGui::Combo("Spawn Mode", &guiState.spawnMode, spawnModes, IM_ARRAYSIZE(spawnModes));

			if (guiState.spawnMode == 0)
				ImGui::SliderFloat("Spawn Density", &guiState.spawnRandomDensity, 0.0f, 1.0f, "%.3f");
		}

		// --- Colors ---
		if (ImGui::CollapsingHeader("Colors"))
		{
			ImGui::ColorEdit3("Alive Color", guiState.aliveColor);
			ImGui::ColorEdit3("Dead Color", guiState.deadColor);
		}

		// --- Rendering ---
		if (ImGui::CollapsingHeader("Rendering"))
		{
			ImGui::Checkbox("Compute Skip", &guiState.enableComputeSkip);
			ImGui::Checkbox("Compute Neighbor AO", &guiState.computeNeighborCount);
			ImGui::Checkbox("Debug: Random Cell Colors", &guiState.debugRandomCellColors);
		}

		// --- Performance ---
		if (ImGui::CollapsingHeader("Performance"))
		{
			const auto& t = guiState.performanceTimings;
			ImGui::Text("Compute:  %.2f ms", t.computeMs);
			ImGui::Text("Render:   %.2f ms", t.renderMs);
			ImGui::Text("Present:  %.2f ms", t.presentMs);
			ImGui::Separator();
			ImGui::Text("GPU Total: %.2f ms", t.computeMs + t.renderMs + t.presentMs);
		}

		// --- Lighting ---
		if (ImGui::CollapsingHeader("Lighting"))
		{
			ImGui::Checkbox("Enable Lighting", &guiState.enableLighting);
			if (guiState.enableLighting)
			{
				ImGui::SliderFloat("Ambient", &guiState.ambientStrength, 0.0f, 1.0f);
				ImGui::SliderFloat("Diffuse", &guiState.diffuseStrength, 0.0f, 1.0f);
				ImGui::SliderFloat("Specular", &guiState.specularStrength, 0.0f, 2.0f);
				ImGui::SliderFloat("Shininess", &guiState.shininess, 1.0f, 256.0f);
				ImGui::SliderFloat3("Light Direction", guiState.lightDirection, -1.0f, 1.0f);
			}
		}

		// --- Culling Planes ---
		// Each enabled plane hides cells where dot(normal, worldPos) > offset.
		// Multiple enabled planes combine: a cell is hidden if ANY plane culls it.
		if (ImGui::CollapsingHeader("Culling Planes"))
		{
			float gridExtent = static_cast<float>(std::max({guiState.gridDimensionX, guiState.gridDimensionY, guiState.gridDimensionZ}));
			for (int planeIndex = 0; planeIndex < 4; planeIndex++)
			{
				ImGui::PushID(planeIndex);
				char label[32];
				snprintf(label, sizeof(label), "Plane %d", planeIndex);
				ImGui::Checkbox(label, &guiState.cullingPlaneEnabled[planeIndex]);
				if (guiState.cullingPlaneEnabled[planeIndex])
				{
					ImGui::SliderFloat3("Normal", guiState.cullingPlaneNormal[planeIndex], -1.0f, 1.0f);
					ImGui::SliderFloat("Offset", &guiState.cullingPlaneOffset[planeIndex], -gridExtent, gridExtent);
					if (ImGui::Button("Align to Camera"))
					{
						// Set normal toward the camera so cells in front of the camera get culled,
						// and place the plane through the camera position. Slide the offset toward
						// negative to push the plane into the scene and reveal interior structure.
						if (camera)
						{
							glm::vec3 cameraForward = camera->GetForwardDirection();
							glm::vec3 cameraPosition = camera->GetPosition();
							glm::vec3 normal = -cameraForward; // points back at camera
							guiState.cullingPlaneNormal[planeIndex][0] = normal.x;
							guiState.cullingPlaneNormal[planeIndex][1] = normal.y;
							guiState.cullingPlaneNormal[planeIndex][2] = normal.z;
							guiState.cullingPlaneOffset[planeIndex] = glm::dot(normal, cameraPosition);
						}
					}
				}
				ImGui::PopID();
			}
		}

		// --- Camera ---
		if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
		{
			const char* cameraModes[] = {"Fixed Pace Orbit", "Fixed Rate Orbit", "Free Camera"};
			if (ImGui::Combo("Camera Mode", &guiState.cameraMode, cameraModes, IM_ARRAYSIZE(cameraModes)))
			{
				if (camera) camera->SetCameraMode(static_cast<CameraMode>(guiState.cameraMode));
			}

			if (guiState.cameraMode < 2)
			{
				if (ImGui::SliderFloat("Orbit Radius", &guiState.orbitRadius, 1.0f, 1000.0f))
					if (camera) camera->SetOrbitRadius(guiState.orbitRadius);
				if (ImGui::SliderFloat("Orbit Speed", &guiState.orbitAngularVelocity, 0.0f, 5.0f))
					if (camera) camera->SetOrbitAngularVelocity(guiState.orbitAngularVelocity);
				if (ImGui::SliderFloat("Orbit Elevation", &guiState.orbitElevation, -1.5f, 1.5f))
					if (camera) camera->SetOrbitElevation(guiState.orbitElevation);
			}
			else
			{
				if (ImGui::SliderFloat("Movement Speed", &guiState.movementSpeed, 1.0f, 100.0f))
					if (camera) camera->SetMovementSpeed(guiState.movementSpeed);
				if (ImGui::SliderFloat("Look Speed", &guiState.lookSpeed, 0.1f, 5.0f))
					if (camera) camera->SetLookSpeed(guiState.lookSpeed);
			}

			if (ImGui::SliderFloat("FOV", &guiState.fieldOfViewDegrees, 10.0f, 120.0f))
				if (camera) camera->SetFieldOfView(glm::radians(guiState.fieldOfViewDegrees));
		}

		services.guiSystem.EndModeFrame();
	}

} // namespace Cave
