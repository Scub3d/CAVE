#include "videoEncodingMode.h"

#include "../vulkan/vulkanInstance.h"
#include "../vulkan/simulationRenderer.h"
#include "../vulkan/systems/computeSystem.h"
#include "../vulkan/systems/videoEncoderSystem.h"
#include "../vulkan/systems/presentSystem.h"
#include "../vulkan/systems/guiSystem.h"
#include "../simulation/simulation.h"
#include "../common/queryManager.h"
#include "../common/startupConfig.h"
#include "../common/logger.h"
#include "../camera.h"
#include "../shapes/cube.h"
#include "../shapes/elongatedRhombicDodecahedron.h"

#include <algorithm>
#include <imgui.h>
#include <glm/glm.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace Cave
{

	VideoEncodingMode::~VideoEncodingMode() = default;

	void VideoEncodingMode::Enter(ModeServices& services)
	{
		auto extent = services.vulkanInstance.GetWindowExtent();
		float aspectRatio = static_cast<float>(extent.width) / static_cast<float>(extent.height);
		_simulationContext.camera = std::make_unique<Camera>(
			services.vulkanInstance.GetVulkanGLFWWindow(), aspectRatio);
		_state = State::Configuring;
		_sessionRunning = false;
		_currentFrameIndex = 0;
	}

	bool VideoEncodingMode::OnFrame(ModeServices& services, float deltaTimeInSeconds)
	{
		switch (_state)
		{
		case State::Configuring:
		{
			BuildGui(services);
			services.presentSystem.PresentGuiOnly(_currentFrameIndex);

			if (services.guiSystem.ConsumeEncodeRequest())
			{
				const GuiState& guiState = services.guiSystem.GetState();
				_videoBatchQueue.clear();

				if (guiState.videoSourceMode == 0 && !guiState.videoImportedPermutations.empty())
				{
					if (guiState.videoEncodeAll)
					{
						for (const auto& p : guiState.videoImportedPermutations)
							_videoBatchQueue.push_back({p});
					}
					else if (guiState.videoSelectedPermutationIndex >= 0 &&
							 guiState.videoSelectedPermutationIndex < static_cast<int>(guiState.videoImportedPermutations.size()))
					{
						_videoBatchQueue.push_back({guiState.videoImportedPermutations[guiState.videoSelectedPermutationIndex]});
					}
				}
				else
				{
					SimulationParameters manualParams{};
					manualParams.birthAndMaxCellStateRules = strtoull(guiState.videoManualBirthRulesInput, nullptr, 10);
					manualParams.survivalAndNeighborhoodRules = strtoull(guiState.videoManualSurvivalRulesInput, nullptr, 10);
					manualParams.shapeAndGridConfiguration =
						(static_cast<uint64_t>(guiState.gridDimensionX - 1) << 5) |
						(static_cast<uint64_t>(guiState.gridDimensionY - 1) << 13) |
						(static_cast<uint64_t>(guiState.gridDimensionZ - 1) << 21);
					_videoBatchQueue.push_back({manualParams});
				}

				if (!_videoBatchQueue.empty())
				{
					_videoCurrentJobIndex = 0;
					GuiState& state = services.guiSystem.GetState();
					state.videoTotalJobs = static_cast<uint32_t>(_videoBatchQueue.size());

					const auto& job = _videoBatchQueue[0];
					const GuiState& gs = services.guiSystem.GetState();

					std::shared_ptr<Shape> shape = (gs.shape == 1)
						? ElongatedRhombicDodecahedron::Create() : Cube::Create();

					_simulationFuture = std::async(std::launch::async,
						[gs, job, shape]()
						{
							return Simulation(
								glm::uvec3(gs.gridDimensionX, gs.gridDimensionY, gs.gridDimensionZ),
								glm::uvec3(gs.spawnDimensionX, gs.spawnDimensionY, gs.spawnDimensionZ),
								gs.spawnMode, job.params,
								ColorRules{
									glm::vec4(gs.aliveColor[0], gs.aliveColor[1], gs.aliveColor[2], 1.0f),
									glm::vec4(gs.deadColor[0], gs.deadColor[1], gs.deadColor[2], 1.0f)},
								shape, gs.spawnRandomDensity);
						});

					_state = State::Loading;
				}
			}
			break;
		}

		case State::Loading:
		{
			BuildGui(services);
			services.presentSystem.PresentGuiOnly(_currentFrameIndex);

			if (_simulationFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
			{
				_simulationContext.simulation = _simulationFuture.get();

				const GuiState& gs = services.guiSystem.GetState();
				const glm::uvec4* globalDimensions = _simulationContext.simulation.GetDimensions();
				CellStateStorageMode storageMode = services.startupConfig
					? services.startupConfig->storageMode : CellStateStorageMode::Image;

				_dualGpuEnabled = services.secondaryDeviceContext != nullptr;

				if (_dualGpuEnabled)
				{

					const auto& neighborDeltaConfigurations = _simulationContext.simulation.GetNeighborDeltaConfigurations();
					_ghostDepth = 0;
					for (const auto& configurationDeltas : neighborDeltaConfigurations)
						for (const auto& delta : configurationDeltas)
							_ghostDepth = std::max(_ghostDepth, static_cast<uint32_t>(std::abs(delta.z)));
					if (_ghostDepth == 0) _ghostDepth = 1;

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

					LOG_INFO("Video encode dual-GPU: globalZ={}, zMid={}, ghostDepth={}", globalDimensions->z, _zMid, _ghostDepth);

					// GPU0 renderer + compute
					_simulationContext.simulationRenderer = std::make_unique<VulkanSimulationRenderer>(
						services.deviceContext, _simulationContext.simulation, services.framesInFlight,
						storageMode, &gpu0Domain);
					_simulationContext.computeSystem = std::make_unique<ComputeSystem>(
						services.deviceContext, *_simulationContext.simulationRenderer);

					// GPU1 renderer + compute
					_gpu1SimulationContext.simulation = _simulationContext.simulation;
					_gpu1SimulationContext.simulationRenderer = std::make_unique<VulkanSimulationRenderer>(
						*services.secondaryDeviceContext, _gpu1SimulationContext.simulation, services.framesInFlight,
						storageMode, &gpu1Domain);
					_gpu1SimulationContext.computeSystem = std::make_unique<ComputeSystem>(
						*services.secondaryDeviceContext, *_gpu1SimulationContext.simulationRenderer);

					// Ghost exchange
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

					// Render systems for compositing (at video resolution)
					vk::Extent2D renderExtent{
						static_cast<uint32_t>(gs.videoResolutionWidth),
						static_cast<uint32_t>(gs.videoResolutionHeight)};
					_gpu0RayMarchRenderSystem = std::make_unique<RayMarchRenderSystem>(
						services.deviceContext, *_simulationContext.simulationRenderer, renderExtent, true);
					_gpu1RayMarchRenderSystem = std::make_unique<RayMarchRenderSystem>(
						*services.secondaryDeviceContext, *_gpu1SimulationContext.simulationRenderer, renderExtent, true);

					// Compositor
					_compositor = std::make_unique<DualGpuCompositor>(services.deviceContext, *services.secondaryDeviceContext);
					_compositor->Initialize(renderExtent, _gpu0RayMarchRenderSystem->GetColorFormat(), services.framesInFlight);

					std::vector<std::shared_ptr<Image>> gpu0ColorAttachments;
					for (uint32_t fi = 0; fi < services.framesInFlight; fi++)
						gpu0ColorAttachments.push_back(_gpu0RayMarchRenderSystem->GetColorAttachment(fi));
					_compositor->SetupDescriptors(gpu0ColorAttachments);

					// Video encoder on GPU0 (encodes composited frames)
					_videoEncoderSystem = std::make_unique<VideoEncoderSystem>(
						services.vulkanInstance, services.deviceContext, *_simulationContext.simulationRenderer,
						gs.videoResolutionWidth, gs.videoResolutionHeight, gs.videoFps);

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

					_videoEncoderSystem = std::make_unique<VideoEncoderSystem>(
						services.vulkanInstance, services.deviceContext, *_simulationContext.simulationRenderer,
						gs.videoResolutionWidth, gs.videoResolutionHeight, gs.videoFps);

					services.deviceContext.GetDevice().waitIdle();
				}

				_simulationContext.camera->SetCameraMode(CameraMode::FixedRateOrbit);
				_simulationContext.camera->SetOrbitRadius(gs.orbitRadius);
				_simulationContext.camera->SetOrbitAngularVelocity(gs.orbitAngularVelocity);
				_simulationContext.camera->SetOrbitElevation(gs.orbitElevation);
				_simulationContext.camera->SetFieldOfView(glm::radians(gs.fieldOfViewDegrees));
				_simulationContext.camera->SetAspectRatio(static_cast<float>(gs.videoResolutionWidth) /
														  static_cast<float>(gs.videoResolutionHeight));

				_videoCurrentTick = 0;
				_videoTotalTicks = gs.videoDurationTicks;
				_videoAccumulatedBitstream.clear();
				_currentFrameIndex = 0;
				_sessionRunning = true;
				_state = State::Encoding;

				LOG_INFO("Video encoding started: {}x{} @{}fps, {} ticks{}",
					gs.videoResolutionWidth, gs.videoResolutionHeight, gs.videoFps, _videoTotalTicks,
					_dualGpuEnabled ? " (dual-GPU)" : "");
			}
			break;
		}

		case State::Encoding:
		{
			auto tickBudgetStart = std::chrono::steady_clock::now();
			const auto tickBudget = std::chrono::milliseconds(12);

			while (std::chrono::steady_clock::now() - tickBudgetStart < tickBudget)
			{
				if (_videoCurrentTick >= _videoTotalTicks)
					break;

				const uint32_t videoFrameIndex = _videoCurrentTick % services.framesInFlight;

				_simulationContext.camera->Update(0.0f);

				// Animate orbit radius: start close (spawn-sized) and pull back to full grid view.
				// Quadratic ease-out — camera pulls back quickly enough early on to keep the expanding
				// simulation in view, then eases into the final grid-filling frame.
				// With --no-camera-zoom the camera is pinned at endRadius for the entire clip.
				{
					float meshScaleRatio = _simulationContext.simulation.GetShape()->GetMeshScaleRatio();
					float gridExtent = _simulationContext.simulation.GetDimensions()->x * 2.0f * meshScaleRatio;
					float spawnExtent = _simulationContext.simulation.GetCenterSpawnAreaDimensions()->x * 2.0f * meshScaleRatio;
					float startRadius = spawnExtent * 3.0f;
					float endRadius = gridExtent * 2.0f;

					bool disableZoom = services.startupConfig && services.startupConfig->disableCameraZoom;
					float currentRadius;
					if (disableZoom)
					{
						currentRadius = endRadius;
					}
					else
					{
						// Linear pull-back. Original quadratic ease-out front-loaded the retreat
						// (75% retreated by midpoint), then a pow(p, 1.5) ease-in over-corrected
						// (whole sim only visible at 1:13 of a 2:00 clip). Linear strikes the
						// middle: whole sim becomes visible around the halfway mark.
						float progress = static_cast<float>(_videoCurrentTick) / static_cast<float>(_videoTotalTicks);
						currentRadius = startRadius + (endRadius - startRadius) * progress;
					}
					_simulationContext.camera->SetOrbitRadius(currentRadius);
				}

				const GuiState& guiState = services.guiSystem.GetState();
				CameraData cameraData = _simulationContext.BuildCameraData(guiState);

				// Refresh overlay metadata + visibility before this frame's encode call.
				if (_videoEncoderSystem)
				{
					_videoEncoderSystem->SetOverlayEnabled(guiState.videoShowOverlay);
					if (guiState.videoShowOverlay)
						_videoEncoderSystem->SetOverlayInfo(BuildOverlayInfo(guiState));
				}

				// On the very first frame, skip the simulation dispatch so the init state is rendered
				// verbatim. Without this, frame 0 shows the post-tick-1 state (the CA rules can grow
				// sparse spawns dramatically in a single step, which is misleading for debugging).
				bool dispatchCompute = (_videoCurrentTick > 0);

				if (_dualGpuEnabled)
				{
					// Dual-GPU: compute on both GPUs, ghost exchange, dual render, composite, encode
					vk::Fence gpu0RenderFence = _gpu0RayMarchRenderSystem->GetInFlightFence(videoFrameIndex);
					_simulationContext.computeSystem->Tick(videoFrameIndex, gpu0RenderFence, dispatchCompute,
						_enableComputeSkip, services.queryManager, services.deviceContext.GetQueryPool());
					_gpu1SimulationContext.computeSystem->Tick(videoFrameIndex, {}, dispatchCompute, _enableComputeSkip);

					// Wait for both computes, then ghost exchange
					services.deviceContext.GetDevice().waitForFences(
						_simulationContext.computeSystem->GetInFlightFence(videoFrameIndex), vk::True, UINT64_MAX);
					services.secondaryDeviceContext->GetDevice().waitForFences(
						_gpu1SimulationContext.computeSystem->GetInFlightFence(videoFrameIndex), vk::True, UINT64_MAX);

					uint32_t nextFrame = (videoFrameIndex + 1) % services.framesInFlight;
					_ghostExchange->Exchange(nextFrame);

					// Render on both GPUs
					RayMarchPushConstants gpu0PushConstants = _simulationContext.BuildRayMarchPushConstants(guiState);
					vk::Semaphore gpu0ComputeSemaphore = _simulationContext.computeSystem->GetCompletedSemaphore(videoFrameIndex);
					uint64_t gpu0ComputeSemaphoreValue = _simulationContext.computeSystem->GetCompletedSemaphoreSignalValue(videoFrameIndex);
					_gpu0RayMarchRenderSystem->RenderFrame(videoFrameIndex,
						gpu0ComputeSemaphore, gpu0ComputeSemaphoreValue, cameraData, gpu0PushConstants);

					RayMarchPushConstants gpu1PushConstants = _gpu1SimulationContext.BuildRayMarchPushConstants(guiState);
					vk::Semaphore gpu1ComputeSemaphore = _gpu1SimulationContext.computeSystem->GetCompletedSemaphore(videoFrameIndex);
					uint64_t gpu1ComputeSemaphoreValue = _gpu1SimulationContext.computeSystem->GetCompletedSemaphoreSignalValue(videoFrameIndex);
					_gpu1RayMarchRenderSystem->RenderFrame(videoFrameIndex,
						gpu1ComputeSemaphore, gpu1ComputeSemaphoreValue, cameraData, gpu1PushConstants);

					// Wait for both renders, then composite
					services.deviceContext.GetDevice().waitForFences(
						_gpu0RayMarchRenderSystem->GetInFlightFence(videoFrameIndex), vk::True, UINT64_MAX);
					services.secondaryDeviceContext->GetDevice().waitForFences(
						_gpu1RayMarchRenderSystem->GetInFlightFence(videoFrameIndex), vk::True, UINT64_MAX);

					vk::Image compositeImage = _compositor->Composite(videoFrameIndex,
						_gpu1RayMarchRenderSystem->GetColorAttachment(videoFrameIndex));

					// Blit composite to encoder and encode
					vk::Extent2D renderExtent = _gpu0RayMarchRenderSystem->GetRenderExtent();
					_videoEncoderSystem->BlitAndEncodeFrame(videoFrameIndex, compositeImage, renderExtent);
				}
				else
				{
					// Single-GPU: compute → render+encode (original path)
					_simulationContext.computeSystem->Tick(videoFrameIndex,
						_videoEncoderSystem->GetRenderFence(videoFrameIndex), dispatchCompute,
						_enableComputeSkip, services.queryManager, services.deviceContext.GetQueryPool());

					RayMarchPushConstants pushConstants = _simulationContext.BuildRayMarchPushConstants(guiState);
					_videoEncoderSystem->RenderAndEncodeFrame(
						videoFrameIndex,
						_simulationContext.computeSystem->GetCompletedSemaphore(videoFrameIndex),
						_simulationContext.computeSystem->GetCompletedSemaphoreSignalValue(videoFrameIndex),
						cameraData, pushConstants,
						services.queryManager, services.deviceContext.GetQueryPool());
				}

				RetrieveTimestamps(services);

				_videoCurrentTick++;
			}

			// Update progress
			GuiState& gs = services.guiSystem.GetState();
			gs.videoCurrentFrame = _videoCurrentTick;
			gs.videoTotalFrames = _videoTotalTicks;
			gs.videoEncodingProgress = static_cast<float>(_videoCurrentTick) / static_cast<float>(_videoTotalTicks);
			gs.videoCurrentJob = _videoCurrentJobIndex;

			BuildGui(services);
			services.presentSystem.PresentGuiOnly(_currentFrameIndex);

			// Check completion
			if (_videoCurrentTick >= _videoTotalTicks)
			{
				services.deviceContext.GetDevice().waitIdle();
				if (_dualGpuEnabled && services.secondaryDeviceContext)
					services.secondaryDeviceContext->GetDevice().waitIdle();

				_videoEncoderSystem->Finish(_videoAccumulatedBitstream);

				const GuiState& outState = services.guiSystem.GetState();
				const auto& job = _videoBatchQueue[_videoCurrentJobIndex];

				std::string filename = std::to_string(job.params.shapeAndGridConfiguration) + "-" +
					std::to_string(job.params.survivalAndNeighborhoodRules) + "-" +
					std::to_string(job.params.birthAndMaxCellStateRules);

				std::string outputFolder = outState.videoOutputFolderPath;
				std::filesystem::create_directories(outputFolder);

				std::string outputPath = outputFolder + filename + ".h264";

				std::ofstream outputFile(outputPath, std::ios::binary);
				outputFile.write(_videoAccumulatedBitstream.data(), _videoAccumulatedBitstream.size());
				outputFile.close();
				LOG_INFO("Raw H.264 written to {} ({} bytes)", outputPath, _videoAccumulatedBitstream.size());

				if (outState.videoKeepH264Only)
				{
					LOG_INFO("Skipping ffmpeg remux (--keep-h264-only). Raw H.264 kept at {}", outputPath);
				}
				else
				{
					std::string mp4Path = outputFolder + filename + ".mp4";
					std::string ffmpegCommand = "ffmpeg -y -framerate " + std::to_string(outState.videoFps) +
						" -i \"" + outputPath + "\" -c copy \"" + mp4Path + "\" 2>&1";
					LOG_INFO("Remuxing to MP4: {}", ffmpegCommand);
					int ffmpegResult = system(ffmpegCommand.c_str());
					if (ffmpegResult == 0)
					{
						LOG_INFO("MP4 written to {}", mp4Path);
						std::filesystem::remove(outputPath);
					}
					else
					{
						LOG_WARNING("ffmpeg remux failed (exit code {}). Raw .h264 file kept.", ffmpegResult);
					}
				}

				// Advance to next job
				_videoCurrentJobIndex++;
				if (_videoCurrentJobIndex < _videoBatchQueue.size())
				{
					const auto& nextJob = _videoBatchQueue[_videoCurrentJobIndex];
					const GuiState& ngs = services.guiSystem.GetState();

					std::shared_ptr<Shape> nextShape = (ngs.shape == 1)
						? ElongatedRhombicDodecahedron::Create() : Cube::Create();

					_simulationContext.simulation = Simulation(
						glm::uvec3(ngs.gridDimensionX, ngs.gridDimensionY, ngs.gridDimensionZ),
						glm::uvec3(ngs.spawnDimensionX, ngs.spawnDimensionY, ngs.spawnDimensionZ),
						ngs.spawnMode, nextJob.params,
						ColorRules{
							glm::vec4(ngs.aliveColor[0], ngs.aliveColor[1], ngs.aliveColor[2], 1.0f),
							glm::vec4(ngs.deadColor[0], ngs.deadColor[1], ngs.deadColor[2], 1.0f)},
						nextShape, ngs.spawnRandomDensity);

					services.deviceContext.GetDevice().waitIdle();
					_simulationContext.simulationRenderer->Reset();
					_simulationContext.computeSystem->RebuildPipeline();
					_videoEncoderSystem->ResetForNewJob();

					_videoCurrentTick = 0;
					_videoAccumulatedBitstream.clear();
				}
				else
				{
					_videoEncoderSystem.reset();
					_compositor.reset();
					_gpu1RayMarchRenderSystem.reset();
					_gpu0RayMarchRenderSystem.reset();
					_ghostExchange.reset();
					_gpu1SimulationContext.DestroyGpuResources();
					_simulationContext.DestroyGpuResources();
					_dualGpuEnabled = false;
					LOG_INFO("All video encoding jobs complete ({} videos)", _videoBatchQueue.size());
					_state = State::Complete;
					_sessionRunning = false;

					if (services.startupConfig && services.startupConfig->autoStart)
					{
						LOG_INFO("Auto-exit: video encoding complete");
						return false;
					}
				}
			}

			if (services.guiSystem.ConsumeCancelEncodeRequest())
			{
				services.deviceContext.GetDevice().waitIdle();
				if (_dualGpuEnabled && services.secondaryDeviceContext)
					services.secondaryDeviceContext->GetDevice().waitIdle();
				_videoEncoderSystem.reset();
				_compositor.reset();
				_gpu1RayMarchRenderSystem.reset();
				_gpu0RayMarchRenderSystem.reset();
				_ghostExchange.reset();
				_gpu1SimulationContext.DestroyGpuResources();
				_simulationContext.DestroyGpuResources();
				_dualGpuEnabled = false;
				_state = State::Configuring;
				_sessionRunning = false;
			}
			break;
		}

		case State::Complete:
		{
			BuildGui(services);
			services.presentSystem.PresentGuiOnly(_currentFrameIndex);

			if (services.guiSystem.ConsumeEncodeRequest())
				_state = State::Configuring;
			break;
		}
		}

		return true;
	}

	void VideoEncodingMode::Exit(ModeServices& services)
	{
		if (_dualGpuEnabled && services.secondaryDeviceContext)
			services.secondaryDeviceContext->GetDevice().waitIdle();
		_videoEncoderSystem.reset();
		_compositor.reset();
		_gpu1RayMarchRenderSystem.reset();
		_gpu0RayMarchRenderSystem.reset();
		_ghostExchange.reset();
		_gpu1SimulationContext.DestroyGpuResources();
		_simulationContext.DestroyGpuResources();
		_simulationContext.camera.reset();
		_dualGpuEnabled = false;
	}

	void VideoEncodingMode::RetrieveTimestamps(ModeServices& services)
	{
		uint32_t queryCount = static_cast<uint32_t>(services.queryManager->GetNextID());
		if (queryCount == 0)
			return;

		// Use eWithAvailability instead of eWait to avoid blocking on query slots
		// that may not have been written this frame (e.g., queries registered by other modes).
		std::vector<uint64_t> timestamps(queryCount);
		auto res = services.deviceContext.GetDevice().getQueryPoolResults(
			services.deviceContext.GetQueryPool(), 0, queryCount,
			timestamps.size() * sizeof(uint64_t), timestamps.data(), sizeof(uint64_t),
			vk::QueryResultFlagBits::e64);
		if (res != vk::Result::eSuccess)
			return;

		services.queryManager->GetTimingsMs(timestamps);
		auto smoothed = services.queryManager->GetSmoothedTimings();

		GuiState& guiState = services.guiSystem.GetState();
		auto get = [&](const std::string& name) -> float {
			auto it = smoothed.find(name);
			return (it != smoothed.end()) ? it->second : 0.0f;
		};

		guiState.performanceTimings.encodeComputeMs = get("encodeCompute");
		guiState.performanceTimings.encodeRenderMs = get("encodeRender");
		guiState.performanceTimings.encodePresentMs = get("encodePresent");
	}

	OverlayInfo VideoEncodingMode::BuildOverlayInfo(const GuiState& guiState)
	{
		const SimulationParameters* params = _simulationContext.simulation.GetSimulationParameters();
		const glm::uvec4* gridDims = _simulationContext.simulation.GetDimensions();
		const glm::uvec4* spawnDims = _simulationContext.simulation.GetCenterSpawnAreaDimensions();

		OverlayInfo info{};
		info.gridX = gridDims->x;
		info.gridY = gridDims->y;
		info.gridZ = gridDims->z;
		info.spawnX = spawnDims->x;
		info.spawnY = spawnDims->y;
		info.spawnZ = spawnDims->z;

		const uint64_t birthRulesPacked = params->birthAndMaxCellStateRules;
		const uint64_t survivalRulesPacked = params->survivalAndNeighborhoodRules;
		const uint64_t neighborhoodFlags = survivalRulesPacked >> Simulation::BIT_SHIFT;

		info.birthRules = GuiSystem::BitmaskToRuleString(birthRulesPacked & Simulation::EXISTENCE_PERMUTATION_BIT_MASK);
		info.survivalRules = GuiSystem::BitmaskToRuleString(survivalRulesPacked & Simulation::EXISTENCE_PERMUTATION_BIT_MASK);
		info.shapeName = _simulationContext.simulation.GetShape() ? _simulationContext.simulation.GetShape()->GetName() : "?";
		info.maxCellState = static_cast<uint32_t>(Simulation::DecodeMaxCellState(birthRulesPacked >> Simulation::BIT_SHIFT));
		info.faceNeighbors = (neighborhoodFlags & Simulation::FACE_NEIGHBORS_MASK) != 0;
		info.edgeNeighbors = (neighborhoodFlags & Simulation::EDGE_NEIGHBORS_MASK) != 0;
		info.cornerNeighbors = (neighborhoodFlags & Simulation::CORNER_NEIGHBORS_MASK) != 0;
		info.wrapAtBoundary = (neighborhoodFlags & Simulation::WRAP_NEIGHBORS_MASK) != 0;

		info.currentTick = _videoCurrentTick;
		info.maxTicks = _videoTotalTicks;
		info.ticksPerSecond = static_cast<uint32_t>(guiState.videoFps);
		info.position = (guiState.videoOverlayPosition == 1) ? OverlayPosition::TopRight : OverlayPosition::TopLeft;
		return info;
	}

	void VideoEncodingMode::BuildGui(ModeServices& services)
	{
		GuiState& guiState = services.guiSystem.GetState();
		bool isEncoding = (_state == State::Encoding);
		bool isLoading = (_state == State::Loading);

		services.guiSystem.BeginModeFrame();

		// --- Source ---
		if (ImGui::CollapsingHeader("Source", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::RadioButton("Import from JSON", &guiState.videoSourceMode, 0);
			ImGui::SameLine();
			ImGui::RadioButton("Manual Entry", &guiState.videoSourceMode, 1);

			if (guiState.videoSourceMode == 0)
			{
				char pathBuffer[512] = {};
				strncpy(pathBuffer, guiState.videoSourceJsonPath.c_str(), sizeof(pathBuffer) - 1);
				if (ImGui::InputText("JSON File Path", pathBuffer, sizeof(pathBuffer)))
					guiState.videoSourceJsonPath = pathBuffer;

				if (ImGui::Button("Load JSON"))
				{
					std::ifstream inputFile(guiState.videoSourceJsonPath);
					if (inputFile.is_open())
					{
						try
						{
							guiState.videoImportedPermutations.clear();
							guiState.videoImportedTicksSurvived.clear();

							auto ingestEntry = [&guiState](const nlohmann::json& entry) {
								SimulationParameters params{};
								params.birthAndMaxCellStateRules = entry["birthAndMaxCellStateRules"].get<uint64_t>();
								params.survivalAndNeighborhoodRules = entry["survivalAndNeighborhoodRules"].get<uint64_t>();
								params.shapeAndGridConfiguration = entry.value("shapeAndGridConfiguration", uint64_t(0));
								guiState.videoImportedPermutations.push_back(params);
								guiState.videoImportedTicksSurvived.push_back(entry.value("ticksSurvived", uint32_t(0)));
							};

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
							LOG_INFO("Loaded {} permutations from {}", guiState.videoImportedPermutations.size(), guiState.videoSourceJsonPath);
						}
						catch (const std::exception& e)
						{
							LOG_ERROR("Failed to parse JSON: {}", e.what());
						}
					}
				}

				if (!guiState.videoImportedPermutations.empty())
				{
					ImGui::Text("%zu permutations loaded", guiState.videoImportedPermutations.size());
					ImGui::Checkbox("Encode All", &guiState.videoEncodeAll);

					if (ImGui::BeginTable("PermutationsTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 150)))
					{
						ImGui::TableSetupScrollFreeze(0, 1);
						ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 30.0f);
						ImGui::TableSetupColumn("Birth Rules", ImGuiTableColumnFlags_WidthStretch);
						ImGui::TableSetupColumn("Survival Rules", ImGuiTableColumnFlags_WidthStretch);
						ImGui::TableSetupColumn("Ticks", ImGuiTableColumnFlags_WidthFixed, 50.0f);
						ImGui::TableHeadersRow();

						for (size_t i = 0; i < guiState.videoImportedPermutations.size(); i++)
						{
							const auto& p = guiState.videoImportedPermutations[i];
							ImGui::TableNextRow();

							bool isSelected = (guiState.videoSelectedPermutationIndex == static_cast<int>(i));

							ImGui::TableNextColumn();
							if (ImGui::Selectable(std::to_string(i).c_str(), isSelected, ImGuiSelectableFlags_SpanAllColumns))
								guiState.videoSelectedPermutationIndex = static_cast<int>(i);
							ImGui::TableNextColumn();
							ImGui::Text("%s", GuiSystem::BitmaskToRuleString(p.birthAndMaxCellStateRules & Simulation::EXISTENCE_PERMUTATION_BIT_MASK).c_str());
							ImGui::TableNextColumn();
							ImGui::Text("%s", GuiSystem::BitmaskToRuleString(p.survivalAndNeighborhoodRules & Simulation::EXISTENCE_PERMUTATION_BIT_MASK).c_str());
							ImGui::TableNextColumn();
							ImGui::Text("%u", guiState.videoImportedTicksSurvived[i]);
						}
						ImGui::EndTable();
					}
				}
			}
			else
			{
				ImGui::Text("Enter the two 64-bit simulation parameter values:");

				ImGui::InputText("birthAndMaxCellStateRules", guiState.videoManualBirthRulesInput,
					sizeof(guiState.videoManualBirthRulesInput), ImGuiInputTextFlags_CharsDecimal);
				ImGui::InputText("survivalAndNeighborhoodRules", guiState.videoManualSurvivalRulesInput,
					sizeof(guiState.videoManualSurvivalRulesInput), ImGuiInputTextFlags_CharsDecimal);

				uint64_t birthVal = strtoull(guiState.videoManualBirthRulesInput, nullptr, 10);
				uint64_t survivalVal = strtoull(guiState.videoManualSurvivalRulesInput, nullptr, 10);

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
			ImGui::SliderInt("Grid X##video", &guiState.gridDimensionX, 1, 1024);
			ImGui::SliderInt("Grid Y##video", &guiState.gridDimensionY, 1, 1024);
			ImGui::SliderInt("Grid Z##video", &guiState.gridDimensionZ, 1, 1024);
			ImGui::Separator();
			ImGui::SliderInt("Spawn X##video", &guiState.spawnDimensionX, 1, guiState.gridDimensionX);
			ImGui::SliderInt("Spawn Y##video", &guiState.spawnDimensionY, 1, guiState.gridDimensionY);
			ImGui::SliderInt("Spawn Z##video", &guiState.spawnDimensionZ, 1, guiState.gridDimensionZ);

			const char* spawnModes[] = {"Random", "Filled"};
			ImGui::Combo("Spawn Mode##video", &guiState.spawnMode, spawnModes, IM_ARRAYSIZE(spawnModes));

			if (guiState.spawnMode == 0)
				ImGui::SliderFloat("Spawn Density##video", &guiState.spawnRandomDensity, 0.0f, 1.0f, "%.3f");
		}

		// --- Video Settings ---
		if (ImGui::CollapsingHeader("Video Settings", ImGuiTreeNodeFlags_DefaultOpen))
		{
			const char* resolutions[] = {
				"256x144", "426x240", "640x360", "854x480",
				"1280x720", "1920x1080", "2560x1440",
				"3840x2160", "5120x2880", "7680x4320"
			};
			static const int resolutionWidths[]  = {256, 426, 640, 854, 1280, 1920, 2560, 3840, 5120, 7680};
			static const int resolutionHeights[] = {144, 240, 360, 480,  720, 1080, 1440, 2160, 2880, 4320};
			static int resolutionIndex = 5;
			if (ImGui::Combo("Resolution", &resolutionIndex, resolutions, IM_ARRAYSIZE(resolutions)))
			{
				guiState.videoResolutionWidth = resolutionWidths[resolutionIndex];
				guiState.videoResolutionHeight = resolutionHeights[resolutionIndex];
			}
			ImGui::SliderInt("FPS", &guiState.videoFps, 1, 144);
			ImGui::SliderInt("Duration (ticks)", &guiState.videoDurationTicks, 50, 2000);

			char outputBuffer[512] = {};
			strncpy(outputBuffer, guiState.videoOutputFolderPath.c_str(), sizeof(outputBuffer) - 1);
			if (ImGui::InputText("Output Folder##video", outputBuffer, sizeof(outputBuffer)))
				guiState.videoOutputFolderPath = outputBuffer;

			ImGui::Checkbox("Show Overlay##video", &guiState.videoShowOverlay);
			if (guiState.videoShowOverlay)
			{
				const char* overlayPositions[] = {"Top Left", "Top Right"};
				ImGui::Combo("Overlay Position##video", &guiState.videoOverlayPosition,
					overlayPositions, IM_ARRAYSIZE(overlayPositions));
			}
		}

		// --- Camera ---
		if (ImGui::CollapsingHeader("Camera##video"))
		{
			ImGui::SliderFloat("Orbit Radius##video", &guiState.orbitRadius, 1.0f, 1000.0f);
			ImGui::SliderFloat("Orbit Speed##video", &guiState.orbitAngularVelocity, 0.0f, 5.0f);
			ImGui::SliderFloat("Orbit Elevation##video", &guiState.orbitElevation, -1.5f, 1.5f);
			ImGui::SliderFloat("FOV##video", &guiState.fieldOfViewDegrees, 10.0f, 120.0f);
		}

		// --- Colors ---
		if (ImGui::CollapsingHeader("Colors##video"))
		{
			ImGui::ColorEdit3("Alive Color##video", guiState.aliveColor);
			ImGui::ColorEdit3("Dead Color##video", guiState.deadColor);
		}

		// --- Performance ---
		if (ImGui::CollapsingHeader("Performance##video"))
		{
			const auto& t = guiState.performanceTimings;
			ImGui::Text("Compute: %.2f ms", t.encodeComputeMs);
			ImGui::Text("Render:  %.2f ms", t.encodeRenderMs);
			ImGui::Text("Present: %.2f ms", t.encodePresentMs);
			ImGui::Separator();
			ImGui::Text("GPU Total: %.2f ms", t.encodeComputeMs + t.encodeRenderMs + t.encodePresentMs);
		}

		// --- Lighting ---
		if (ImGui::CollapsingHeader("Lighting##video"))
		{
			ImGui::Checkbox("Enable Lighting##video", &guiState.enableLighting);
			if (guiState.enableLighting)
			{
				ImGui::SliderFloat("Ambient##video", &guiState.ambientStrength, 0.0f, 1.0f);
				ImGui::SliderFloat("Diffuse##video", &guiState.diffuseStrength, 0.0f, 1.0f);
				ImGui::SliderFloat("Specular##video", &guiState.specularStrength, 0.0f, 2.0f);
				ImGui::SliderFloat("Shininess##video", &guiState.shininess, 1.0f, 256.0f);
				ImGui::SliderFloat3("Light Direction##video", guiState.lightDirection, -1.0f, 1.0f);
			}
		}

		// --- Controls ---
		ImGui::Separator();
		if (isLoading)
		{
			ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Loading simulation...");
		}
		else if (isEncoding)
		{
			char overlay[128];
			snprintf(overlay, sizeof(overlay), "Frame %u / %u (Job %u / %u)",
				guiState.videoCurrentFrame, guiState.videoTotalFrames,
				guiState.videoCurrentJob + 1, guiState.videoTotalJobs);
			ImGui::ProgressBar(guiState.videoEncodingProgress, ImVec2(-1, 0), overlay);

			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));
			if (ImGui::Button("Cancel Encoding"))
				services.guiSystem.InjectCancelEncodeRequest();
			ImGui::PopStyleColor(2);
		}
		else
		{
			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.2f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.8f, 0.3f, 1.0f));
			if (ImGui::Button("Encode"))
				services.guiSystem.InjectEncodeRequest();
			ImGui::PopStyleColor(2);
		}

		services.guiSystem.EndModeFrame();
	}

} // namespace Cave
