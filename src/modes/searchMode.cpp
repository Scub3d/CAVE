#include "searchMode.h"

#include "../vulkan/vulkanInstance.h"
#include "../vulkan/systems/searchSystem.h"
#include "../vulkan/systems/presentSystem.h"
#include "../vulkan/systems/guiSystem.h"
#include "../vulkan/searchShaderGenerator.h"
#include "../simulation/simulation.h"
#include "../common/queryManager.h"
#include "../common/logger.h"
#include "../common/permutationMath.h"
#include "../common/startupConfig.h"
#include "../shapes/cube.h"
#include "../shapes/elongatedRhombicDodecahedron.h"

#include <imgui.h>
#include <chrono>
#include <cmath>
#include <array>

namespace Cave
{

	SearchMode::~SearchMode() = default;

	void SearchMode::Enter(ModeServices& services)
	{
		_searchShaderGenerator = std::make_unique<SearchShaderGenerator>(services.deviceContext);
		if (services.secondaryDeviceContext)
			_searchShaderGeneratorGpu1 = std::make_unique<SearchShaderGenerator>(*services.secondaryDeviceContext);
		_state = State::Configuring;
		_sessionRunning = false;
		_currentFrameIndex = 0;
	}

	bool SearchMode::OnFrame(ModeServices& services, float deltaTimeInSeconds)
	{
		switch (_state)
		{
		case State::Configuring:
		{
			BuildGui(services);
			services.presentSystem.PresentGuiOnly(_currentFrameIndex);

			if (services.guiSystem.ConsumeStartRequest())
			{
				const GuiState& guiState = services.guiSystem.GetState();

				// Enumerate valid neighborhood configurations
				std::vector<uint64_t> validNeighborhoodConfigs;
				{
					uint64_t selectedMask = 0;
					if (guiState.searchFaceNeighbors) selectedMask |= Simulation::FACE_NEIGHBORS_MASK;
					if (guiState.searchEdgeNeighbors) selectedMask |= Simulation::EDGE_NEIGHBORS_MASK;
					if (guiState.searchCornerNeighbors) selectedMask |= Simulation::CORNER_NEIGHBORS_MASK;

					for (uint64_t subset = 1; subset <= selectedMask; subset++)
					{
						if ((subset & selectedMask) == subset && subset != 0)
						{
							validNeighborhoodConfigs.push_back(subset);
							if (guiState.searchWrapAtBoundary)
								validNeighborhoodConfigs.push_back(subset | Simulation::WRAP_NEIGHBORS_MASK);
						}
					}

					if (validNeighborhoodConfigs.empty())
						validNeighborhoodConfigs.push_back(Simulation::FACE_NEIGHBORS_MASK);
				}

				// Resolve shape
				std::shared_ptr<Shape> shape;
				switch (guiState.searchShape)
				{
				case 1:  shape = ElongatedRhombicDodecahedron::Create(); break;
				default: shape = Cube::Create(); break;
				}

				// Build grid configs (shared across all chunks — SearchGridConfig depends only on
				// shape + neighborhood + maxCS, not on grid dimensions).
				std::vector<SearchGridConfig> gridConfigs;
				for (int maxCellState = guiState.searchMinMaxCellState; maxCellState <= guiState.searchMaxMaxCellState; maxCellState++)
				{
					for (uint64_t neighborhoodFlags : validNeighborhoodConfigs)
					{
						bool faces = (neighborhoodFlags & Simulation::FACE_NEIGHBORS_MASK) != 0;
						bool edges = (neighborhoodFlags & Simulation::EDGE_NEIGHBORS_MASK) != 0;
						bool corners = (neighborhoodFlags & Simulation::CORNER_NEIGHBORS_MASK) != 0;
						auto neighborDeltaConfigurations = shape->CalculateNeighborDeltaConfigurations(faces, edges, corners);
						uint32_t neighborCount = neighborDeltaConfigurations.empty() ? 0
							: static_cast<uint32_t>(neighborDeltaConfigurations[0].size());

						uint64_t maxPermutation = (neighborCount < 60) ? ((1ULL << neighborCount) - 1) : Simulation::EXISTENCE_PERMUTATION_BIT_MASK;

						gridConfigs.push_back(SearchGridConfig{
							static_cast<uint64_t>(maxCellState),
							neighborhoodFlags,
							maxPermutation,
							neighborCount
						});
					}
				}

				// File-prefix base string (no grid-size suffix — that's per-chunk)
				std::string neighborhoodTag;
				if (guiState.searchFaceNeighbors) neighborhoodTag += "F";
				if (guiState.searchEdgeNeighbors) neighborhoodTag += "E";
				if (guiState.searchCornerNeighbors) neighborhoodTag += "C";
				if (guiState.searchWrapAtBoundary) neighborhoodTag += "W";
				if (neighborhoodTag.empty()) neighborhoodTag = "F";

				std::string filePrefixBase =
					"cs" + std::to_string(guiState.searchMinMaxCellState) +
					"-" + std::to_string(guiState.searchMaxMaxCellState) +
					"_" + neighborhoodTag +
					"_t" + std::to_string(guiState.searchMaxTicksToSurvive);
				if (guiState.searchMaxRuleBits > 0)
					filePrefixBase += "_b" + std::to_string(guiState.searchMaxRuleBits);

				std::string outputFolder = guiState.searchOutputFolderPath + shape->GetName() + "/";

				// Populate sweep context — immutable across chunks
				_sweepContext = SweepContext{};
				_sweepContext.shape = shape;
				_sweepContext.gridConfigs = std::move(gridConfigs);
				_sweepContext.maxTicksToSurvive = guiState.searchMaxTicksToSurvive;
				_sweepContext.maxRuleBits = guiState.searchMaxRuleBits;
				_sweepContext.simulationWorkgroupSize = guiState.searchWorkgroupSize;
				_sweepContext.minMaxCellState = guiState.searchMinMaxCellState;
				_sweepContext.maxMaxCellState = guiState.searchMaxMaxCellState;
				_sweepContext.spawnAreaDimensionX = guiState.searchSpawnDimensionX;
				_sweepContext.spawnAreaDimensionY = guiState.searchSpawnDimensionY;
				_sweepContext.spawnAreaDimensionZ = guiState.searchSpawnDimensionZ;
				_sweepContext.spawnMode = static_cast<uint32_t>(guiState.searchSpawnMode);
				_sweepContext.spawnDensityIndex = static_cast<uint32_t>(
					std::clamp(static_cast<int>(guiState.searchSpawnRandomDensity * 16.0f) - 1, 0, 15));
				_sweepContext.outputFolder = outputFolder;
				_sweepContext.filePrefixBase = filePrefixBase;

				// Enumerate chunks as the cartesian product of all enabled sweep axes.
				_pendingChunks.clear();

				// Axis 1: grid size (cube side, applied to X=Y=Z)
				std::vector<uint32_t> gridSizes;
				if (guiState.searchGridSizeSweepEnabled
					&& guiState.searchGridSizeSweepStep > 0
					&& guiState.searchGridSizeSweepMin <= guiState.searchGridSizeSweepMax)
				{
					for (int g = guiState.searchGridSizeSweepMin;
						g <= guiState.searchGridSizeSweepMax;
						g += guiState.searchGridSizeSweepStep)
					{
						gridSizes.push_back(static_cast<uint32_t>(g));
					}
				}
				else
				{
					// Fall back to the non-sweep grid dimension. Asymmetric in principle,
					// but the sweep is cube-only so we treat this as a single chunk at (X,Y,Z).
					gridSizes.push_back(static_cast<uint32_t>(guiState.searchGridDimensionX));
				}

				// Axis 2: density index 0..15 (only meaningful when spawn mode == Random)
				std::vector<uint32_t> densityIndices;
				bool sweepDensity = guiState.searchDensitySweepEnabled
					&& _sweepContext.spawnMode == 0
					&& guiState.searchDensitySweepStep > 0.0f
					&& guiState.searchDensitySweepMin <= guiState.searchDensitySweepMax;
				if (sweepDensity)
				{
					for (float d = guiState.searchDensitySweepMin;
						d <= guiState.searchDensitySweepMax + 1e-6f;
						d += guiState.searchDensitySweepStep)
					{
						int index = std::clamp(static_cast<int>(d * 16.0f) - 1, 0, 15);
						densityIndices.push_back(static_cast<uint32_t>(index));
					}
				}
				else
				{
					densityIndices.push_back(_sweepContext.spawnDensityIndex);
				}

				// Axis 3: run-seed index 0..K-1
				std::vector<uint32_t> seedIndices;
				int seedCount = std::max(1, guiState.searchSeedSweepCount);
				for (int s = 0; s < seedCount; s++)
					seedIndices.push_back(static_cast<uint32_t>(s));

				uint32_t partitionCount = static_cast<uint32_t>(std::max(1, guiState.searchChunksPerConfig));

				auto buildChunk = [&](uint32_t gridX, uint32_t gridY, uint32_t gridZ,
					uint32_t densityIndex, uint32_t seedIndex,
					uint32_t partitionIndex, uint32_t partitionCountValue)
				{
					ChunkConfiguration chunkConfig{};
					chunkConfig.shapeId = 0;
					chunkConfig.gridDimensionX = gridX;
					chunkConfig.gridDimensionY = gridY;
					chunkConfig.gridDimensionZ = gridZ;
					chunkConfig.spawnAreaDimensionX = std::min(_sweepContext.spawnAreaDimensionX, gridX);
					chunkConfig.spawnAreaDimensionY = std::min(_sweepContext.spawnAreaDimensionY, gridY);
					chunkConfig.spawnAreaDimensionZ = std::min(_sweepContext.spawnAreaDimensionZ, gridZ);
					chunkConfig.spawnPlacementId = 0;
					chunkConfig.spawnRegionShapeId = _sweepContext.spawnMode;
					chunkConfig.spawnSizeIndex = 0;
					chunkConfig.spawnDensityIndex = densityIndex;
					chunkConfig.runSeedIndex = seedIndex;
					chunkConfig.partitionIndex = partitionIndex;
					chunkConfig.partitionCount = partitionCountValue;
					return chunkConfig;
				};

				for (uint32_t g : gridSizes)
					for (uint32_t d : densityIndices)
						for (uint32_t s : seedIndices)
							for (uint32_t p = 0; p < partitionCount; p++)
								_pendingChunks.push_back(buildChunk(g, g, g, d, s, p, partitionCount));

				_totalChunkCount = static_cast<uint32_t>(_pendingChunks.size());
				_completedChunkCount = 0;
				_viableHistory.clear();
				_possibleHistory.clear();
				_unviableHistory.clear();
				_ruleAggregator.Clear();

				// Compute VRAM budget in bytes from the user's gigabyte setting.
				_vramBudgetBytesPerGpu = static_cast<uint64_t>(
					std::max(0.5f, guiState.searchGpuVramBudgetGb) * 1024.0 * 1024.0 * 1024.0);
				_vramUsedBytesGpu0 = 0;
				_vramUsedBytesGpu1 = 0;
				_activeSystemsGpu0.clear();
				_activeSystemsGpu1.clear();
				_activeVramGpu0.clear();
				_activeVramGpu1.clear();

				LOG_INFO("Starting search sweep: {} chunk(s), maxCS {}-{}, VRAM budget {:.1f} GB/GPU, output: {}",
					_totalChunkCount,
					_sweepContext.minMaxCellState, _sweepContext.maxMaxCellState,
					_vramBudgetBytesPerGpu / 1073741824.0,
					outputFolder);

				// Transition to Running; the scheduler loop in State::Running will
				// immediately spawn as many chunks as fit in the VRAM budget.
				_sessionRunning = true;
				_searchPaused = false;
				_state = State::Running;
			}
			break;
		}

		case State::Running:
		{
			// Tick all active SearchSystems on both GPUs up to an 8 ms per-frame budget.
			// Each pool iteration submits one tick per GPU; we loop until either the
			// budget is exhausted or every system is mid-submission.
			if (!_searchPaused)
			{
				auto tickBudgetStart = std::chrono::steady_clock::now();
				const auto tickBudget = std::chrono::milliseconds(8);

				while (std::chrono::steady_clock::now() - tickBudgetStart < tickBudget)
				{
					bool anyProgress = false;
					for (auto& system : _activeSystemsGpu0)
					{
						if (system->IsTickComplete())
						{
							system->FlushProgress();
							services.deviceContext.GetDevice().resetQueryPool(services.deviceContext.GetQueryPool(), 0, 32);
							system->SubmitNextTick(services.queryManager, services.deviceContext.GetQueryPool());
							anyProgress = true;
						}
					}
					for (auto& system : _activeSystemsGpu1)
					{
						if (system->IsTickComplete())
						{
							system->FlushProgress();
							services.secondaryDeviceContext->GetDevice().resetQueryPool(services.secondaryDeviceContext->GetQueryPool(), 0, 32);
							system->SubmitNextTick();
							anyProgress = true;
						}
					}
					if (!anyProgress) break;
				}
				RetrieveTimestamps(services);
			}

			// Aggregate progress across all active SearchSystems for the GUI dashboard.
			_lastSearchProgress = SearchProgress{};
			auto accumulateProgress = [&](const SearchSystem& system)
			{
				SearchProgress p = system.GetProgress();
				_lastSearchProgress.savedViableCount += p.savedViableCount;
				_lastSearchProgress.savedPossibleCount += p.savedPossibleCount;
				_lastSearchProgress.savedUnviableCount += p.savedUnviableCount;
				_lastSearchProgress.totalGridCount += p.totalGridCount;
				_lastSearchProgress.activeGridCount += p.activeGridCount;
				// currentPermutation / currentGridInfo: take the first active system's for display
				if (_lastSearchProgress.currentGridInfo.numberOfSimulationTicksSurvived == 0)
				{
					_lastSearchProgress.currentPermutation = p.currentPermutation;
					_lastSearchProgress.currentGridInfo = p.currentGridInfo;
				}
			};
			for (auto& system : _activeSystemsGpu0) accumulateProgress(*system);
			for (auto& system : _activeSystemsGpu1) accumulateProgress(*system);

			BuildGui(services);
			services.presentSystem.PresentGuiOnly(_currentFrameIndex);

			// Chunk-level work stealing: a SearchSystem with idle slots (partially done)
			// can steal work from a sibling SearchSystem of matching grid dimensions on
			// the same GPU pool. Runs BEFORE harvest so a stealing chunk can keep its
			// otherwise-idle slots alive with stolen work instead of being torn down.
			//
			// Lifecycle when stealing fails: the chunk retains its idle slots (unfilled),
			// completes its still-busy slots, eventually hits IsRunning()==false, and is
			// harvested on a subsequent tick. The spawn loop then fills the freed VRAM
			// with pending chunks, which may themselves share grid dims with still-busy
			// chunks and steal from them — iterative reuse without manual coordination.
			auto attemptCrossChunkStealing = [](std::vector<std::unique_ptr<SearchSystem>>& pool)
			{
				for (size_t thief = 0; thief < pool.size(); thief++)
				{
					if (!pool[thief]->HasIdleSlots()) continue;
					const auto& thiefCfg = pool[thief]->GetChunkConfiguration();
					for (size_t donor = 0; donor < pool.size(); donor++)
					{
						if (donor == thief) continue;
						if (!pool[donor]->HasBusySlots()) continue;
						const auto& donorCfg = pool[donor]->GetChunkConfiguration();
						if (thiefCfg.gridDimensionX != donorCfg.gridDimensionX
							|| thiefCfg.gridDimensionY != donorCfg.gridDimensionY
							|| thiefCfg.gridDimensionZ != donorCfg.gridDimensionZ)
							continue; // buffer sizes would mismatch
						pool[thief]->TryStealFromOtherChunk(*pool[donor]);
					}
				}
			};
			attemptCrossChunkStealing(_activeSystemsGpu0);
			attemptCrossChunkStealing(_activeSystemsGpu1);

			// Harvest any finished systems into the aggregator, freeing their VRAM.
			HarvestCompletedOnGpu0(services);
			HarvestCompletedOnGpu1(services);

			// Fill available VRAM with pending chunks. Prefer the GPU with more free
			// budget so concurrent systems stay roughly balanced.
			while (!_pendingChunks.empty())
			{
				const ChunkConfiguration& next = _pendingChunks.front();
				uint64_t chunkVramBytes = PermutationMath::EstimateChunkVramBytes(
					next.gridDimensionX, next.gridDimensionY, next.gridDimensionZ,
					static_cast<uint32_t>(_sweepContext.gridConfigs.size()));

				if (chunkVramBytes > _vramBudgetBytesPerGpu)
				{
					// Single chunk exceeds one GPU's entire budget. Log and abort the sweep —
					// future enhancement: split gridConfigs across GPUs to fit.
					LOG_ERROR("Chunk requires {:.2f} GB but VRAM budget is only {:.2f} GB per GPU. Aborting sweep.",
						chunkVramBytes / 1073741824.0, _vramBudgetBytesPerGpu / 1073741824.0);
					_pendingChunks.clear();
					break;
				}

				uint64_t freeGpu0 = _vramBudgetBytesPerGpu - _vramUsedBytesGpu0;
				uint64_t freeGpu1 = services.secondaryDeviceContext
					? _vramBudgetBytesPerGpu - _vramUsedBytesGpu1
					: 0;

				bool spawnedHere = false;
				if (freeGpu0 >= freeGpu1)
				{
					if (TrySpawnChunkOnGpu0(services, next, chunkVramBytes))
					{
						_pendingChunks.erase(_pendingChunks.begin());
						spawnedHere = true;
					}
					else if (services.secondaryDeviceContext
						&& TrySpawnChunkOnGpu1(services, next, chunkVramBytes))
					{
						_pendingChunks.erase(_pendingChunks.begin());
						spawnedHere = true;
					}
				}
				else
				{
					if (TrySpawnChunkOnGpu1(services, next, chunkVramBytes))
					{
						_pendingChunks.erase(_pendingChunks.begin());
						spawnedHere = true;
					}
					else if (TrySpawnChunkOnGpu0(services, next, chunkVramBytes))
					{
						_pendingChunks.erase(_pendingChunks.begin());
						spawnedHere = true;
					}
				}

				if (!spawnedHere)
					break; // both GPUs at budget — wait for a harvest before retrying
			}

			// Sweep complete when nothing pending and nothing active.
			if (_pendingChunks.empty()
				&& _activeSystemsGpu0.empty()
				&& _activeSystemsGpu1.empty())
			{
				LOG_INFO("Sweep complete. Viable: {}, Possible: {}, Unviable: {}",
					_lastSearchProgress.savedViableCount, _lastSearchProgress.savedPossibleCount,
					_lastSearchProgress.savedUnviableCount);

				std::string aggregatePath = _sweepContext.outputFolder
					+ _sweepContext.filePrefixBase + "_aggregate.jsonl";
				_ruleAggregator.WriteJsonl(aggregatePath);
				LOG_INFO("Aggregated {} unique rules to {}",
					_ruleAggregator.UniqueRuleCount(), aggregatePath);

				_state = State::Complete;
				_sessionRunning = false;

				if (services.startupConfig && services.startupConfig->autoStart)
				{
					LOG_INFO("Auto-exit: search sweep complete");
					return false;
				}
			}

			if (services.guiSystem.ConsumePauseRequest())
				_searchPaused = true;

			if (services.guiSystem.ConsumeResumeRequest())
				_searchPaused = false;

			if (services.guiSystem.ConsumeStopRequest())
			{
				FinalizeAllChunks(services);
				_pendingChunks.clear();

				if (_ruleAggregator.UniqueRuleCount() > 0)
				{
					std::string aggregatePath = _sweepContext.outputFolder
						+ _sweepContext.filePrefixBase + "_aggregate_partial.jsonl";
					_ruleAggregator.WriteJsonl(aggregatePath);
					LOG_INFO("Wrote partial aggregate ({} unique rules) to {}",
						_ruleAggregator.UniqueRuleCount(), aggregatePath);
				}

				LOG_INFO("Search stopped mid-sweep ({} of {} chunks completed). Viable: {}, Possible: {}, Unviable: {}",
					_completedChunkCount, _totalChunkCount,
					_lastSearchProgress.savedViableCount, _lastSearchProgress.savedPossibleCount,
					_lastSearchProgress.savedUnviableCount);
				_state = State::Configuring;
				_sessionRunning = false;
				_searchPaused = false;
			}
			break;
		}

		case State::Complete:
		{
			BuildGui(services);
			services.presentSystem.PresentGuiOnly(_currentFrameIndex);

			if (services.guiSystem.ConsumeStartRequest())
			{
				services.deviceContext.GetDevice().waitIdle();
				if (services.secondaryDeviceContext)
					services.secondaryDeviceContext->GetDevice().waitIdle();
				_activeSystemsGpu0.clear();
				_activeSystemsGpu1.clear();
				_activeVramGpu0.clear();
				_activeVramGpu1.clear();
				_vramUsedBytesGpu0 = 0;
				_vramUsedBytesGpu1 = 0;
				_state = State::Configuring;
			}
			break;
		}
		}

		return true;
	}

	void SearchMode::Exit(ModeServices& services)
	{
		FinalizeAllChunks(services);
		_searchShaderGeneratorGpu1.reset();
		_searchShaderGenerator.reset();
	}

	void SearchMode::RetrieveTimestamps(ModeServices& services)
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

		services.queryManager->GetTimingsMs(timestamps);
		auto smoothed = services.queryManager->GetSmoothedTimings();

		GuiState& guiState = services.guiSystem.GetState();
		auto get = [&](const std::string& name) -> float {
			auto it = smoothed.find(name);
			return (it != smoothed.end()) ? it->second : 0.0f;
		};

		guiState.performanceTimings.searchSimulationMs = get("searchSimulation");
		guiState.performanceTimings.searchManagerMs = get("searchManager");
		guiState.performanceTimings.searchCopyMs = get("searchCopy");
	}

	void SearchMode::BuildGui(ModeServices& services)
	{
		GuiState& guiState = services.guiSystem.GetState();
		bool isSearchRunning = (_state == State::Running);

		services.guiSystem.BeginModeFrame();

		// --- Shape ---
		if (ImGui::CollapsingHeader("Shape", ImGuiTreeNodeFlags_DefaultOpen))
		{
			const char* shapes[] = {"Cube", "Elongated Dodecahedron"};
			ImGui::Combo("Shape##search", &guiState.searchShape, shapes, IM_ARRAYSIZE(shapes));
		}

		// --- Max Cell State Search Range ---
		if (ImGui::CollapsingHeader("Max Cell State Search Range", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::TextWrapped("Searches all max cell state values from min to max.");
			ImGui::SliderInt("Min##searchCS", &guiState.searchMinMaxCellState, 1, 16);
			ImGui::SliderInt("Max##searchCS", &guiState.searchMaxMaxCellState, guiState.searchMinMaxCellState, 16);
		}

		// --- Neighborhood Types to Search ---
		if (ImGui::CollapsingHeader("Neighborhood Types to Search", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::TextWrapped("All permutations of the selected types will be searched (e.g. F alone, E alone, F+E together).");
			ImGui::Checkbox("Faces##search", &guiState.searchFaceNeighbors);
			ImGui::SameLine();
			ImGui::Checkbox("Edges##search", &guiState.searchEdgeNeighbors);
			ImGui::SameLine();
			ImGui::Checkbox("Corners##search", &guiState.searchCornerNeighbors);
			ImGui::Checkbox("Wrap at Boundary##search", &guiState.searchWrapAtBoundary);
		}

		// --- Grid & Spawn ---
		if (ImGui::CollapsingHeader("Grid & Spawn"))
		{
			ImGui::Checkbox("Sweep grid size (serial chunks)##search", &guiState.searchGridSizeSweepEnabled);
			if (guiState.searchGridSizeSweepEnabled)
			{
				ImGui::SliderInt("Min##searchGridSweep", &guiState.searchGridSizeSweepMin, 1, 512);
				ImGui::SliderInt("Max##searchGridSweep",
					&guiState.searchGridSizeSweepMax,
					guiState.searchGridSizeSweepMin, 1024);
				ImGui::SliderInt("Step##searchGridSweep", &guiState.searchGridSizeSweepStep, 1, 256);
				int chunkCount = (guiState.searchGridSizeSweepMax - guiState.searchGridSizeSweepMin) / std::max(1, guiState.searchGridSizeSweepStep) + 1;
				ImGui::Text("  -> %d chunk(s) at sizes %d, %d, ... %d (cube each)",
					chunkCount,
					guiState.searchGridSizeSweepMin,
					guiState.searchGridSizeSweepMin + guiState.searchGridSizeSweepStep,
					guiState.searchGridSizeSweepMax);
			}
			else
			{
				ImGui::SliderInt("Grid X##search", &guiState.searchGridDimensionX, 1, 1024);
				ImGui::SliderInt("Grid Y##search", &guiState.searchGridDimensionY, 1, 1024);
				ImGui::SliderInt("Grid Z##search", &guiState.searchGridDimensionZ, 1, 1024);
			}

			ImGui::Separator();
			ImGui::SliderInt("Spawn X##search", &guiState.searchSpawnDimensionX, 1, guiState.searchGridDimensionX);
			ImGui::SliderInt("Spawn Y##search", &guiState.searchSpawnDimensionY, 1, guiState.searchGridDimensionY);
			ImGui::SliderInt("Spawn Z##search", &guiState.searchSpawnDimensionZ, 1, guiState.searchGridDimensionZ);

			const char* spawnModes[] = {"Random", "Filled"};
			ImGui::Combo("Spawn Mode##search", &guiState.searchSpawnMode, spawnModes, IM_ARRAYSIZE(spawnModes));

			if (guiState.searchSpawnMode == 0)
				ImGui::SliderFloat("Spawn Density##search", &guiState.searchSpawnRandomDensity, 0.0f, 1.0f, "%.3f");

			if (guiState.searchSpawnMode == 0)
			{
				ImGui::Checkbox("Sweep spawn density (serial chunks)##search", &guiState.searchDensitySweepEnabled);
				if (guiState.searchDensitySweepEnabled)
				{
					ImGui::SliderFloat("Density min##searchDensitySweep", &guiState.searchDensitySweepMin, 0.0f, 1.0f, "%.3f");
					ImGui::SliderFloat("Density max##searchDensitySweep",
						&guiState.searchDensitySweepMax,
						guiState.searchDensitySweepMin, 1.0f, "%.3f");
					ImGui::SliderFloat("Density step##searchDensitySweep", &guiState.searchDensitySweepStep, 0.01f, 1.0f, "%.3f");
					int densityChunkCount = static_cast<int>(
						(guiState.searchDensitySweepMax - guiState.searchDensitySweepMin) /
						std::max(0.001f, guiState.searchDensitySweepStep)) + 1;
					ImGui::Text("  -> %d density value(s)", densityChunkCount);
				}
			}

			ImGui::SliderInt("Seed runs per chunk##searchSeedSweep", &guiState.searchSeedSweepCount, 1, 16);
			if (guiState.searchSeedSweepCount > 1)
				ImGui::Text("  -> each (grid, density) chunk runs %d times with distinct seeds", guiState.searchSeedSweepCount);

			ImGui::SliderFloat("GPU VRAM budget (GB)##searchVramBudget",
				&guiState.searchGpuVramBudgetGb, 0.5f, 46.0f, "%.1f GB");
			{
				// Estimate how many chunks fit per GPU at the current grid size and sweep slot count.
				int selectedNeighborsLocal = (guiState.searchFaceNeighbors ? 1 : 0)
					+ (guiState.searchEdgeNeighbors ? 1 : 0)
					+ (guiState.searchCornerNeighbors ? 1 : 0);
				int neighborhoodCountLocal = selectedNeighborsLocal > 0 ? (1 << selectedNeighborsLocal) - 1 : 1;
				if (guiState.searchWrapAtBoundary && neighborhoodCountLocal > 0) neighborhoodCountLocal *= 2;
				int maxCsCountLocal = guiState.searchMaxMaxCellState - guiState.searchMinMaxCellState + 1;
				uint32_t estimatedSlotCount = static_cast<uint32_t>(std::max(1, maxCsCountLocal * std::max(1, neighborhoodCountLocal)));
				uint32_t previewGridSide = guiState.searchGridSizeSweepEnabled
					? static_cast<uint32_t>(guiState.searchGridSizeSweepMax)
					: static_cast<uint32_t>(guiState.searchGridDimensionX);
				uint64_t perChunkBytes = PermutationMath::EstimateChunkVramBytes(
					previewGridSide, previewGridSide, previewGridSide, estimatedSlotCount);
				uint64_t budgetBytes = static_cast<uint64_t>(guiState.searchGpuVramBudgetGb * 1073741824.0);
				int chunksPerGpu = perChunkBytes > 0 ? static_cast<int>(budgetBytes / perChunkBytes) : 0;
				ImGui::Text("  ~%.2f GB / chunk at %ux%ux%u  ->  ~%d chunks fit per GPU",
					perChunkBytes / 1073741824.0, previewGridSide, previewGridSide, previewGridSide, chunksPerGpu);
			}
		}

		// --- Search Settings ---
		if (ImGui::CollapsingHeader("Search Settings", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::SliderInt("Max Ticks to Survive", &guiState.searchMaxTicksToSurvive, 10, 1000);

			ImGui::SliderInt("Max Rule Bits (0 = unbounded)##searchMaxRuleBits",
				&guiState.searchMaxRuleBits, 0, 16);
			if (guiState.searchMaxRuleBits > 0)
				ImGui::TextColored(ImVec4(0.6f, 0.7f, 0.9f, 1.0f),
					"  Only rules with <= %d birth bits and <= %d survival bits will be tested.",
					guiState.searchMaxRuleBits, guiState.searchMaxRuleBits);

			int maxCellStateCount = guiState.searchMaxMaxCellState - guiState.searchMinMaxCellState + 1;
			int neighborhoodConfigCount = 0;
			int selectedNeighborTypes = 0;
			if (guiState.searchFaceNeighbors) selectedNeighborTypes++;
			if (guiState.searchEdgeNeighbors) selectedNeighborTypes++;
			if (guiState.searchCornerNeighbors) selectedNeighborTypes++;
			if (selectedNeighborTypes > 0)
				neighborhoodConfigCount = (1 << selectedNeighborTypes) - 1;
			if (guiState.searchWrapAtBoundary && neighborhoodConfigCount > 0)
				neighborhoodConfigCount *= 2;

			int totalGrids = maxCellStateCount * std::max(1, neighborhoodConfigCount);
			ImGui::Text("Parallel simulations: %d (%d cell states x %d configs)",
				totalGrids, maxCellStateCount, std::max(1, neighborhoodConfigCount));

			if (selectedNeighborTypes > 0)
			{
				std::shared_ptr<Shape> previewShape = (guiState.searchShape == 1)
					? ElongatedRhombicDodecahedron::Create()
					: Cube::Create();
				uint64_t totalPermutations = PermutationMath::SearchTotalPermutationCount(
					*previewShape,
					guiState.searchMinMaxCellState, guiState.searchMaxMaxCellState,
					guiState.searchFaceNeighbors, guiState.searchEdgeNeighbors, guiState.searchCornerNeighbors,
					guiState.searchWrapAtBoundary);
				ImGui::Text("Total rule permutations: %llu", (unsigned long long)totalPermutations);
			}
		}

		// --- Output ---
		if (ImGui::CollapsingHeader("Output"))
		{
			const char* shapeNames[] = {"cube"};
			std::string fullOutputPath = guiState.searchOutputFolderPath + shapeNames[guiState.searchShape] + "/";
			ImGui::Text("Output folder: %s", fullOutputPath.c_str());

			std::string neighborhoodTag;
			if (guiState.searchFaceNeighbors) neighborhoodTag += "F";
			if (guiState.searchEdgeNeighbors) neighborhoodTag += "E";
			if (guiState.searchCornerNeighbors) neighborhoodTag += "C";
			if (guiState.searchWrapAtBoundary) neighborhoodTag += "W";
			if (neighborhoodTag.empty()) neighborhoodTag = "F";

			std::string filePrefix =
				"cs" + std::to_string(guiState.searchMinMaxCellState) +
				"-" + std::to_string(guiState.searchMaxMaxCellState) +
				"_" + neighborhoodTag +
				"_" + std::to_string(guiState.searchGridDimensionX) + "x" +
				std::to_string(guiState.searchGridDimensionY) + "x" +
				std::to_string(guiState.searchGridDimensionZ) +
				"_t" + std::to_string(guiState.searchMaxTicksToSurvive);

			ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f), "  %s_viable.jsonl", filePrefix.c_str());
			ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.5f, 1.0f), "  %s_possible.jsonl", filePrefix.c_str());
			ImGui::TextColored(ImVec4(0.8f, 0.5f, 0.5f, 1.0f), "  %s_unviable.jsonl", filePrefix.c_str());

			char outputFolderBuffer[512] = {};
			strncpy(outputFolderBuffer, guiState.searchOutputFolderPath.c_str(), sizeof(outputFolderBuffer) - 1);
			if (ImGui::InputText("Base Output Folder", outputFolderBuffer, sizeof(outputFolderBuffer)))
				guiState.searchOutputFolderPath = outputFolderBuffer;
		}

		// --- Performance ---
		if (isSearchRunning || _searchPaused)
		{
			if (ImGui::CollapsingHeader("Performance##search"))
			{
				const auto& t = guiState.performanceTimings;
				ImGui::Text("Simulation: %.2f ms", t.searchSimulationMs);
				ImGui::Text("Manager:    %.2f ms", t.searchManagerMs);
				ImGui::Text("Copy:       %.2f ms", t.searchCopyMs);
				ImGui::Separator();
				ImGui::Text("GPU Total:  %.2f ms", t.searchSimulationMs + t.searchManagerMs + t.searchCopyMs);
			}
		}

		// --- Dashboard ---
		if (&_lastSearchProgress && (isSearchRunning || _searchPaused))
		{
			if (ImGui::CollapsingHeader("Progress", ImGuiTreeNodeFlags_DefaultOpen))
			{
				if (_totalChunkCount > 1)
				{
					ImGui::Text("Sweep: %u / %u chunks complete, %u pending, %zu active (GPU0) + %zu (GPU1)",
						_completedChunkCount, _totalChunkCount,
						static_cast<uint32_t>(_pendingChunks.size()),
						_activeSystemsGpu0.size(), _activeSystemsGpu1.size());
				}
				ImGui::Text("VRAM: GPU0 %.2f / %.2f GB   GPU1 %.2f / %.2f GB",
					_vramUsedBytesGpu0 / 1073741824.0, _vramBudgetBytesPerGpu / 1073741824.0,
					_vramUsedBytesGpu1 / 1073741824.0, _vramBudgetBytesPerGpu / 1073741824.0);

				float completionFraction = (_lastSearchProgress.totalGridCount > 0)
					? static_cast<float>(_lastSearchProgress.totalGridCount - _lastSearchProgress.activeGridCount) / static_cast<float>(_lastSearchProgress.totalGridCount)
					: 0.0f;
				char progressOverlay[64];
				snprintf(progressOverlay, sizeof(progressOverlay), "%u / %u grids complete",
					_lastSearchProgress.totalGridCount - _lastSearchProgress.activeGridCount, _lastSearchProgress.totalGridCount);
				ImGui::ProgressBar(completionFraction, ImVec2(-1, 0), progressOverlay);

				ImGui::Text("Viable: %u  |  Possible: %u  |  Unviable: %u",
					_lastSearchProgress.savedViableCount, _lastSearchProgress.savedPossibleCount, _lastSearchProgress.savedUnviableCount);
			}

			DrawResultsOverTimeGraph(&_lastSearchProgress);

			// Grid heatmap / per-grid stats need one representative SearchSystem.
			// Pick the first active one on GPU0 (or GPU1 if GPU0 is empty) — the
			// pools can have multiple chunks in flight but the heatmap is a cross-
			// section of the current "which rule is each slot on" state, not aggregate.
			const SearchSystem* representativeSystem = nullptr;
			if (!_activeSystemsGpu0.empty()) representativeSystem = _activeSystemsGpu0.front().get();
			else if (!_activeSystemsGpu1.empty()) representativeSystem = _activeSystemsGpu1.front().get();

			if (representativeSystem)
			{
				DrawGridHeatmap(representativeSystem);
				DrawPerGridStatsTable(representativeSystem);
			}
		}

		// Rule-space heatmap is driven by the aggregator, which persists across all
		// chunks of the sweep and also through Complete state — so draw it outside
		// the (running || paused) gate so results stay visible after the sweep ends.
		if (_ruleAggregator.UniqueRuleCount() > 0)
			DrawRuleSpaceHeatmap();

		// --- Controls ---
		ImGui::Separator();
		if (isSearchRunning)
		{
			if (!_searchPaused)
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.7f, 0.2f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.8f, 0.3f, 1.0f));
				if (ImGui::Button("Pause Search"))
					services.guiSystem.InjectPauseRequest();
				ImGui::PopStyleColor(2);
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.7f, 0.2f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.8f, 0.3f, 1.0f));
				if (ImGui::Button("Resume Search"))
					services.guiSystem.InjectResumeRequest();
				ImGui::PopStyleColor(2);
			}

			ImGui::SameLine();

			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));
			if (ImGui::Button("Stop Search"))
				services.guiSystem.InjectStopRequest();
			ImGui::PopStyleColor(2);
		}
		else
		{
			bool hasNeighborhoodSelection = guiState.searchFaceNeighbors
				|| guiState.searchEdgeNeighbors
				|| guiState.searchCornerNeighbors;

			if (!hasNeighborhoodSelection)
				ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Select at least one of Faces / Edges / Corners");

			ImGui::BeginDisabled(!hasNeighborhoodSelection);
			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.2f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.8f, 0.3f, 1.0f));
			if (ImGui::Button("Start Search"))
				services.guiSystem.InjectStartRequest();
			ImGui::PopStyleColor(2);
			ImGui::EndDisabled();
		}

		services.guiSystem.EndModeFrame();
	}

	void SearchMode::DrawResultsOverTimeGraph(const SearchProgress* progress)
	{
		if (!ImGui::CollapsingHeader("Results Over Time"))
			return;

		_viableHistory.push_back(static_cast<float>(progress->savedViableCount));
		_possibleHistory.push_back(static_cast<float>(progress->savedPossibleCount));
		_unviableHistory.push_back(static_cast<float>(progress->savedUnviableCount));

		int count = static_cast<int>(_viableHistory.size());
		ImVec2 graphSize(ImGui::GetContentRegionAvail().x, 120);

		ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.2f, 0.9f, 0.2f, 1.0f));
		ImGui::PlotLines("##viable", _viableHistory.data(), count, 0, "Viable", 0, FLT_MAX, graphSize);
		ImGui::PopStyleColor();

		ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.9f, 0.9f, 0.2f, 1.0f));
		ImGui::PlotLines("##possible", _possibleHistory.data(), count, 0, "Possible", 0, FLT_MAX, graphSize);
		ImGui::PopStyleColor();

		ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.9f, 0.3f, 0.3f, 1.0f));
		ImGui::PlotLines("##unviable", _unviableHistory.data(), count, 0, "Unviable", 0, FLT_MAX, graphSize);
		ImGui::PopStyleColor();
	}

	void SearchMode::DrawGridHeatmap(const SearchSystem* searchSystem)
	{
		if (!ImGui::CollapsingHeader("Grid Heatmap"))
			return;

		uint32_t gridCount = searchSystem->GetGridCount();
		if (gridCount == 0) return;

		const auto& permutations = searchSystem->GetLastReadPermutations();
		const auto& gridInfos = searchSystem->GetLastReadGridInfos();
		const auto& gridConfigs = searchSystem->GetGridConfigs();

		uint32_t columns = static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<float>(gridCount))));
		uint32_t rows = (gridCount + columns - 1) / columns;

		float availableWidth = ImGui::GetContentRegionAvail().x;
		float cellSize = std::max(8.0f, std::min(24.0f, (availableWidth - columns * 2.0f) / columns));
		float padding = 2.0f;

		ImVec2 origin = ImGui::GetCursorScreenPos();
		ImDrawList* drawList = ImGui::GetWindowDrawList();

		for (uint32_t gridIndex = 0; gridIndex < gridCount; gridIndex++)
		{
			uint32_t col = gridIndex % columns;
			uint32_t row = gridIndex / columns;

			ImVec2 cellMin(origin.x + col * (cellSize + padding), origin.y + row * (cellSize + padding));
			ImVec2 cellMax(cellMin.x + cellSize, cellMin.y + cellSize);

			const auto& perm = permutations[gridIndex];
			const auto& info = gridInfos[gridIndex];
			bool isActive = (perm.survivalAndNeighborhoodRules != 0 || perm.birthAndMaxCellStateRules != 0);

			ImU32 color;
			if (!isActive)
				color = IM_COL32(80, 80, 80, 255);
			else if (info.status == 2)
				color = IM_COL32(50, 200, 50, 255);
			else if (info.status == 1)
				color = IM_COL32(200, 200, 50, 255);
			else
				color = IM_COL32(50, 100, 200, 255);

			drawList->AddRectFilled(cellMin, cellMax, color, 2.0f);

			if (ImGui::IsMouseHoveringRect(cellMin, cellMax))
			{
				std::string neighborhoodString;
				uint64_t flags = gridConfigs[gridIndex].neighborhoodFlags;
				if (flags & Simulation::FACE_NEIGHBORS_MASK) neighborhoodString += "F";
				if (flags & Simulation::EDGE_NEIGHBORS_MASK) { if (!neighborhoodString.empty()) neighborhoodString += ","; neighborhoodString += "E"; }
				if (flags & Simulation::CORNER_NEIGHBORS_MASK) { if (!neighborhoodString.empty()) neighborhoodString += ","; neighborhoodString += "C"; }
				if (flags & Simulation::WRAP_NEIGHBORS_MASK) { if (!neighborhoodString.empty()) neighborhoodString += ","; neighborhoodString += "W"; }

				ImGui::BeginTooltip();
				ImGui::Text("Grid %u", gridIndex);
				ImGui::Text("Max Cell State: %llu", gridConfigs[gridIndex].maxCellState);
				ImGui::Text("Neighborhood: %s", neighborhoodString.c_str());
				ImGui::Text("Status: %s", !isActive ? "Exhausted" : (info.status == 2 ? "Viable" : (info.status == 1 ? "Died" : "Running")));
				ImGui::Text("Ticks: %u", info.numberOfSimulationTicksSurvived);
				ImGui::EndTooltip();
			}
		}

		float totalHeight = rows * (cellSize + padding);
		ImGui::Dummy(ImVec2(availableWidth, totalHeight));
	}

	void SearchMode::DrawPerGridStatsTable(const SearchSystem* searchSystem)
	{
		if (!ImGui::CollapsingHeader("Grid Details"))
			return;

		uint32_t gridCount = searchSystem->GetGridCount();
		if (gridCount == 0) return;

		const auto& permutations = searchSystem->GetLastReadPermutations();
		const auto& gridInfos = searchSystem->GetLastReadGridInfos();
		const auto& gridConfigs = searchSystem->GetGridConfigs();

		ImGuiTableFlags tableFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
			ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;

		float tableHeight = std::min(300.0f, ImGui::GetContentRegionAvail().y);

		if (ImGui::BeginTable("GridStatsTable", 7, tableFlags, ImVec2(0, tableHeight)))
		{
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableSetupColumn("ID",                    ImGuiTableColumnFlags_WidthFixed, 30.0f);
			ImGui::TableSetupColumn("Max Cell State",        ImGuiTableColumnFlags_WidthFixed, 100.0f);
			ImGui::TableSetupColumn("Neighborhood",          ImGuiTableColumnFlags_WidthFixed, 100.0f);
			ImGui::TableSetupColumn("Status",                ImGuiTableColumnFlags_WidthFixed, 70.0f);
			ImGui::TableSetupColumn("Ticks",                 ImGuiTableColumnFlags_WidthFixed, 50.0f);
			ImGui::TableSetupColumn("Current Birth Rules",   ImGuiTableColumnFlags_WidthStretch);
			ImGui::TableSetupColumn("Current Survival Rules", ImGuiTableColumnFlags_WidthStretch);
			ImGui::TableHeadersRow();

			for (uint32_t gridIndex = 0; gridIndex < gridCount; gridIndex++)
			{
				const auto& perm = permutations[gridIndex];
				const auto& info = gridInfos[gridIndex];
				const auto& config = gridConfigs[gridIndex];
				bool isActive = (perm.survivalAndNeighborhoodRules != 0 || perm.birthAndMaxCellStateRules != 0);

				ImGui::TableNextRow();

				ImGui::TableNextColumn();
				ImGui::Text("%u", gridIndex);

				ImGui::TableNextColumn();
				ImGui::Text("%llu", config.maxCellState);

				ImGui::TableNextColumn();
				{
					std::string neighborhoodString;
					if (config.neighborhoodFlags & Simulation::FACE_NEIGHBORS_MASK) neighborhoodString += "F";
					if (config.neighborhoodFlags & Simulation::EDGE_NEIGHBORS_MASK)
					{
						if (!neighborhoodString.empty()) neighborhoodString += ",";
						neighborhoodString += "E";
					}
					if (config.neighborhoodFlags & Simulation::CORNER_NEIGHBORS_MASK)
					{
						if (!neighborhoodString.empty()) neighborhoodString += ",";
						neighborhoodString += "C";
					}
					if (config.neighborhoodFlags & Simulation::WRAP_NEIGHBORS_MASK)
					{
						if (!neighborhoodString.empty()) neighborhoodString += ",";
						neighborhoodString += "W";
					}
					ImGui::Text("%s", neighborhoodString.c_str());
				}

				ImGui::TableNextColumn();
				if (!isActive)
					ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Done");
				else if (info.status == 2)
					ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Viable");
				else if (info.status == 1)
					ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "Died");
				else
					ImGui::TextColored(ImVec4(0.3f, 0.5f, 0.9f, 1.0f), "Running");

				ImGui::TableNextColumn();
				ImGui::Text("%u", info.numberOfSimulationTicksSurvived);

				ImGui::TableNextColumn();
				if (isActive)
				{
					std::string birthStr = GuiSystem::BitmaskToRuleString(
						perm.birthAndMaxCellStateRules & Simulation::EXISTENCE_PERMUTATION_BIT_MASK);
					ImGui::Text("%s", birthStr.c_str());
				}
				else
				{
					ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "-");
				}

				ImGui::TableNextColumn();
				if (isActive)
				{
					std::string survivalStr = GuiSystem::BitmaskToRuleString(
						perm.survivalAndNeighborhoodRules & Simulation::EXISTENCE_PERMUTATION_BIT_MASK);
					ImGui::Text("%s", survivalStr.c_str());
				}
				else
				{
					ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "-");
				}
			}

			ImGui::EndTable();
		}
	}

	// Shared helper: build the per-chunk file prefix and log. Both GPU spawn paths use this.
	static std::string BuildChunkPrefix(const std::string& filePrefixBase,
		uint32_t chunkOrdinal, const ChunkConfiguration& chunkConfig)
	{
		return filePrefixBase
			+ "_c" + std::to_string(chunkOrdinal)
			+ "_g" + std::to_string(chunkConfig.gridDimensionX) + "x" + std::to_string(chunkConfig.gridDimensionY) + "x" + std::to_string(chunkConfig.gridDimensionZ)
			+ "_d" + std::to_string(chunkConfig.spawnDensityIndex)
			+ "_s" + std::to_string(chunkConfig.runSeedIndex);
	}

	// Harvest one SearchSystem's in-memory results into the shared rule aggregator.
	// Called from the harvest loop before the SearchSystem is destroyed.
	void SearchMode_HarvestResultsToAggregator(RuleAggregator& aggregator, const SearchSystem& searchSystem)
	{
		uint32_t shapeId = searchSystem.GetChunkConfiguration().shapeId;
		for (const auto& result : searchSystem.GetViableResults())
			aggregator.Record(result.permutation, result.ticksSurvived,
				RuleAggregator::Category::Viable, shapeId);
		for (const auto& result : searchSystem.GetPossibleResults())
			aggregator.Record(result.permutation, result.ticksSurvived,
				RuleAggregator::Category::Possible, shapeId);
		for (const auto& result : searchSystem.GetUnviableResults())
			aggregator.Record(result.permutation, result.ticksSurvived,
				RuleAggregator::Category::Unviable, shapeId);
	}

	bool SearchMode::TrySpawnChunkOnGpu0(ModeServices& services,
		const ChunkConfiguration& chunkConfig, uint64_t chunkVramBytes)
	{
		if (_vramUsedBytesGpu0 + chunkVramBytes > _vramBudgetBytesPerGpu)
			return false;

		uint32_t chunkOrdinal = static_cast<uint32_t>(_activeSystemsGpu0.size() + _activeSystemsGpu1.size()
			+ _completedChunkCount + 1);
		std::string chunkPrefix = BuildChunkPrefix(_sweepContext.filePrefixBase, chunkOrdinal, chunkConfig);

		vk::ShaderModule simulationShaderModule = _searchShaderGenerator->GetShaderModule(
			*_sweepContext.shape, chunkConfig.gridDimensionX, chunkConfig.gridDimensionY, chunkConfig.gridDimensionZ,
			static_cast<uint32_t>(_sweepContext.maxRuleBits),
			static_cast<uint32_t>(_sweepContext.simulationWorkgroupSize));
		vk::ShaderModule managerShaderModule = _searchShaderGenerator->GetManagerShaderModule(
			chunkConfig.gridDimensionX, chunkConfig.gridDimensionY, chunkConfig.gridDimensionZ, _sweepContext.maxTicksToSurvive,
			static_cast<uint32_t>(_sweepContext.maxRuleBits));

		bool simulationUses3DDispatch = _sweepContext.shape && _sweepContext.shape->GetName() == "cube";
		auto searchSystem = std::make_unique<SearchSystem>(
			services.deviceContext, _sweepContext.gridConfigs, chunkConfig,
			simulationShaderModule, managerShaderModule,
			_sweepContext.outputFolder, chunkPrefix + "_gpu0",
			static_cast<uint32_t>(_sweepContext.maxRuleBits),
			static_cast<uint32_t>(_sweepContext.simulationWorkgroupSize),
			simulationUses3DDispatch);
		searchSystem->SubmitNextTick();

		_activeSystemsGpu0.push_back(std::move(searchSystem));
		_activeVramGpu0.push_back(chunkVramBytes);
		_vramUsedBytesGpu0 += chunkVramBytes;

		LOG_INFO("Spawned chunk {}/{} on GPU0: grid {}x{}x{} (VRAM now {:.2f}/{:.2f} GB, {} chunks active)",
			chunkOrdinal, _totalChunkCount,
			chunkConfig.gridDimensionX, chunkConfig.gridDimensionY, chunkConfig.gridDimensionZ,
			_vramUsedBytesGpu0 / 1073741824.0, _vramBudgetBytesPerGpu / 1073741824.0,
			_activeSystemsGpu0.size());
		return true;
	}

	bool SearchMode::TrySpawnChunkOnGpu1(ModeServices& services,
		const ChunkConfiguration& chunkConfig, uint64_t chunkVramBytes)
	{
		if (!services.secondaryDeviceContext) return false;
		if (_vramUsedBytesGpu1 + chunkVramBytes > _vramBudgetBytesPerGpu)
			return false;

		uint32_t chunkOrdinal = static_cast<uint32_t>(_activeSystemsGpu0.size() + _activeSystemsGpu1.size()
			+ _completedChunkCount + 1);
		std::string chunkPrefix = BuildChunkPrefix(_sweepContext.filePrefixBase, chunkOrdinal, chunkConfig);

		vk::ShaderModule simulationShaderModule = _searchShaderGeneratorGpu1->GetShaderModule(
			*_sweepContext.shape, chunkConfig.gridDimensionX, chunkConfig.gridDimensionY, chunkConfig.gridDimensionZ,
			static_cast<uint32_t>(_sweepContext.maxRuleBits),
			static_cast<uint32_t>(_sweepContext.simulationWorkgroupSize));
		vk::ShaderModule managerShaderModule = _searchShaderGeneratorGpu1->GetManagerShaderModule(
			chunkConfig.gridDimensionX, chunkConfig.gridDimensionY, chunkConfig.gridDimensionZ, _sweepContext.maxTicksToSurvive,
			static_cast<uint32_t>(_sweepContext.maxRuleBits));

		bool simulationUses3DDispatch = _sweepContext.shape && _sweepContext.shape->GetName() == "cube";
		auto searchSystem = std::make_unique<SearchSystem>(
			*services.secondaryDeviceContext, _sweepContext.gridConfigs, chunkConfig,
			simulationShaderModule, managerShaderModule,
			_sweepContext.outputFolder, chunkPrefix + "_gpu1",
			static_cast<uint32_t>(_sweepContext.maxRuleBits),
			static_cast<uint32_t>(_sweepContext.simulationWorkgroupSize),
			simulationUses3DDispatch);
		searchSystem->SubmitNextTick();

		_activeSystemsGpu1.push_back(std::move(searchSystem));
		_activeVramGpu1.push_back(chunkVramBytes);
		_vramUsedBytesGpu1 += chunkVramBytes;

		LOG_INFO("Spawned chunk {}/{} on GPU1: grid {}x{}x{} (VRAM now {:.2f}/{:.2f} GB, {} chunks active)",
			chunkOrdinal, _totalChunkCount,
			chunkConfig.gridDimensionX, chunkConfig.gridDimensionY, chunkConfig.gridDimensionZ,
			_vramUsedBytesGpu1 / 1073741824.0, _vramBudgetBytesPerGpu / 1073741824.0,
			_activeSystemsGpu1.size());
		return true;
	}

	// Sweeps one pool: any SearchSystem whose IsRunning() returns false gets flushed,
	// its results harvested, its VRAM released, and is removed. Multiple systems can
	// complete on the same frame.
	static void HarvestPool(DeviceContext& deviceContext,
		std::vector<std::unique_ptr<SearchSystem>>& pool,
		std::vector<uint64_t>& perSystemVram,
		uint64_t& vramUsed,
		uint32_t& completedChunkCount,
		uint32_t totalChunkCount,
		RuleAggregator& aggregator,
		const char* gpuLabel)
	{
		for (size_t index = 0; index < pool.size(); )
		{
			if (!pool[index]->IsRunning())
			{
				deviceContext.GetDevice().waitIdle();
				pool[index]->FlushProgress();
				pool[index]->WriteToDisk();
				SearchMode_HarvestResultsToAggregator(aggregator, *pool[index]);

				uint64_t freed = perSystemVram[index];
				vramUsed -= std::min(vramUsed, freed);
				completedChunkCount++;

				LOG_INFO("Chunk complete on {} ({}/{}). VRAM freed {:.2f} GB.",
					gpuLabel, completedChunkCount, totalChunkCount, freed / 1073741824.0);

				pool.erase(pool.begin() + index);
				perSystemVram.erase(perSystemVram.begin() + index);
			}
			else
			{
				index++;
			}
		}
	}

	void SearchMode::HarvestCompletedOnGpu0(ModeServices& services)
	{
		HarvestPool(services.deviceContext, _activeSystemsGpu0, _activeVramGpu0,
			_vramUsedBytesGpu0, _completedChunkCount, _totalChunkCount, _ruleAggregator, "GPU0");
	}

	void SearchMode::HarvestCompletedOnGpu1(ModeServices& services)
	{
		if (!services.secondaryDeviceContext) return;
		HarvestPool(*services.secondaryDeviceContext, _activeSystemsGpu1, _activeVramGpu1,
			_vramUsedBytesGpu1, _completedChunkCount, _totalChunkCount, _ruleAggregator, "GPU1");
	}

	void SearchMode::FinalizeAllChunks(ModeServices& services)
	{
		HarvestCompletedOnGpu0(services);
		HarvestCompletedOnGpu1(services);

		// Force-harvest anything still running — used for Stop / Exit paths.
		if (!_activeSystemsGpu0.empty())
		{
			services.deviceContext.GetDevice().waitIdle();
			for (size_t index = 0; index < _activeSystemsGpu0.size(); index++)
			{
				_activeSystemsGpu0[index]->FlushProgress();
				_activeSystemsGpu0[index]->WriteToDisk();
				SearchMode_HarvestResultsToAggregator(_ruleAggregator, *_activeSystemsGpu0[index]);
			}
			_activeSystemsGpu0.clear();
			_activeVramGpu0.clear();
			_vramUsedBytesGpu0 = 0;
		}
		if (!_activeSystemsGpu1.empty() && services.secondaryDeviceContext)
		{
			services.secondaryDeviceContext->GetDevice().waitIdle();
			for (size_t index = 0; index < _activeSystemsGpu1.size(); index++)
			{
				_activeSystemsGpu1[index]->FlushProgress();
				_activeSystemsGpu1[index]->WriteToDisk();
				SearchMode_HarvestResultsToAggregator(_ruleAggregator, *_activeSystemsGpu1[index]);
			}
			_activeSystemsGpu1.clear();
			_activeVramGpu1.clear();
			_vramUsedBytesGpu1 = 0;
		}
	}

	void SearchMode::DrawRuleSpaceHeatmap()
	{
		if (!ImGui::CollapsingHeader("Rule Space Heatmap"))
			return;

		// Bucket aggregator entries by (popcount(birth), popcount(survival)). Each bucket
		// holds the mean viability rate of all rules with that pair of bitmask population
		// counts. Intuitively: "how well do rules with N births and M survivals do?"
		struct Bucket { uint32_t count = 0; float sumViabilityRate = 0.0f; };

		// Bucket grid sized to 61 × 61 (covers the 60-bit rule mask). Shrinks at draw time
		// to the actual occupied range so small-neighbor searches get a dense plot.
		static constexpr int MAX_BITS = 61;
		std::array<std::array<Bucket, MAX_BITS>, MAX_BITS> buckets{};
		int maxOccupiedX = 0;
		int maxOccupiedY = 0;
		uint32_t totalRulesInAggregate = static_cast<uint32_t>(_ruleAggregator.UniqueRuleCount());

		if (totalRulesInAggregate == 0)
		{
			ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
				"No aggregate data yet — run a sweep or wait for the first chunk to complete.");
			return;
		}

		// The aggregator doesn't currently expose its map directly, so we expose a
		// minimal const accessor here via #if -- but since ruleAggregator.h is a header,
		// and we've kept the private map inaccessible, we instead add a friend-free
		// accessor on RuleAggregator itself. We do that inline here:
		_ruleAggregator.ForEachRule([&](const RuleAggregator::RuleKey& key,
			const RuleAggregator::RuleStats& stats)
		{
			int birthBits = 0;
			int survivalBits = 0;
			uint64_t b = key.birthBitmask;
			uint64_t s = key.survivalBitmask;
			while (b) { birthBits += (b & 1ULL); b >>= 1; }
			while (s) { survivalBits += (s & 1ULL); s >>= 1; }
			if (birthBits >= MAX_BITS || survivalBits >= MAX_BITS) return;

			Bucket& bucket = buckets[survivalBits][birthBits];
			bucket.count++;
			bucket.sumViabilityRate += stats.totalRuns > 0
				? static_cast<float>(stats.viableRuns) / static_cast<float>(stats.totalRuns)
				: 0.0f;
			if (birthBits    > maxOccupiedX) maxOccupiedX = birthBits;
			if (survivalBits > maxOccupiedY) maxOccupiedY = survivalBits;
		});

		int cellsX = maxOccupiedX + 1;
		int cellsY = maxOccupiedY + 1;

		ImVec2 available = ImGui::GetContentRegionAvail();
		float cellSide = std::max(8.0f, std::min(24.0f, std::min(available.x / cellsX, 300.0f / cellsY)));
		ImVec2 origin = ImGui::GetCursorScreenPos();
		ImDrawList* drawList = ImGui::GetWindowDrawList();

		// Background
		drawList->AddRectFilled(origin,
			ImVec2(origin.x + cellsX * cellSide, origin.y + cellsY * cellSide),
			IM_COL32(18, 18, 24, 255));

		for (int y = 0; y < cellsY; y++)
		{
			for (int x = 0; x < cellsX; x++)
			{
				const Bucket& bucket = buckets[y][x];
				if (bucket.count == 0) continue;
				float meanRate = bucket.sumViabilityRate / static_cast<float>(bucket.count);
				// Red (0.0) → yellow (0.5) → green (1.0) gradient.
				uint8_t r = static_cast<uint8_t>(std::clamp(1.0f - meanRate, 0.0f, 1.0f) * 255.0f);
				uint8_t g = static_cast<uint8_t>(std::clamp(meanRate, 0.0f, 1.0f) * 255.0f);
				ImVec2 topLeft(origin.x + x * cellSide, origin.y + (cellsY - 1 - y) * cellSide);
				drawList->AddRectFilled(topLeft,
					ImVec2(topLeft.x + cellSide, topLeft.y + cellSide),
					IM_COL32(r, g, 40, 255));
			}
		}

		// Reserve space so subsequent ImGui widgets don't overlap the drawn grid.
		ImGui::Dummy(ImVec2(cellsX * cellSide, cellsY * cellSide));
		ImGui::Text("X = birth bit count (0..%d)   Y = survival bit count (0..%d)   rules aggregated: %u",
			cellsX - 1, cellsY - 1, totalRulesInAggregate);
		ImGui::Text("Color: red = rule cohort dies, green = rule cohort survives.");

		// Hover inspection
		ImVec2 mousePos = ImGui::GetMousePos();
		if (mousePos.x >= origin.x && mousePos.x < origin.x + cellsX * cellSide
			&& mousePos.y >= origin.y && mousePos.y < origin.y + cellsY * cellSide)
		{
			int hoverX = static_cast<int>((mousePos.x - origin.x) / cellSide);
			int hoverY = cellsY - 1 - static_cast<int>((mousePos.y - origin.y) / cellSide);
			if (hoverX >= 0 && hoverX < cellsX && hoverY >= 0 && hoverY < cellsY)
			{
				const Bucket& bucket = buckets[hoverY][hoverX];
				if (bucket.count > 0)
				{
					float meanRate = bucket.sumViabilityRate / static_cast<float>(bucket.count);
					ImGui::BeginTooltip();
					ImGui::Text("Birth bits: %d   Survival bits: %d", hoverX, hoverY);
					ImGui::Text("Rules in bucket: %u   Mean viabilityRate: %.3f", bucket.count, meanRate);
					ImGui::EndTooltip();
				}
			}
		}
	}

} // namespace Cave
