#pragma once

#include "applicationModeHandler.h"
#include "../common/structs.h"
#include "../common/ruleAggregator.h"
#include "../vulkan/systems/searchSystem.h"
#include "../vulkan/searchShaderGenerator.h"
#include "../shapes/shape.h"

#include <memory>
#include <vector>
#include <string>

namespace Cave
{
	class SearchMode : public ApplicationModeHandler
	{
	public:
		SearchMode() = default;
		~SearchMode() override;

		SearchMode(const SearchMode&) = delete;
		SearchMode& operator=(const SearchMode&) = delete;

		void Enter(ModeServices& services) override;
		bool OnFrame(ModeServices& services, float deltaTimeInSeconds) override;
		void Exit(ModeServices& services) override;

	private:
		enum class State { Configuring, Running, Complete };
		State _state = State::Configuring;

		// Pool of concurrent SearchSystems per GPU. The scheduler spawns new systems
		// from the pending-chunks queue whenever free VRAM allows, and harvests them
		// on completion. Pool size is bounded by the user's VRAM budget, not a fixed
		// count — e.g., a 49³ sweep (1.2 GB/chunk) can run 30+ concurrent chunks per
		// GPU, while a 100³ sweep (10 GB/chunk) fits only 4.
		std::vector<std::unique_ptr<SearchSystem>> _activeSystemsGpu0;
		std::vector<std::unique_ptr<SearchSystem>> _activeSystemsGpu1;
		uint64_t _vramUsedBytesGpu0 = 0;
		uint64_t _vramUsedBytesGpu1 = 0;
		uint64_t _vramBudgetBytesPerGpu = 0;

		// Parallel ChunkConfigurations for each active system, since SearchSystem only
		// exposes its own ChunkConfiguration; we track per-active the estimated VRAM
		// so harvest can decrement the per-GPU counter.
		std::vector<uint64_t> _activeVramGpu0;
		std::vector<uint64_t> _activeVramGpu1;

		std::unique_ptr<SearchShaderGenerator> _searchShaderGenerator;
		std::unique_ptr<SearchShaderGenerator> _searchShaderGeneratorGpu1;

		bool _sessionRunning = false;
		bool _searchPaused = false;
		SearchProgress _lastSearchProgress{};
		uint32_t _currentFrameIndex = 0;

		// Dashboard history (moved from GuiSystem)
		std::vector<float> _viableHistory;
		std::vector<float> _possibleHistory;
		std::vector<float> _unviableHistory;

		// Sweep state — a sweep is an ordered list of ChunkConfigurations produced from the
		// user's sweep spec (grid-size range × density range × seed count × ...). Chunks run
		// serially; after each chunk completes, the next is spun up.
		struct SweepContext
		{
			std::shared_ptr<Shape> shape;
			std::vector<SearchGridConfig> gridConfigs;          // shared across all chunks in the sweep
			int maxTicksToSurvive = 0;
			int maxRuleBits = 0;                                // 0 = unbounded, K>0 caps popcount(birth) and popcount(survival)
			int simulationWorkgroupSize = 64;                   // threads per workgroup for the simulation kernel
			int minMaxCellState = 0;
			int maxMaxCellState = 0;
			uint32_t spawnAreaDimensionX = 0;
			uint32_t spawnAreaDimensionY = 0;
			uint32_t spawnAreaDimensionZ = 0;
			uint32_t spawnMode = 1;
			uint32_t spawnDensityIndex = 15;
			std::string outputFolder;
			std::string filePrefixBase;                         // "_chunk_N_gXxYxZ" gets appended per chunk
		};
		SweepContext _sweepContext;
		std::vector<ChunkConfiguration> _pendingChunks;
		uint32_t _completedChunkCount = 0;
		uint32_t _totalChunkCount = 0;
		RuleAggregator _ruleAggregator;

		void RetrieveTimestamps(ModeServices& services);
		void BuildGui(ModeServices& services);
		void DrawResultsOverTimeGraph(const SearchProgress* progress);
		void DrawGridHeatmap(const SearchSystem* searchSystem);
		void DrawPerGridStatsTable(const SearchSystem* searchSystem);
		void DrawRuleSpaceHeatmap();

		// Attempts to spawn one chunk on the chosen GPU, updating the VRAM counter.
		// Returns true if spawned, false if VRAM was already exhausted (caller should
		// wait for a harvest before retrying). Called from the scheduler loop.
		bool TrySpawnChunkOnGpu0(ModeServices& services, const ChunkConfiguration& chunkConfig, uint64_t chunkVramBytes);
		bool TrySpawnChunkOnGpu1(ModeServices& services, const ChunkConfiguration& chunkConfig, uint64_t chunkVramBytes);

		// Harvest completed SearchSystems on one GPU: flush their output, feed results
		// to _ruleAggregator, free the associated VRAM, remove from the pool.
		void HarvestCompletedOnGpu0(ModeServices& services);
		void HarvestCompletedOnGpu1(ModeServices& services);

		// Teardown path for all remaining systems (Stop / Exit). Used for a clean
		// shutdown even if the sweep wasn't driven to completion.
		void FinalizeAllChunks(ModeServices& services);
	};

} // namespace Cave
