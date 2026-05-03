#pragma once

#include <vulkan/vulkan.hpp>
#include <memory>
#include <vector>
#include <string>
#include <fstream>

#include "../deviceContext.h"
#include "../buffer.h"
#include "../pipelines/computePipeline.h"
#include "../descriptors.h"
#include "../../common/structs.h"
#include "../../common/queryManager.h"

namespace Cave
{
	// Per-grid configuration: fixed (maxCellState, neighborhoodConfig) in MSBs,
	// iterates through birth/survival rules in the lower 60 bits.
	struct SearchGridConfig
	{
		uint64_t maxCellState;         // 1-15, stored in upper 4 bits
		uint64_t neighborhoodFlags;    // face/edge/corner/wrap in upper 4 bits
		uint64_t maxPermutation;       // max value for the lower 60 rule bits (depends on neighbor count)
		uint32_t neighborCount;        // total number of neighbors for this config
	};

	// CPU-readable snapshot of search progress — polled each frame for the UI.
	struct SearchProgress
	{
		SimulationParameters currentPermutation;
		GridInfo currentGridInfo;
		uint32_t savedViableCount;
		uint32_t savedPossibleCount;
		uint32_t savedUnviableCount;
		uint32_t totalGridCount;
		uint32_t activeGridCount;
	};

	// Runs N cellular automata simulations in parallel on the GPU.
	// Each grid gets a fixed (maxCellState, neighborhoodConfig) combination
	// and iterates through all valid birth/survival rule permutations.
	//
	// Per tick: dispatches N simulation shaders then N manager shaders,
	// separated by a pipeline barrier. Single command buffer, single submission.
	//
	// Iteration encoding for the lower 60 bits of birth/survival permutation fields:
	//   K == 0 (no bit cap): lower 60 bits ARE the rule bitmask, iterating 1..(2^N-1).
	//   K  > 0 (bit cap):    lower 60 bits are a 1-based INDEX into _slotValidBitmasks
	//                         (CPU mirror) / validBitmasksBuffer (GPU SSBO). The shader
	//                         looks up the actual bitmask before evaluating the rule.
	//                         maxPermutation = validBitmasks.size(), so iteration is
	//                         O(1) per advance with no skip-loop and no invalid rules.
	class SearchSystem
	{
	private:
		DeviceContext &_deviceContext;
		uint32_t _gridCount;
		uint32_t _totalCellCount;

		// ---- Compute pipelines (shared across all grids) ----
		std::shared_ptr<Descriptor> _simulationDescriptor;
		std::unique_ptr<ComputePipeline> _simulationPipeline;

		std::shared_ptr<Descriptor> _managerDescriptor;
		std::unique_ptr<ComputePipeline> _managerPipeline;

		// ---- Per-grid GPU buffers ----
		// Simulation shader bindings (per grid option):
		//   0: GridPermutations (SimulationParameters, readonly)
		//   1: GridSnapshot (read-write, atomic)
		//   2: CellsIn (readonly)
		//   3: CellsOut (read-write)
		//   4: ValidBitmasks (readonly uint64[]) — only used when MAX_RULE_BITS > 0;
		//      lower 60 bits of permutation are 1-based indices into this buffer.
		//
		// Manager shader bindings (per grid option):
		//   0: InitialCellsData
		//   1: CellsData (same as CellsIn)
		//   2: GridSnapshot
		//   3: GridMaxPermutation
		//   4: GridPermutations
		//   5: GridPermutationsCopy (CPU-readable)
		//   6: GridInfo
		//   7: GridInfoCopy (CPU-readable)
		struct GridBuffers
		{
			std::shared_ptr<Buffer> gridPermutationsBuffer;
			std::shared_ptr<Buffer> gridSnapshotBuffer;
			std::shared_ptr<Buffer> cellsInBuffer;
			std::shared_ptr<Buffer> cellsOutBuffer;
			std::shared_ptr<Buffer> initialCellsBuffer;
			std::shared_ptr<Buffer> gridMaxPermutationBuffer;
			std::shared_ptr<Buffer> gridPermutationsCopyBuffer;
			std::shared_ptr<Buffer> gridInfoBuffer;
			std::shared_ptr<Buffer> gridInfoCopyBuffer;
			// Work-stealing: survival-axis upper bound. By default equal to
			// maxPermutation (so the slot iterates its full range); the CPU work-stealer
			// narrows this when it splits a slot's remaining survival range with an
			// idle slot that volunteered to help.
			std::shared_ptr<Buffer> gridSurvivalEndPermutationBuffer;
			// Per-slot enumeration of valid bitmasks for K-bit-capped search. Sized to fit
			// the LARGEST possible enumeration across all neighborhoods present in this
			// chunk, so a steal that re-targets this slot to a different neighborhood
			// can re-upload without reallocating. Allocated as a 1-element dummy when
			// K == 0 (descriptor binding stays valid; shader never reads it).
			std::shared_ptr<Buffer> validBitmasksBuffer;
		};
		std::vector<GridBuffers> _gridBuffers;
		std::vector<SearchGridConfig> _gridConfigs;
		ChunkConfiguration _chunkConfig;

		// CPU mirror of each slot's GPU-side survivalEndPermutation. Initially equal
		// to the slot's maxPermutation; work-stealing narrows it when an idle slot
		// takes the upper half of a busy slot's survival range.
		std::vector<uint64_t> _slotSurvivalEndPermutations;

		// CPU mirror of each slot's current GPU-side maxPermutation. Starts equal
		// to the slot's original _gridConfigs[i].maxPermutation but changes when a
		// slot steals work with a different neighborhood (different neighborCount
		// → different maxPermutation). Reading busyConfig.maxPermutation directly
		// would be stale once the slot has already stolen work once.
		std::vector<uint64_t> _slotMaxPermutations;

		// CPU mirror of each slot's currently-active validBitmasks list (the K-bit-capped
		// enumeration the GPU buffer is populated with). When K == 0, every entry is
		// empty (the simulation shader's K==0 path doesn't touch the buffer). On a steal
		// across different neighborhoods, the idle slot adopts the donor's list.
		std::vector<std::vector<uint64_t>> _slotValidBitmasks;

		// Size in elements of every slot's validBitmasksBuffer. Computed at init from
		// the max validBitmasks count across all neighborhoods present in this chunk's
		// _gridConfigs (so steals between neighborhoods always fit). 1 when K == 0.
		uint32_t _validBitmasksBufferElementCount = 1;

		// Max bits set per rule bitmask cap for this chunk's search. 0 = unbounded.
		// Set at construction; immutable afterward. When > 0, iteration uses 1-based
		// INDICES into _slotValidBitmasks; when 0, iteration uses raw bitmasks.
		uint32_t _maxRuleBits = 0;

		// Simulation kernel workgroup size (threads per local block). Hardcoded into
		// the generated shader's `layout(local_size_x = N)` and into the host-side
		// dispatch-count math. Lower values = more workgroups in flight per dispatch
		// = better SM coverage when each chunk's grid is small.
		uint32_t _simulationWorkgroupSize = 64;

		// Cube simulation shader uses 3D dispatch with fixed (8, 8, 4) numthreads and
		// cooperative tile loading. ERD still uses 1D dispatch keyed off
		// _simulationWorkgroupSize. This flag picks which dispatch shape to record.
		bool _simulationUses3DDispatch = false;

		// Cumulative work-stealing statistics, exposed via GetStealStatistics(). Zeroed
		// at SearchSystem construction.
		uint64_t _workStealCount = 0;
		uint64_t _workStealTotalSurvivalRangeMoved = 0;

		// ---- Command resources ----
		vk::CommandPool _computeCommandPool;
		vk::CommandBuffer _commandBuffer;
		vk::Fence _tickFence;

		// ---- Per-grid CPU-side state ----
		std::vector<SimulationParameters> _lastReadPermutations;
		std::vector<GridInfo> _lastReadGridInfos;
		std::vector<SimulationParameters> _lastSavedPermutations; // track saved to avoid duplicates
		std::vector<bool> _gridsNeedingCellReset; // grids that completed and need initialCells → cellsIn copy

		// ---- In-memory result caches (written to disk once at end) ----
	public:
		struct SearchResult
		{
			SimulationParameters permutation;
			uint32_t ticksSurvived;
			uint32_t totalBirths = 0;       // accumulated across the run
			uint32_t totalDeaths = 0;       // accumulated across the run
			uint32_t maxAliveCount = 0;     // peak alive cell count seen during the run
			uint32_t finalAliveCount = 0;   // alive cell count at the last tick
		};
	private:
		std::vector<SearchResult> _viableResults;
		std::vector<SearchResult> _possibleResults;
		std::vector<SearchResult> _unviableResults;

		std::string _viableOutputPath;
		std::string _possibleOutputPath;
		std::string _unviableOutputPath;

		// JSONL append-as-we-go output files. Each result is written as one compact
		// JSON object per line, flushed immediately, so crashes / forced kills /
		// long-running searches do not lose cached results.
		std::ofstream _viableFile;
		std::ofstream _possibleFile;
		std::ofstream _unviableFile;

	private:
		void BuildBuffers(const std::vector<SearchGridConfig> &gridConfigs);
		void BuildSimulationPipeline(vk::ShaderModule simulationShaderModule);
		void BuildManagerPipeline(vk::ShaderModule managerShaderModule);
		void RecordCommandBuffer();

		void PollCopyBuffers();
		void CacheCompletedResults();
		void AppendResultLine(std::ofstream &outputFile, const SearchResult &result);

		// Grid-level work stealing — invoked after CacheCompletedResults has
		// processed a slot's completed permutation and marked it "needing reset".
		// For each idle slot, scans active slots with the same maxCellState (so the
		// initialCells buffer is directly reusable) and transfers half of the busiest
		// one's remaining survival range to the idle slot.
		void AttemptWorkStealingForIdleSlots();

		// Core steal loop: iterate THIS SearchSystem's idle slots, find the busiest
		// compatible slot in `slotSource`, split its survival range, hand the upper
		// half to the idle slot. `slotSource == *this` → within-chunk (grid-level)
		// stealing. `slotSource == otherChunk` → cross-chunk (pool-level) stealing.
		uint32_t TryStealIntoIdleSlotsFromSource(SearchSystem& slotSource);

	public:
		SearchSystem(DeviceContext &deviceContext,
					 const std::vector<SearchGridConfig> &gridConfigs,
					 const ChunkConfiguration &chunkConfig,
					 vk::ShaderModule simulationShaderModule,
					 vk::ShaderModule managerShaderModule,
					 const std::string &outputFolderPath,
					 const std::string &filePrefix,
					 uint32_t maxRuleBits,
					 uint32_t simulationWorkgroupSize,
					 bool simulationUses3DDispatch);
		~SearchSystem();

		SearchSystem(const SearchSystem &) = delete;
		SearchSystem &operator=(const SearchSystem &) = delete;

		// Returns true if the previous tick's GPU work has completed (non-blocking).
		bool IsTickComplete();

		// Records and submits the next tick's command buffer to the GPU.
		void SubmitNextTick(std::shared_ptr<QueryManager> queryManager = nullptr, vk::QueryPool queryPool = {});

		// Reads copy buffers and caches completed results.
		// Call AFTER IsTickComplete() returns true and BEFORE SubmitNextTick().
		void FlushProgress();

		bool IsRunning();
		void WriteToDisk();
		SearchProgress GetProgress() const;

		uint32_t GetGridCount() const { return _gridCount; }

		uint32_t GetMaxRuleBits() const { return _maxRuleBits; }
		const std::vector<SimulationParameters>& GetLastReadPermutations() const { return _lastReadPermutations; }
		const std::vector<GridInfo>& GetLastReadGridInfos() const { return _lastReadGridInfos; }
		const std::vector<SearchGridConfig>& GetGridConfigs() const { return _gridConfigs; }
		const ChunkConfiguration& GetChunkConfiguration() const { return _chunkConfig; }
		const std::vector<SearchResult>& GetViableResults() const { return _viableResults; }
		const std::vector<SearchResult>& GetPossibleResults() const { return _possibleResults; }
		const std::vector<SearchResult>& GetUnviableResults() const { return _unviableResults; }

		// Cumulative work-stealing stats — number of steals performed by this
		// SearchSystem, and total sum of survival-range widths moved between slots.
		uint64_t GetStealCount() const { return _workStealCount; }
		uint64_t GetStolenSurvivalRange() const { return _workStealTotalSurvivalRangeMoved; }

		// Pool / chunk-level stealing helpers. A SearchSystem is "idle" when at least
		// one of its slots has a NULL permutation (slot exhausted its range and is
		// available for reuse). "Busy" means at least one slot still has work.
		bool HasIdleSlots() const;
		bool HasBusySlots() const;

		// Fill this SearchSystem's idle slots with work stolen from another SearchSystem's
		// busy slots. Caller is responsible for ensuring both SearchSystems share grid
		// dimensions (different dims = incompatible buffer sizes). Returns number of
		// steals performed. The donor's slots remain running with narrowed survival
		// ranges; this SearchSystem's idle slots come alive with the upper halves.
		uint32_t TryStealFromOtherChunk(SearchSystem& donor);
	};
}
