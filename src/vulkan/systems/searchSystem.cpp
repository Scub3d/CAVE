#include "searchSystem.h"
#include "../../common/logger.h"
#include "../../common/permutationMath.h"
#include "../../simulation/simulation.h"

#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace Cave
{
	SearchSystem::SearchSystem(DeviceContext &deviceContext,
							   const std::vector<SearchGridConfig> &gridConfigs,
							   const ChunkConfiguration &chunkConfig,
							   vk::ShaderModule simulationShaderModule,
							   vk::ShaderModule managerShaderModule,
							   const std::string &outputFolderPath,
							   const std::string &filePrefix,
							   uint32_t maxRuleBits,
							   uint32_t simulationWorkgroupSize,
							   bool simulationUses3DDispatch)
		: _deviceContext{deviceContext},
		  _gridCount{static_cast<uint32_t>(gridConfigs.size())},
		  _totalCellCount{chunkConfig.gridDimensionX * chunkConfig.gridDimensionY * chunkConfig.gridDimensionZ},
		  _gridConfigs{gridConfigs},
		  _chunkConfig{chunkConfig},
		  _maxRuleBits{maxRuleBits},
		  _simulationWorkgroupSize{simulationWorkgroupSize},
		  _simulationUses3DDispatch{simulationUses3DDispatch}
	{
		_computeCommandPool = _deviceContext.GetComputeCommandPool();
		_tickFence = _deviceContext.CreateFence();

		// Pre-enumerate validBitmasks per slot when K > 0. The lower 60 bits of each
		// permutation field will then be a 1-based INDEX into _slotValidBitmasks[slot]
		// and the GPU's validBitmasksBuffer, which makes manager iteration O(1) and
		// eliminates the skip-loop / popcount safety net entirely.
		_slotValidBitmasks.resize(_gridCount);
		if (_maxRuleBits > 0)
		{
			uint32_t maxValidBitmasksCount = 1;
			for (uint32_t gridIndex = 0; gridIndex < _gridCount; gridIndex++)
			{
				_slotValidBitmasks[gridIndex] = PermutationMath::EnumerateValidBitmasks(
					gridConfigs[gridIndex].neighborCount, _maxRuleBits);
				if (_slotValidBitmasks[gridIndex].size() > maxValidBitmasksCount)
					maxValidBitmasksCount = static_cast<uint32_t>(_slotValidBitmasks[gridIndex].size());
				// Mutate the local copy of gridConfigs we hold so maxPermutation reflects
				// the index space, not the raw bitmask space.
				_gridConfigs[gridIndex].maxPermutation = _slotValidBitmasks[gridIndex].size();
			}
			_validBitmasksBufferElementCount = maxValidBitmasksCount;
		}

		// Ensure output directory exists. File names encode the search parameters.
		// JSONL format: one compact JSON object per line, appended as each result is
		// cached. A crash or force-kill preserves every completed permutation written
		// up to the moment of termination.
		std::filesystem::create_directories(outputFolderPath);
		_viableOutputPath = outputFolderPath + filePrefix + "_viable.jsonl";
		_possibleOutputPath = outputFolderPath + filePrefix + "_possible.jsonl";
		_unviableOutputPath = outputFolderPath + filePrefix + "_unviable.jsonl";

		std::filesystem::remove(_viableOutputPath);
		std::filesystem::remove(_possibleOutputPath);
		std::filesystem::remove(_unviableOutputPath);

		_viableFile.open(_viableOutputPath, std::ios::out | std::ios::app);
		_possibleFile.open(_possibleOutputPath, std::ios::out | std::ios::app);
		_unviableFile.open(_unviableOutputPath, std::ios::out | std::ios::app);

		if (!_viableFile.is_open() || !_possibleFile.is_open() || !_unviableFile.is_open())
			LOG_ERROR("SearchSystem: failed to open one or more JSONL output files in {}", outputFolderPath);

		_lastReadPermutations.resize(_gridCount);
		_lastReadGridInfos.resize(_gridCount);
		_lastSavedPermutations.resize(_gridCount, SimulationParameters{});
		_gridsNeedingCellReset.resize(_gridCount, false);

		// Initialize CPU-side mirrors of the per-slot iteration bounds. _slotMaxPermutations
		// always holds the full rule-space ceiling (work-stealing reads it off a busy slot
		// to know what range it could share). _slotSurvivalEndPermutations holds the slot's
		// *current* upper bound — equal to maxPermutation by default, but pre-narrowed when
		// --chunks-per-config N > 1 so partition i takes survival range
		// [1 + i × maxPermutation/N, (i+1) × maxPermutation/N].
		_slotSurvivalEndPermutations.resize(_gridCount);
		_slotMaxPermutations.resize(_gridCount);
		for (uint32_t gridIndex = 0; gridIndex < _gridCount; gridIndex++)
		{
			uint64_t partitionedEnd = _gridConfigs[gridIndex].maxPermutation;
			if (_chunkConfig.partitionCount > 1)
				partitionedEnd = static_cast<uint64_t>(_chunkConfig.partitionIndex + 1)
					* _gridConfigs[gridIndex].maxPermutation / _chunkConfig.partitionCount;
			_slotSurvivalEndPermutations[gridIndex] = partitionedEnd;
			_slotMaxPermutations[gridIndex] = _gridConfigs[gridIndex].maxPermutation;
		}

		BuildBuffers(_gridConfigs);
		BuildSimulationPipeline(simulationShaderModule);
		BuildManagerPipeline(managerShaderModule);
		RecordCommandBuffer();

		if (!_simulationPipeline || !_simulationPipeline->GetPipeline())
			LOG_ERROR("SearchSystem: simulation pipeline creation failed!");
		if (!_managerPipeline || !_managerPipeline->GetPipeline())
			LOG_ERROR("SearchSystem: manager pipeline creation failed!");

		LOG_INFO("SearchSystem created: {} parallel grids, {} cells per grid", _gridCount, _totalCellCount);
		if (_chunkConfig.partitionCount > 1)
		{
			uint64_t survivalStart = 1ull + static_cast<uint64_t>(_chunkConfig.partitionIndex)
				* _gridConfigs[0].maxPermutation / _chunkConfig.partitionCount;
			LOG_INFO("  partition {}/{}: survival range [{}, {}] (of {} per slot, grid 0)",
				_chunkConfig.partitionIndex + 1, _chunkConfig.partitionCount,
				survivalStart, _slotSurvivalEndPermutations[0], _gridConfigs[0].maxPermutation);
		}
	}

	SearchSystem::~SearchSystem()
	{
		_deviceContext.GetDevice().waitIdle();
		WriteToDisk();
		_deviceContext.GetDevice().destroyFence(_tickFence);
	}

	void SearchSystem::BuildBuffers(const std::vector<SearchGridConfig> &gridConfigs)
	{
		_gridBuffers.resize(_gridCount);
		// 8 cells packed per uint32_t as 4-bit nibbles. 0.5 bytes/cell — another 8×
		// VRAM reduction on top of the uvec4 → uint migration (32× vs the original).
		// Each compute thread owns one uint (= 8 cells) so there are no read-modify-write
		// races between threads on shared bytes.
		// Layout depends on shader: cube uses 3D-tiled (uintsPerRow × gridY × gridZ uints
		// with end-of-row padding when gridX isn't a multiple of 8) to match
		// searchSimulation.slang's `uintIdx = uintX + uintY * UINTS_PER_ROW + uintZ
		// * UINTS_PER_ROW * GRID_Y_SIZE`. ERD uses linear (cellIndex / 8) per
		// searchSimulationERD.slang. Mixing the two corrupts neighbor reads on grids
		// whose X dim isn't a multiple of 8 — caused rules to die spuriously fast at
		// e.g. grid=49 (uintsPerRow=7 gives 7-cell padding per row).
		uint32_t uintsPerRow = (_chunkConfig.gridDimensionX + 7u) / 8u;
		uint32_t packedUintCount = _simulationUses3DDispatch
			? uintsPerRow * _chunkConfig.gridDimensionY * _chunkConfig.gridDimensionZ
			: (_totalCellCount + 7u) / 8u;
		vk::DeviceSize cellBufferSize = sizeof(uint32_t) * packedUintCount;
		glm::uvec3 gridDimensions(_chunkConfig.gridDimensionX, _chunkConfig.gridDimensionY, _chunkConfig.gridDimensionZ);
		glm::uvec3 spawnDimensions(_chunkConfig.spawnAreaDimensionX, _chunkConfig.spawnAreaDimensionY, _chunkConfig.spawnAreaDimensionZ);

		// Generate initial cell state: alive cells (w = max possible = 15) in spawn area,
		// dead cells (w = 0) everywhere else. The manager shader uses the per-grid
		// maxCellState from the permutation, but we initialize with 15 (max) so cells
		// are always "alive" at the start regardless of which maxCellState the grid uses.
		// The simulation shader will treat cells with w == maxCellState as alive.
		// Since we want cells to start alive and maxCellState varies per grid, we set w
		// to a high value — but the simulation shader only considers w == maxCellState as
		// "fully alive." So we'll upload per-grid initial data with the correct maxCellState.
		// For simplicity, build ONE initial cell data array per unique maxCellState.

		for (uint32_t gridIndex = 0; gridIndex < _gridCount; gridIndex++)
		{
			auto &buffers = _gridBuffers[gridIndex];
			const auto &config = gridConfigs[gridIndex];
			std::string prefix = "Search Grid " + std::to_string(gridIndex);

			// Build starting permutation for this grid.
			// Birth axis always starts at 1 (minimum rule: "1 neighbor" birth).
			// Survival axis starts at 1 by default, or at the partition's lower bound
			// when --chunks-per-config N > 1 (partition i starts at 1 + i × maxPermutation/N).
			// Do NOT use EXISTENCE_MIN_PERMUTATION (1 << 60) — it corrupts even-numbered
			// maxCellState values by setting bit 60 in the upper 4-bit field.
			uint64_t partitionedSurvivalStart = 1ull;
			if (_chunkConfig.partitionCount > 1)
				partitionedSurvivalStart = 1ull + static_cast<uint64_t>(_chunkConfig.partitionIndex)
					* config.maxPermutation / _chunkConfig.partitionCount;

			// When partitionCount > maxPermutation, integer division produces some partitions
			// with start > end (degenerate range). Without this check those slots would still
			// run ONE simulation tick at the start value before the manager detects exhaustion,
			// double-counting that survival with the next partition's valid range. Mark such
			// slots as NULL_MASK upfront so the simulation never runs for them.
			bool partitionRangeDegenerate = partitionedSurvivalStart > _slotSurvivalEndPermutations[gridIndex];

			SimulationParameters startingPermutation{};
			if (partitionRangeDegenerate)
			{
				// All-zero permutation = NULL_MASK convention; manager treats this as "slot
				// exhausted" and the simulation shader's nibble loop won't be triggered.
				startingPermutation = SimulationParameters{};
			}
			else
			{
				startingPermutation.birthAndMaxCellStateRules =
					(Simulation::EncodeMaxCellState(config.maxCellState) << Simulation::BIT_SHIFT) | 1;
				startingPermutation.survivalAndNeighborhoodRules =
					(config.neighborhoodFlags << Simulation::BIT_SHIFT) | partitionedSurvivalStart;
				// Grid dimensions no longer encoded here — they're compile-time constants in the
				// search simulation shader (baked per-chunk). shapeAndGridConfiguration is retained
				// for layout parity but only holds the shape ID (bits 0-4) for now.
				startingPermutation.shapeAndGridConfiguration =
					static_cast<uint64_t>(_chunkConfig.shapeId) & 0x1Fu;
			}

			// Permutations buffer — written by manager, read by simulation
			buffers.gridPermutationsBuffer = std::make_shared<Buffer>(
				_deviceContext, sizeof(SimulationParameters),
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
				VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
				false, 0, prefix + " Permutations");
			buffers.gridPermutationsBuffer->Upload(&startingPermutation, sizeof(SimulationParameters));

			// Snapshot buffer — written by simulation (atomics), read by manager
			GridSnapshot emptySnapshot{0, 0, 0, 0};
			buffers.gridSnapshotBuffer = std::make_shared<Buffer>(
				_deviceContext, sizeof(GridSnapshot),
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
				VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
				false, 0, prefix + " Snapshot");
			buffers.gridSnapshotBuffer->Upload(&emptySnapshot, sizeof(GridSnapshot));

			// Build initial cell data for this grid's maxCellState. Packed format:
			// cell at linear index i → uint index i/8, nibble position i%8.
			uint32_t maxCellState = static_cast<uint32_t>(config.maxCellState);
			std::vector<uint32_t> initialCellData(packedUintCount, 0u);

			// Resolve sweep-axis values. spawnSizeIndex == 0 means "full configured spawn area";
			// index N > 0 scales linearly from 1 to spawnAreaDimension along each axis.
			auto resolveSpawnDimension = [&](uint32_t axisSize, uint32_t sizeIndex) -> uint32_t
			{
				if (sizeIndex == 0) return axisSize;
				uint32_t scaled = static_cast<uint32_t>(
					(static_cast<uint64_t>(axisSize) * (sizeIndex + 1) + 15) / 16);
				return std::max(1u, std::min(axisSize, scaled));
			};
			glm::uvec3 effectiveSpawnDimensions(
				resolveSpawnDimension(spawnDimensions.x, _chunkConfig.spawnSizeIndex),
				resolveSpawnDimension(spawnDimensions.y, _chunkConfig.spawnSizeIndex),
				resolveSpawnDimension(spawnDimensions.z, _chunkConfig.spawnSizeIndex));

			bool spawnIsRandom = (_chunkConfig.spawnRegionShapeId == 0);
			// Density index 0..15 maps to (1..16) / 16. Full density = 16/16 = 1.0.
			float density = (static_cast<float>(_chunkConfig.spawnDensityIndex) + 1.0f) / 16.0f;

			// Wang hash — same algorithm as cellInit.slang, so a given (runSeedIndex, cell)
			// pair produces identical results on CPU and GPU.
			auto wangHash = [](uint32_t seed) -> uint32_t
			{
				seed = (seed ^ 61u) ^ (seed >> 16);
				seed *= 9u;
				seed = seed ^ (seed >> 4);
				seed *= 0x27d4eb2du;
				seed = seed ^ (seed >> 15);
				return seed;
			};
			auto cellHash = [&](uint32_t x, uint32_t y, uint32_t z) -> uint32_t
			{
				uint32_t h = wangHash(x * 73856093u + _chunkConfig.runSeedIndex);
				h = wangHash(h ^ (y * 19349663u));
				h = wangHash(h ^ (z * 83492791u));
				return h;
			};
			uint32_t densityThreshold = (density >= 1.0f)
				? 0xFFFFFFFFu
				: static_cast<uint32_t>(density * 4294967296.0);

			for (uint32_t x = 0; x < gridDimensions.x; x++)
			{
				for (uint32_t y = 0; y < gridDimensions.y; y++)
				{
					for (uint32_t z = 0; z < gridDimensions.z; z++)
					{
						bool inSpawnArea =
							x >= (gridDimensions.x - effectiveSpawnDimensions.x) / 2 && x < (gridDimensions.x + effectiveSpawnDimensions.x) / 2 &&
							y >= (gridDimensions.y - effectiveSpawnDimensions.y) / 2 && y < (gridDimensions.y + effectiveSpawnDimensions.y) / 2 &&
							z >= (gridDimensions.z - effectiveSpawnDimensions.z) / 2 && z < (gridDimensions.z + effectiveSpawnDimensions.z) / 2;

						if (!inSpawnArea) continue;

						bool alive = spawnIsRandom
							? (density >= 1.0f || cellHash(x, y, z) < densityThreshold)
							: true;
						if (alive)
						{
							// Layout must match the active shader's cell-buffer addressing.
							// cube (searchSimulation.slang): 3D-tiled —
							//   uintIdx = uintX + uintY * uintsPerRow + uintZ * uintsPerRow * gridY,
							//   with row padding when gridX isn't a multiple of 8.
							// ERD (searchSimulationERD.slang): linear —
							//   cellIndex = x + y*gridX + z*gridX*gridY, then divide by 8.
							uint32_t uintIndex;
							uint32_t nibble;
							if (_simulationUses3DDispatch)
							{
								uint32_t uintX = x / 8u;
								nibble = x % 8u;
								uintIndex = uintX + y * uintsPerRow + z * uintsPerRow * gridDimensions.y;
							}
							else
							{
								uint32_t cellIndex = x + (y * gridDimensions.x) + (z * gridDimensions.x * gridDimensions.y);
								uintIndex = cellIndex / 8u;
								nibble    = cellIndex % 8u;
							}
							initialCellData[uintIndex] |= (maxCellState & 0xFu) << (nibble * 4u);
						}
					}
				}
			}

			// Cell buffers — ping-pong. cellsIn starts with the initial state.
			buffers.cellsInBuffer = std::make_shared<Buffer>(
				_deviceContext, static_cast<uint32_t>(cellBufferSize),
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
				VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
				false, 0, prefix + " Cells In");
			buffers.cellsInBuffer->Upload(initialCellData.data(), static_cast<uint32_t>(cellBufferSize));

			buffers.cellsOutBuffer = std::make_shared<Buffer>(
				_deviceContext, static_cast<uint32_t>(cellBufferSize),
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
				VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
				false, 0, prefix + " Cells Out");

			// Initial cells — copied back to cellsIn when the manager advances permutations
			buffers.initialCellsBuffer = std::make_shared<Buffer>(
				_deviceContext, static_cast<uint32_t>(cellBufferSize),
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
				VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
				false, 0, prefix + " Initial Cells");
			buffers.initialCellsBuffer->Upload(initialCellData.data(), static_cast<uint32_t>(cellBufferSize));

			// Max permutation
			buffers.gridMaxPermutationBuffer = std::make_shared<Buffer>(
				_deviceContext, sizeof(uint64_t),
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
				VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
				false, 0, prefix + " Max Permutation");
			buffers.gridMaxPermutationBuffer->Upload(&config.maxPermutation, sizeof(uint64_t));

			// Survival-axis end permutation. Equal to maxPermutation by default so each slot
			// iterates its full survival range 1..maxPermutation; pre-narrowed at construction
			// time when --chunks-per-config N > 1 (see _slotSurvivalEndPermutations init);
			// the CPU work-stealer can narrow it further later to carve out a sub-range
			// for an idle slot.
			buffers.gridSurvivalEndPermutationBuffer = std::make_shared<Buffer>(
				_deviceContext, sizeof(uint64_t),
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
				VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
				false, 0, prefix + " Survival End Permutation");
			buffers.gridSurvivalEndPermutationBuffer->Upload(&_slotSurvivalEndPermutations[gridIndex], sizeof(uint64_t));

			// CPU-readable copy buffers — initialize with starting data so the CPU
			// sees valid state before the GPU writes its first update.
			buffers.gridPermutationsCopyBuffer = std::make_shared<Buffer>(
				_deviceContext, sizeof(SimulationParameters),
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
				VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
				false, 0, prefix + " Permutations Copy");
			memcpy(buffers.gridPermutationsCopyBuffer->GetVmaAllocationInfo().pMappedData,
				&startingPermutation, sizeof(SimulationParameters));

			GridInfo zeroInfo{};
			buffers.gridInfoBuffer = std::make_shared<Buffer>(
				_deviceContext, sizeof(GridInfo),
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
				VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
				false, 0, prefix + " Grid Info");
			buffers.gridInfoBuffer->Upload(&zeroInfo, sizeof(GridInfo));

			buffers.gridInfoCopyBuffer = std::make_shared<Buffer>(
				_deviceContext, sizeof(GridInfo),
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
				VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
				false, 0, prefix + " Grid Info Copy");
			memcpy(buffers.gridInfoCopyBuffer->GetVmaAllocationInfo().pMappedData,
				&zeroInfo, sizeof(GridInfo));

			// ValidBitmasks SSBO. Sized to fit the max enumeration across this chunk's
			// neighborhoods so cross-neighborhood steals can re-upload without realloc.
			// When K == 0, allocated as a 1-element dummy that the shader never reads.
			vk::DeviceSize validBitmasksSize = sizeof(uint64_t) * _validBitmasksBufferElementCount;
			buffers.validBitmasksBuffer = std::make_shared<Buffer>(
				_deviceContext, validBitmasksSize,
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
				VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
				false, 0, prefix + " Valid Bitmasks");
			if (_maxRuleBits > 0 && !_slotValidBitmasks[gridIndex].empty())
			{
				buffers.validBitmasksBuffer->Upload(
					_slotValidBitmasks[gridIndex].data(),
					static_cast<uint32_t>(sizeof(uint64_t) * _slotValidBitmasks[gridIndex].size()));
			}

			_lastReadPermutations[gridIndex] = startingPermutation;
		}
	}

	void SearchSystem::BuildSimulationPipeline(vk::ShaderModule simulationShaderModule)
	{
		// Simulation shader has 5 bindings. Create one option per grid.
		_simulationDescriptor = std::make_shared<Descriptor>(_deviceContext);

		for (uint32_t gridIndex = 0; gridIndex < _gridCount; gridIndex++)
		{
			auto &buffers = _gridBuffers[gridIndex];
			_simulationDescriptor->BindBufferToDescriptorSet(0, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, buffers.gridPermutationsBuffer);
			_simulationDescriptor->BindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, buffers.gridSnapshotBuffer);
			_simulationDescriptor->BindBufferToDescriptorSet(2, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, buffers.cellsInBuffer);
			_simulationDescriptor->BindBufferToDescriptorSet(3, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, buffers.cellsOutBuffer);
			_simulationDescriptor->BindBufferToDescriptorSet(4, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, buffers.validBitmasksBuffer);
		}

		_simulationDescriptor->Build();

		auto simulationShader = std::make_shared<Shader>(_deviceContext, simulationShaderModule);
		_simulationPipeline = std::make_unique<ComputePipeline>(_deviceContext, simulationShader);
		_simulationPipeline->AddDescriptorSet(0, _simulationDescriptor);
		_simulationPipeline->Build();
	}

	void SearchSystem::BuildManagerPipeline(vk::ShaderModule managerShaderModule)
	{
		// Manager shader has 9 bindings (0-7 original + 8 survival-end-permutation).
		_managerDescriptor = std::make_shared<Descriptor>(_deviceContext);

		for (uint32_t gridIndex = 0; gridIndex < _gridCount; gridIndex++)
		{
			auto &buffers = _gridBuffers[gridIndex];
			_managerDescriptor->BindBufferToDescriptorSet(0, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, buffers.initialCellsBuffer);
			_managerDescriptor->BindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, buffers.cellsInBuffer);
			_managerDescriptor->BindBufferToDescriptorSet(2, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, buffers.gridSnapshotBuffer);
			_managerDescriptor->BindBufferToDescriptorSet(3, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, buffers.gridMaxPermutationBuffer);
			_managerDescriptor->BindBufferToDescriptorSet(4, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, buffers.gridPermutationsBuffer);
			_managerDescriptor->BindBufferToDescriptorSet(5, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, buffers.gridPermutationsCopyBuffer);
			_managerDescriptor->BindBufferToDescriptorSet(6, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, buffers.gridInfoBuffer);
			_managerDescriptor->BindBufferToDescriptorSet(7, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, buffers.gridInfoCopyBuffer);
			_managerDescriptor->BindBufferToDescriptorSet(8, vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute, buffers.gridSurvivalEndPermutationBuffer);
		}

		_managerDescriptor->Build();

		auto managerShader = std::make_shared<Shader>(_deviceContext, managerShaderModule);
		_managerPipeline = std::make_unique<ComputePipeline>(_deviceContext, managerShader);
		_managerPipeline->AddDescriptorSet(0, _managerDescriptor);
		_managerPipeline->Build();
	}

	void SearchSystem::RecordCommandBuffer()
	{
		_commandBuffer = _deviceContext.CreateCommandBuffer(_computeCommandPool);
		// Command buffer is re-recorded each tick (not SimultaneousUse)
		// because we need to fillBuffer the snapshots before each simulation dispatch.
	}

	bool SearchSystem::IsTickComplete()
	{
		auto waitResult = _deviceContext.GetDevice().waitForFences(_tickFence, vk::True, 0);
		if (waitResult == vk::Result::eTimeout)
			return false;
		if (waitResult != vk::Result::eSuccess)
		{
			LOG_ERROR("SearchSystem: fence error");
			return false;
		}
		return true;
	}

	void SearchSystem::SubmitNextTick(std::shared_ptr<QueryManager> queryManager, vk::QueryPool queryPool)
	{
		_deviceContext.GetDevice().resetFences(_tickFence);

		_commandBuffer.reset();

		vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		_commandBuffer.begin(beginInfo);

		// 8 cells packed per uint32_t. Layout matches BuildBuffers (3D-tiled for cube,
		// linear for ERD). See the comment block in BuildBuffers for details.
		uint32_t resetUintsPerRow = (_chunkConfig.gridDimensionX + 7u) / 8u;
		vk::DeviceSize cellBufferSizeForReset = _simulationUses3DDispatch
			? sizeof(uint32_t) * resetUintsPerRow * _chunkConfig.gridDimensionY * _chunkConfig.gridDimensionZ
			: sizeof(uint32_t) * ((_totalCellCount + 7u) / 8u);

		// Phase 0: reset cell data for grids that completed a permutation.
		// The manager shader advances the permutation and resets GridInfo but can't
		// efficiently copy 117K cells. We do it here via copyBuffer.
		bool anyResets = false;
		for (uint32_t gridIndex = 0; gridIndex < _gridCount; gridIndex++)
		{
			if (_gridsNeedingCellReset[gridIndex])
			{
				vk::BufferCopy resetCopyRegion{};
				resetCopyRegion.setSize(cellBufferSizeForReset);
				_commandBuffer.copyBuffer(
					_gridBuffers[gridIndex].initialCellsBuffer->GetBuffer(),
					_gridBuffers[gridIndex].cellsInBuffer->GetBuffer(),
					1, &resetCopyRegion);
				_gridsNeedingCellReset[gridIndex] = false;
				anyResets = true;
			}
		}

		if (anyResets)
		{
			// Barrier: cell reset transfers must complete before simulation reads
			vk::MemoryBarrier resetToComputeBarrier = vk::MemoryBarrier(
				vk::AccessFlagBits::eTransferWrite,                                  // srcAccessMask
				vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite   // dstAccessMask
			);
			_commandBuffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer,      // srcStageMask
				vk::PipelineStageFlagBits::eComputeShader, // dstStageMask
				{}, resetToComputeBarrier, {}, {});
		}

		// Cube simulation shader uses 3D dispatch with fixed (8, 8, 4) = 256 numthreads.
		// Each thread owns 1 packed uint = 8 cells along X. Workgroup covers 8 uints
		// × 8 Y × 4 Z = 64×8×4 cells = 2048 cells. Dispatch shape: per-axis ceil-divide
		// with X-axis sized in uints (= ceil(gridX/8)). _simulationWorkgroupSize is
		// kept on the SearchSystem for ERD's 1D-dispatch path; cube ignores it.
		uint32_t uintsPerRow = (_chunkConfig.gridDimensionX + 7u) / 8u;
		uint32_t cubeDispatchX = (uintsPerRow + 7u) / 8u;
		uint32_t cubeDispatchY = (_chunkConfig.gridDimensionY + 7u) / 8u;
		uint32_t cubeDispatchZ = (_chunkConfig.gridDimensionZ + 3u) / 4u;
		// ERD path (legacy 1D dispatch) — kept for the ERD shape's existing shader.
		uint32_t totalUints = (_totalCellCount + 7u) / 8u;
		uint32_t erdDispatchGroupCount = (totalUints + _simulationWorkgroupSize - 1u) / _simulationWorkgroupSize;

		// Phase 1: zero snapshots + dispatch simulation for all grids
		if (queryManager) queryManager->WriteTimestamp(_commandBuffer, queryPool, vk::PipelineStageFlagBits::eTopOfPipe, "searchSimulation_start");
		for (uint32_t gridIndex = 0; gridIndex < _gridCount; gridIndex++)
		{
			_commandBuffer.fillBuffer(
				_gridBuffers[gridIndex].gridSnapshotBuffer->GetBuffer(),
				0, sizeof(GridSnapshot), 0);

			vk::MemoryBarrier fillToComputeBarrier = vk::MemoryBarrier(
				vk::AccessFlagBits::eTransferWrite,                                  // srcAccessMask
				vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite   // dstAccessMask
			);
			_commandBuffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer,      // srcStageMask
				vk::PipelineStageFlagBits::eComputeShader, // dstStageMask
				{}, fillToComputeBarrier, {}, {});

			_simulationPipeline->Bind(_commandBuffer, vk::PipelineBindPoint::eCompute,
				0, Pipeline::DescriptorOption{gridIndex});
			if (_simulationUses3DDispatch)
				_commandBuffer.dispatch(cubeDispatchX, cubeDispatchY, cubeDispatchZ);
			else
				_commandBuffer.dispatch(erdDispatchGroupCount, 1, 1);
		}

		if (queryManager) queryManager->WriteTimestamp(_commandBuffer, queryPool, vk::PipelineStageFlagBits::eBottomOfPipe, "searchSimulation_end");

		// Barrier: all simulation writes must complete before manager reads
		vk::MemoryBarrier simulationToManagerBarrier = vk::MemoryBarrier(
			vk::AccessFlagBits::eShaderWrite, // srcAccessMask
			vk::AccessFlagBits::eShaderRead   // dstAccessMask
		);
		_commandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eComputeShader, // srcStageMask
			vk::PipelineStageFlagBits::eComputeShader, // dstStageMask
			{}, simulationToManagerBarrier, {}, {});

		// Phase 2: dispatch manager for all grids
		if (queryManager) queryManager->WriteTimestamp(_commandBuffer, queryPool, vk::PipelineStageFlagBits::eTopOfPipe, "searchManager_start");
		for (uint32_t gridIndex = 0; gridIndex < _gridCount; gridIndex++)
		{
			_managerPipeline->Bind(_commandBuffer, vk::PipelineBindPoint::eCompute,
				0, Pipeline::DescriptorOption{gridIndex});
			_commandBuffer.dispatch(1, 1, 1);
		}

		if (queryManager) queryManager->WriteTimestamp(_commandBuffer, queryPool, vk::PipelineStageFlagBits::eBottomOfPipe, "searchManager_end");

		// Barrier: manager writes must complete before the cell buffer copy
		vk::MemoryBarrier managerToCopyBarrier = vk::MemoryBarrier(
			vk::AccessFlagBits::eShaderWrite,    // srcAccessMask
			vk::AccessFlagBits::eTransferRead     // dstAccessMask
		);
		_commandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eComputeShader, // srcStageMask
			vk::PipelineStageFlagBits::eTransfer,      // dstStageMask
			{}, managerToCopyBarrier, {}, {});

		// Phase 3: copy cellsOut → cellsIn for each grid so the next tick
		// reads the updated cell state (without this, the simulation always
		// re-reads the initial data — no actual cell state progression).
		if (queryManager) queryManager->WriteTimestamp(_commandBuffer, queryPool, vk::PipelineStageFlagBits::eTopOfPipe, "searchCopy_start");
		// Match BuildBuffers layout (3D-tiled for cube, linear for ERD).
		uint32_t copyUintsPerRow = (_chunkConfig.gridDimensionX + 7u) / 8u;
		vk::DeviceSize cellBufferSize = _simulationUses3DDispatch
			? sizeof(uint32_t) * copyUintsPerRow * _chunkConfig.gridDimensionY * _chunkConfig.gridDimensionZ
			: sizeof(uint32_t) * ((_totalCellCount + 7u) / 8u);
		for (uint32_t gridIndex = 0; gridIndex < _gridCount; gridIndex++)
		{
			vk::BufferCopy copyRegion{};
			copyRegion.setSize(cellBufferSize);
			_commandBuffer.copyBuffer(
				_gridBuffers[gridIndex].cellsOutBuffer->GetBuffer(),
				_gridBuffers[gridIndex].cellsInBuffer->GetBuffer(),
				1, &copyRegion);
		}

		if (queryManager) queryManager->WriteTimestamp(_commandBuffer, queryPool, vk::PipelineStageFlagBits::eBottomOfPipe, "searchCopy_end");

		_commandBuffer.end();

		vk::SubmitInfo submitInfo(
			0, nullptr, nullptr,     // waitSemaphoreCount, pWaitSemaphores, pWaitDstStageMask
			1, &_commandBuffer,      // commandBufferCount, pCommandBuffers
			0, nullptr               // signalSemaphoreCount, pSignalSemaphores
		);

		_deviceContext.GetComputeQueue().submit(submitInfo, _tickFence);
	}

	void SearchSystem::PollCopyBuffers()
	{
		// Read directly from the host-mapped copy buffers (no staging buffer needed).
		// These buffers were created with VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
		// and VMA_ALLOCATION_CREATE_MAPPED_BIT, so pMappedData is always valid.
		for (uint32_t gridIndex = 0; gridIndex < _gridCount; gridIndex++)
		{
			const void* permutationData = _gridBuffers[gridIndex].gridPermutationsCopyBuffer->GetVmaAllocationInfo().pMappedData;
			memcpy(&_lastReadPermutations[gridIndex], permutationData, sizeof(SimulationParameters));

			const void* gridInfoData = _gridBuffers[gridIndex].gridInfoCopyBuffer->GetVmaAllocationInfo().pMappedData;
			memcpy(&_lastReadGridInfos[gridIndex], gridInfoData, sizeof(GridInfo));
		}
	}

	void SearchSystem::CacheCompletedResults()
	{
		for (uint32_t gridIndex = 0; gridIndex < _gridCount; gridIndex++)
		{
			const auto &gridInfo = _lastReadGridInfos[gridIndex];
			const auto &permutation = _lastReadPermutations[gridIndex];

			if (gridInfo.status == 0)
				continue;

			// Skip exhausted grids (NULL permutation = search space fully explored)
			if (permutation.birthAndMaxCellStateRules == 0 && permutation.survivalAndNeighborhoodRules == 0)
				continue;

			// Skip if we already cached this exact permutation for this grid
			if (permutation.birthAndMaxCellStateRules == _lastSavedPermutations[gridIndex].birthAndMaxCellStateRules &&
				permutation.survivalAndNeighborhoodRules == _lastSavedPermutations[gridIndex].survivalAndNeighborhoodRules)
				continue;

			_lastSavedPermutations[gridIndex] = permutation;
			_gridsNeedingCellReset[gridIndex] = true;

			// Translate index → bitmask before recording. When K>0 the lower 60 bits
			// of `permutation` are 1-based indices into _slotValidBitmasks[gridIndex];
			// JSONL consumers want the actual rule bitmasks. The dedup mirror above
			// keeps the raw (index) form so the next-tick comparison stays cheap.
			SimulationParameters outputPermutation = permutation;
			if (_maxRuleBits > 0)
			{
				const auto& slotBitmasks = _slotValidBitmasks[gridIndex];
				uint64_t birthIndex    = (permutation.birthAndMaxCellStateRules    & Simulation::EXISTENCE_PERMUTATION_BIT_MASK) - 1;
				uint64_t survivalIndex = (permutation.survivalAndNeighborhoodRules & Simulation::EXISTENCE_PERMUTATION_BIT_MASK) - 1;
				if (birthIndex < slotBitmasks.size() && survivalIndex < slotBitmasks.size())
				{
					uint64_t maxCellStateBits     = permutation.birthAndMaxCellStateRules    & ~Simulation::EXISTENCE_PERMUTATION_BIT_MASK;
					uint64_t neighborhoodFlagBits = permutation.survivalAndNeighborhoodRules & ~Simulation::EXISTENCE_PERMUTATION_BIT_MASK;
					outputPermutation.birthAndMaxCellStateRules    = maxCellStateBits     | slotBitmasks[birthIndex];
					outputPermutation.survivalAndNeighborhoodRules = neighborhoodFlagBits | slotBitmasks[survivalIndex];
				}
			}

			SearchResult result{
				outputPermutation,
				gridInfo.numberOfSimulationTicksSurvived,
				gridInfo.totalNumberOfCellsThatWereBorn,
				gridInfo.totalNumberOfCellsThatDied,
				gridInfo.maxAliveCount,
				gridInfo.finalAliveCount
			};
			uint64_t maxCellState = Simulation::DecodeMaxCellState(permutation.birthAndMaxCellStateRules >> Simulation::BIT_SHIFT);

			if (gridInfo.status == 2)
			{
				_viableResults.push_back(result);
				AppendResultLine(_viableFile, result);
			}
			else if (gridInfo.numberOfSimulationTicksSurvived > maxCellState + 1)
			{
				_possibleResults.push_back(result);
				AppendResultLine(_possibleFile, result);
			}
			else
			{
				_unviableResults.push_back(result);
				AppendResultLine(_unviableFile, result);
			}
		}
	}

	bool SearchSystem::IsRunning()
	{
		for (uint32_t gridIndex = 0; gridIndex < _gridCount; gridIndex++)
		{
			const auto &perm = _lastReadPermutations[gridIndex];
			if (perm.survivalAndNeighborhoodRules != 0 || perm.birthAndMaxCellStateRules != 0)
				return true;
		}
		return false;
	}

	void SearchSystem::FlushProgress()
	{
		PollCopyBuffers();
		CacheCompletedResults();
		AttemptWorkStealingForIdleSlots();
	}

	bool SearchSystem::HasIdleSlots() const
	{
		for (uint32_t i = 0; i < _gridCount; i++)
		{
			const auto& p = _lastReadPermutations[i];
			if (p.birthAndMaxCellStateRules == 0 && p.survivalAndNeighborhoodRules == 0)
				return true;
		}
		return false;
	}

	bool SearchSystem::HasBusySlots() const
	{
		for (uint32_t i = 0; i < _gridCount; i++)
		{
			const auto& p = _lastReadPermutations[i];
			if (p.birthAndMaxCellStateRules != 0 || p.survivalAndNeighborhoodRules != 0)
				return true;
		}
		return false;
	}

	uint32_t SearchSystem::TryStealFromOtherChunk(SearchSystem& donor)
	{
		uint32_t stealsBefore = static_cast<uint32_t>(_workStealCount);
		TryStealIntoIdleSlotsFromSource(donor);
		return static_cast<uint32_t>(_workStealCount) - stealsBefore;
	}

	void SearchSystem::AttemptWorkStealingForIdleSlots()
	{
		TryStealIntoIdleSlotsFromSource(*this);
	}

	uint32_t SearchSystem::TryStealIntoIdleSlotsFromSource(SearchSystem& slotSource)
	{
		bool anySteal = false;
		uint32_t stealsThisCall = 0;
		uint64_t initialStealCount = _workStealCount;

		// Walk every slot in THIS SearchSystem; for each idle one (permutation currently
		// NULL), try to steal half of the remaining survival range from the busiest
		// compatible active slot in `slotSource`. "Compatible" = same maxCellState, so
		// the idle slot's initialCells buffer can be reused without regeneration.
		// Different-maxCS stealing is possible in principle but requires rewriting the
		// packed initialCells; leaving that for a follow-up.
		for (uint32_t idleSlotIndex = 0; idleSlotIndex < _gridCount; idleSlotIndex++)
		{
			const auto& idlePermutation = _lastReadPermutations[idleSlotIndex];
			if (idlePermutation.birthAndMaxCellStateRules != 0
				|| idlePermutation.survivalAndNeighborhoodRules != 0)
				continue;

			// Idle slot compatibility key: ORIGINAL maxCellState (the initialCells buffer
			// was built for that value). A busy slot running a stolen rule may have a
			// different current maxCellState than its own original — we compare by the
			// CURRENT maxCellState encoded in the busy slot's permutation.
			uint64_t idleOriginalMaxCellState = _gridConfigs[idleSlotIndex].maxCellState;

			// Find the active slot (in the source SearchSystem — may be *this or another)
			// with the largest remaining survival range whose CURRENT maxCellState matches
			// the idle slot's original maxCellState.
			int32_t bestBusySlotIndex = -1;
			uint64_t bestBusyRemainingSurvivals = 0;
			uint64_t bestBusyCurrentSurvival = 0;
			uint64_t bestBusyEnd = 0;
			for (uint32_t busySlotIndex = 0; busySlotIndex < slotSource._gridCount; busySlotIndex++)
			{
				// When sourcing from the same SearchSystem, don't steal from self.
				if (&slotSource == this && busySlotIndex == idleSlotIndex) continue;

				const auto& busyPermutation = slotSource._lastReadPermutations[busySlotIndex];
				if (busyPermutation.birthAndMaxCellStateRules == 0
					&& busyPermutation.survivalAndNeighborhoodRules == 0)
					continue; // also idle

				uint64_t busyCurrentMaxCellState = Simulation::DecodeMaxCellState(
					busyPermutation.birthAndMaxCellStateRules >> Simulation::BIT_SHIFT);
				if (busyCurrentMaxCellState != idleOriginalMaxCellState)
					continue; // incompatible maxCS (would need cell rebuild)

				uint64_t currentSurvival = busyPermutation.survivalAndNeighborhoodRules & Simulation::EXISTENCE_PERMUTATION_BIT_MASK;
				uint64_t survivalEnd = slotSource._slotSurvivalEndPermutations[busySlotIndex];
				if (survivalEnd <= currentSurvival) continue;

				uint64_t remaining = survivalEnd - currentSurvival;
				if (remaining > bestBusyRemainingSurvivals)
				{
					bestBusyRemainingSurvivals = remaining;
					bestBusySlotIndex = static_cast<int32_t>(busySlotIndex);
					bestBusyCurrentSurvival = currentSurvival;
					bestBusyEnd = survivalEnd;
				}
			}

			if (bestBusySlotIndex < 0) continue;

			// Pull the busy slot's CURRENT running config from its live permutation +
			// tracked max-permutation (note: these live on slotSource, not `this`, when
			// stealing cross-chunk). Using the original _gridConfigs here would be
			// stale once the busy slot has itself received stolen work.
			const auto& bestBusyPermutation = slotSource._lastReadPermutations[static_cast<uint32_t>(bestBusySlotIndex)];
			uint64_t busyCurrentMaxCellStateEncoded = bestBusyPermutation.birthAndMaxCellStateRules >> Simulation::BIT_SHIFT;
			uint64_t busyCurrentNeighborhoodFlags = bestBusyPermutation.survivalAndNeighborhoodRules >> Simulation::BIT_SHIFT;
			uint64_t busyCurrentMaxPermutation = slotSource._slotMaxPermutations[static_cast<uint32_t>(bestBusySlotIndex)];

			// Pick a split point. The donor's survival range is contiguous in iteration
			// space — every value in [currentSurvival+1, busyEnd] is a valid rule
			// regardless of K (K==0 → raw bitmask space, K>0 → 1-based index space into
			// pre-enumerated validBitmasks). A plain midpoint always produces two
			// non-empty halves; the size threshold avoids tail-end churn.
			static constexpr uint64_t MIN_DONOR_REMAINING_FOR_STEAL = 32;
			if (bestBusyRemainingSurvivals < MIN_DONOR_REMAINING_FOR_STEAL) continue;
			uint64_t midpoint = bestBusyCurrentSurvival + bestBusyRemainingSurvivals / 2;
			uint64_t busyNewEnd = midpoint - 1;

			// Update busy slot (in slotSource): narrow its survival range upper bound.
			slotSource._slotSurvivalEndPermutations[static_cast<uint32_t>(bestBusySlotIndex)] = busyNewEnd;
			slotSource._gridBuffers[static_cast<uint32_t>(bestBusySlotIndex)]
				.gridSurvivalEndPermutationBuffer->Upload(&busyNewEnd, sizeof(uint64_t));

			// Set up idle slot to pick up the upper half of the original survival range,
			// running the busy slot's CURRENT config (not slot gridConfigs's original).
			SimulationParameters stolenPermutation{};
			stolenPermutation.birthAndMaxCellStateRules =
				(busyCurrentMaxCellStateEncoded << Simulation::BIT_SHIFT) | 1;
			stolenPermutation.survivalAndNeighborhoodRules =
				(busyCurrentNeighborhoodFlags << Simulation::BIT_SHIFT) | midpoint;
			stolenPermutation.shapeAndGridConfiguration =
				static_cast<uint64_t>(_chunkConfig.shapeId) & 0x1Fu;

			auto& idleBuffers = _gridBuffers[idleSlotIndex];
			idleBuffers.gridPermutationsBuffer->Upload(&stolenPermutation, sizeof(SimulationParameters));
			idleBuffers.gridMaxPermutationBuffer->Upload(&busyCurrentMaxPermutation, sizeof(uint64_t));
			idleBuffers.gridSurvivalEndPermutationBuffer->Upload(&bestBusyEnd, sizeof(uint64_t));
			_slotSurvivalEndPermutations[idleSlotIndex] = bestBusyEnd;
			_slotMaxPermutations[idleSlotIndex] = busyCurrentMaxPermutation;

			// When K>0, the lower 60 bits are indices into the per-slot validBitmasks
			// SSBO. The idle slot is now running the donor's neighborhood, so it must
			// adopt the donor's validBitmasks (which the donor's GPU buffer already
			// holds). Copy the CPU mirror and re-upload the idle slot's GPU buffer.
			if (_maxRuleBits > 0)
			{
				_slotValidBitmasks[idleSlotIndex] =
					slotSource._slotValidBitmasks[static_cast<uint32_t>(bestBusySlotIndex)];
				if (!_slotValidBitmasks[idleSlotIndex].empty())
				{
					idleBuffers.validBitmasksBuffer->Upload(
						_slotValidBitmasks[idleSlotIndex].data(),
						static_cast<uint32_t>(sizeof(uint64_t) * _slotValidBitmasks[idleSlotIndex].size()));
				}
			}

			GridInfo freshGridInfo{0, 0, 0, 0};
			idleBuffers.gridInfoBuffer->Upload(&freshGridInfo, sizeof(GridInfo));

			// Flag that cellsIn should be reset from initialCells on the next tick —
			// the GPU handles that in Phase 0 of the recorded command buffer.
			_gridsNeedingCellReset[idleSlotIndex] = true;

			// Clear the "already saved" tracking for this slot so the newly stolen
			// permutation gets recorded when it eventually completes.
			_lastSavedPermutations[idleSlotIndex] = SimulationParameters{};
			_lastReadPermutations[idleSlotIndex] = stolenPermutation;

			_workStealCount++;
			_workStealTotalSurvivalRangeMoved += (bestBusyEnd - midpoint + 1);
			anySteal = true;

			LOG_DEBUG("Work steal: slot {} -> slot {} (maxCS={}, neighborhood=0x{:X}, maxPerm={}), split survival at {}, idle takes [{}, {}], busy keeps [{}, {}]",
				bestBusySlotIndex, idleSlotIndex,
				static_cast<uint32_t>(Simulation::DecodeMaxCellState(busyCurrentMaxCellStateEncoded)),
				static_cast<uint32_t>(busyCurrentNeighborhoodFlags),
				busyCurrentMaxPermutation,
				midpoint, midpoint, bestBusyEnd,
				bestBusyCurrentSurvival + 1, busyNewEnd);
		}

		// If we uploaded new buffer data via Buffer::Upload (graphics queue, single-time
		// command), force a device-wide waitIdle before the next compute submission.
		// Without this, the compute queue could read buffers that the graphics queue's
		// transfer has not yet released ownership of — undefined behavior under EXCLUSIVE
		// sharing mode, which manifests as crashes a few ticks later.
		if (anySteal)
		{
			_deviceContext.GetDevice().waitIdle();
			// Also wait on the donor's device if cross-chunk steals touched a different one.
			if (&slotSource._deviceContext != &_deviceContext)
				slotSource._deviceContext.GetDevice().waitIdle();
		}
		stealsThisCall = static_cast<uint32_t>(_workStealCount - initialStealCount);
		return stealsThisCall;
	}

	SearchProgress SearchSystem::GetProgress() const
	{
		// Return progress from the first active grid for the UI display
		SimulationParameters samplePermutation{};
		GridInfo sampleGridInfo{};
		uint32_t activeCount = 0;

		for (uint32_t gridIndex = 0; gridIndex < _gridCount; gridIndex++)
		{
			const auto &perm = _lastReadPermutations[gridIndex];
			if (perm.survivalAndNeighborhoodRules != 0 || perm.birthAndMaxCellStateRules != 0)
			{
				if (activeCount == 0)
				{
					samplePermutation = perm;
					sampleGridInfo = _lastReadGridInfos[gridIndex];
				}
				activeCount++;
			}
		}

		return SearchProgress{
			samplePermutation,
			sampleGridInfo,
			static_cast<uint32_t>(_viableResults.size()),
			static_cast<uint32_t>(_possibleResults.size()),
			static_cast<uint32_t>(_unviableResults.size()),
			_gridCount,
			activeCount};
	}

	void SearchSystem::AppendResultLine(std::ofstream &outputFile, const SearchResult &result)
	{
		if (!outputFile.is_open())
			return;

		nlohmann::json entry;
		entry["birthAndMaxCellStateRules"] = result.permutation.birthAndMaxCellStateRules;
		entry["survivalAndNeighborhoodRules"] = result.permutation.survivalAndNeighborhoodRules;
		entry["shapeAndGridConfiguration"] = result.permutation.shapeAndGridConfiguration;
		entry["ticksSurvived"] = result.ticksSurvived;
		entry["totalBirths"] = result.totalBirths;
		entry["totalDeaths"] = result.totalDeaths;
		entry["maxAliveCount"] = result.maxAliveCount;
		entry["finalAliveCount"] = result.finalAliveCount;

		// Compact single-line form (dump() with no indent). Small enough (<500B) to be an
		// atomic append on both NTFS and ext4 — readers never see a partial line.
		outputFile << entry.dump() << '\n';
		outputFile.flush();
	}

	void SearchSystem::WriteToDisk()
	{
		// Results stream to JSONL files continuously as they are cached; this is now
		// just a best-effort flush plus a summary log entry for visibility at the
		// end of a search session.
		if (_viableFile.is_open())   _viableFile.flush();
		if (_possibleFile.is_open()) _possibleFile.flush();
		if (_unviableFile.is_open()) _unviableFile.flush();

		LOG_INFO("Search results flushed — viable: {}, possible: {}, unviable: {} (streamed to {}, {}, {})",
			_viableResults.size(), _possibleResults.size(), _unviableResults.size(),
			_viableOutputPath, _possibleOutputPath, _unviableOutputPath);
	}
}
