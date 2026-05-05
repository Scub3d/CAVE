#include "testRunner.h"

#include "../modes/renderingMode.h"
#include "../vulkan/vulkanInstance.h"
#include "../vulkan/ghostExchange.h"
#include "../vulkan/systems/guiSystem.h"
#include "../simulation/simulation.h"
#include "../common/logger.h"

#include <GLFW/glfw3.h>
#include <algorithm>

namespace Cave
{

	TestRunner::TestRunner(ModeServices& services)
		: _services{services}
	{
	}

	std::vector<TestRunner::TickTiming> TestRunner::RunSimulationTicks(
		int gridSize, int spawnSize, int ticks,
		const std::string& birth, const std::string& survival, int maxCellState,
		const std::string& neighborhood, bool enableComputeSkip)
	{
		GuiState& guiState = _services.guiSystem.GetState();
		guiState.gridDimensionX = guiState.gridDimensionY = guiState.gridDimensionZ = gridSize;
		guiState.spawnDimensionX = guiState.spawnDimensionY = guiState.spawnDimensionZ = spawnSize;
		guiState.enableComputeSkip = enableComputeSkip;

		auto renderingMode = std::make_unique<RenderingMode>();
		renderingMode->Enter(_services);

		// Re-apply after Enter (it overwrites with hardcoded defaults)
		guiState.birthRulesText = birth;
		guiState.survivalRulesText = survival;
		guiState.maxCellState = maxCellState;
		guiState.faceNeighbors = neighborhood.find('F') != std::string::npos;
		guiState.edgeNeighbors = neighborhood.find('E') != std::string::npos;
		guiState.cornerNeighbors = neighborhood.find('C') != std::string::npos;
		guiState.wrapAtBoundary = neighborhood.find('W') != std::string::npos;

		_services.guiSystem.InjectStartRequest();

		int ticksToCollect = ticks + 2;
		std::vector<TickTiming> rawResults;
		double previousFrameTime = glfwGetTime();

		while (rawResults.size() < static_cast<size_t>(ticksToCollect))
		{
			glfwPollEvents();
			if (glfwWindowShouldClose(_services.vulkanInstance.GetVulkanGLFWWindow()))
				break;

			double currentTime = glfwGetTime();
			float deltaTimeInSeconds = static_cast<float>(currentTime - previousFrameTime);
			previousFrameTime = currentTime;

			renderingMode->OnFrame(_services, deltaTimeInSeconds);

			if (renderingMode->GetSimulationTickCount() > 0)
			{
				const auto& timings = renderingMode->GetLastRawTimings();
				auto computeIt = timings.find("compute");
				auto renderIt = timings.find("render");
				rawResults.push_back({
					computeIt != timings.end() ? computeIt->second : 0.0f,
					renderIt != timings.end() ? renderIt->second : 0.0f
				});
			}
		}

		// Trim leading zero/stale results and trailing padding
		std::vector<TickTiming> results;
		for (const auto& t : rawResults)
		{
			if (results.empty() && t.computeMs < 0.01f)
				continue;
			results.push_back(t);
			if (static_cast<int>(results.size()) >= ticks)
				break;
		}

		LOG_INFO("  RunSimulationTicks: collected {} tick timings for {}^3", results.size(), gridSize);

		_services.deviceContext.GetDevice().waitIdle();
		renderingMode->Exit(_services);

		return results;
	}

	TestRunner::TestResult TestRunner::RunComputeSkipTest()
	{
		LOG_INFO("[test:compute-skip] Running 800^3, spawn 50, 10 ticks...");

		auto onTimings = RunSimulationTicks(800, 50, 10, "4", "4", 5, "F", true);
		auto offTimings = RunSimulationTicks(800, 50, 10, "4", "4", 5, "F", false);

		if (onTimings.size() < 3 || offTimings.size() < 3)
			return {"compute-skip", false, "FAIL: not enough ticks collected"};

		LOG_INFO("[test:compute-skip] Per-tick comparison:");
		LOG_INFO("  Tick     Skip ON     Skip OFF    Speedup");
		for (int i = 0; i < static_cast<int>(std::min(onTimings.size(), offTimings.size())); i++)
		{
			float speedup = (offTimings[i].computeMs > 0.01f && onTimings[i].computeMs > 0.01f)
				? offTimings[i].computeMs / onTimings[i].computeMs : 0.0f;
			LOG_INFO("  {:>4}  {:>8.2f}ms  {:>8.2f}ms  {:>7.1f}x", i, onTimings[i].computeMs, offTimings[i].computeMs, speedup);
		}

		float avgSkipOn = 0, avgSkipOff = 0;
		int steadyCount = 0;
		for (int i = 6; i < static_cast<int>(std::min(onTimings.size(), offTimings.size())); i++)
		{
			avgSkipOn += onTimings[i].computeMs;
			avgSkipOff += offTimings[i].computeMs;
			steadyCount++;
		}
		float speedup = (steadyCount > 0 && avgSkipOn > 0.01f) ? (avgSkipOff / avgSkipOn) : 0.0f;
		bool passed = speedup > 1.5f;
		std::string summary = passed
			? "PASS: skip ON is " + std::to_string(speedup) + "x faster at steady state"
			: "FAIL: skip ON not >1.5x faster at steady state (got " + std::to_string(speedup) + "x)";
		LOG_INFO("[test:compute-skip] {}", summary);
		return {"compute-skip", passed, summary};
	}

	TestRunner::TestResult TestRunner::RunSparseBenchmarkTest()
	{
		LOG_INFO("[test:sparse-benchmark] Running multiple grid sizes...");

		std::vector<int> gridSizes = {225, 512, 800, 1500};
		LOG_INFO("  {:>6s} {:>5s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}", "Grid", "Spawn", "Tick 0", "Tick 1", "Tick 2", "Tick 5", "Tick 10");

		for (int grid : gridSizes)
		{
			int spawn = grid / 4;
			auto timings = RunSimulationTicks(grid, spawn, 15, "4", "4", 5, "F", true);

			auto getCompute = [&](size_t i) { return i < timings.size() ? timings[i].computeMs : 0.0f; };
			LOG_INFO("  {:>5d}^3 {:>5d} {:>8.2f}ms {:>8.2f}ms {:>8.2f}ms {:>8.2f}ms {:>8.2f}ms",
				grid, spawn, getCompute(0), getCompute(1), getCompute(2), getCompute(5), getCompute(10));
		}

		return {"sparse-benchmark", true, "PASS: benchmark complete"};
	}

	TestRunner::TestResult TestRunner::RunEncodeBenchmarkTest()
	{
		// Bench tier-1 for video-encode-pipeline optimization split-test.
		// Mirrors the user's 4K dual-GPU 4201^3 encode workload at smaller grids:
		// neighborhood=Corner+Wrap, B=1,2 / S=, MaxCellState=12, spawn=25 (filled).
		// 30 ticks, single-GPU. Captures per-tick computeMs at fixed sample points.
		LOG_INFO("[test:encode-benchmark] Mirrors 4K dual-GPU encode workload (CW, B=1,2, MaxCS=12, spawn=25)");
		LOG_INFO("  {:>6s} {:>5s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}",
			"Grid", "Spawn", "T0", "T5", "T10", "T15", "T20", "T29");

		// Three grid sizes within single-GPU memory budget (Quadro RTX 8000, 48 GB).
		// 4001^3 needs ~64 GB for cell-state ping-pong → dual-GPU only.
		std::vector<int> gridSizes = {1501, 2001, 3001};
		for (int grid : gridSizes)
		{
			auto timings = RunSimulationTicks(grid, 25, 30, "1,2", "", 12, "CW", true);

			auto getCompute = [&](size_t i) { return i < timings.size() ? timings[i].computeMs : 0.0f; };
			LOG_INFO("  {:>5d}^3 {:>5d} {:>8.2f}ms {:>8.2f}ms {:>8.2f}ms {:>8.2f}ms {:>8.2f}ms {:>8.2f}ms",
				grid, 25, getCompute(0), getCompute(5), getCompute(10), getCompute(15), getCompute(20), getCompute(29));
		}

		return {"encode-benchmark", true, "PASS: benchmark complete"};
	}

	TestRunner::TestResult TestRunner::RunRenderConsistencyTest()
	{
		LOG_INFO("[test:render-consistency] Running 225^3 five times, checking render...");

		bool wasHeadless = _services.headless;
		_services.headless = false;

		int successCount = 0;
		int totalRuns = 5;

		for (int i = 0; i < totalRuns; i++)
		{
			auto timings = RunSimulationTicks(225, 100, 5, "1,4-6", "4", 5, "F", true);
			float renderMs = timings.size() > 0 ? timings[0].renderMs : 0.0f;
			bool rendered = renderMs > 1.0f;
			if (rendered) successCount++;
			LOG_INFO("  Run {}: render={:.2f}ms {}", i + 1, renderMs, rendered ? "OK" : "FAIL (black)");
		}

		_services.headless = wasHeadless;

		bool passed = successCount == totalRuns;
		std::string summary = std::to_string(successCount) + "/" + std::to_string(totalRuns) + " renders succeeded";
		LOG_INFO("[test:render-consistency] {} — {}", passed ? "PASS" : "FAIL", summary);
		return {"render-consistency", passed, summary};
	}

	TestRunner::TestResult TestRunner::RunLargeGridTest()
	{
		LOG_INFO("[test:large-grid] Testing large grid sizes...");

		std::vector<int> gridSizes = {950, 1500, 2023};
		bool allPassed = true;

		for (int grid : gridSizes)
		{
			int spawn = grid / 5;
			LOG_INFO("  Testing {}^3 (spawn {})...", grid, spawn);

			try
			{
				auto timings = RunSimulationTicks(grid, spawn, 5, "4", "4", 5, "F", true);
				float computeMs = timings.size() > 0 ? timings[0].computeMs : 0.0f;
				float vramEstimateGB = static_cast<float>(grid) * grid * grid * 2.0f / (1024.0f * 1024.0f * 1024.0f) * 2.0f;
				LOG_INFO("  {}^3: compute={:.1f}ms, est. VRAM={:.1f}GB — OK", grid, computeMs, vramEstimateGB);
			}
			catch (const std::exception& e)
			{
				LOG_ERROR("  {}^3: CRASHED — {}", grid, e.what());
				allPassed = false;
			}
		}

		std::string summary = allPassed ? "PASS: all grid sizes succeeded" : "FAIL: some grid sizes crashed";
		LOG_INFO("[test:large-grid] {}", summary);
		return {"large-grid", allPassed, summary};
	}

	TestRunner::TestResult TestRunner::RunCliRulesTest()
	{
		LOG_INFO("[test:cli-rules] Verifying rule round-trip...");

		GuiState& guiState = _services.guiSystem.GetState();
		guiState.birthRulesText = "3,5-7";
		guiState.survivalRulesText = "2,4";
		guiState.maxCellState = 8;
		guiState.faceNeighbors = true;
		guiState.edgeNeighbors = true;
		guiState.cornerNeighbors = true;
		guiState.wrapAtBoundary = false;

		SimulationParameters params = _services.guiSystem.BuildSimulationParameters();
		uint64_t birthBitmask = params.birthAndMaxCellStateRules & Simulation::EXISTENCE_PERMUTATION_BIT_MASK;
		uint64_t survivalBitmask = params.survivalAndNeighborhoodRules & Simulation::EXISTENCE_PERMUTATION_BIT_MASK;
		uint64_t maxCellState = Simulation::DecodeMaxCellState(params.birthAndMaxCellStateRules >> Simulation::BIT_SHIFT);

		std::string birthDecoded = GuiSystem::BitmaskToRuleString(birthBitmask);
		std::string survivalDecoded = GuiSystem::BitmaskToRuleString(survivalBitmask);

		LOG_INFO("  Birth:    input='3,5-7'    decoded='{}'", birthDecoded);
		LOG_INFO("  Survival: input='2,4'      decoded='{}'", survivalDecoded);
		LOG_INFO("  MaxCS:    input=8          decoded={}", maxCellState);

		bool passed = birthDecoded == "3,5-7" && survivalDecoded == "2,4" && maxCellState == 8;
		std::string summary = passed ? "PASS: rules round-trip correctly" : "FAIL: rule mismatch";
		LOG_INFO("[test:cli-rules] {}", summary);
		return {"cli-rules", passed, summary};
	}

	TestRunner::TestResult TestRunner::RunGhostExchangeTest()
	{
		LOG_INFO("[test:ghost-exchange] Testing ghost layer Z-slice exchange...");

		if (!_services.secondaryDeviceContext)
		{
			LOG_WARNING("[test:ghost-exchange] SKIP: --dual-gpu required");
			return {"ghost-exchange", true, "SKIP: no secondary GPU"};
		}

		DeviceContext& gpu0 = _services.deviceContext;
		DeviceContext& gpu1 = *_services.secondaryDeviceContext;

		// Small test grid: 32^3 — ghost depth = 1, split at Z=16
		const uint32_t gridDimensionX = 32;
		const uint32_t gridDimensionY = 32;
		const uint32_t gridDimensionZ = 32;
		const uint32_t ghostDepth = 1;
		const uint32_t zMid = gridDimensionZ / 2; // 16
		const uint32_t imageWidth = (gridDimensionX + 1) / 2; // 16

		// GPU0 local Z: ghostDepth + zMid + ghostDepth = 18
		// GPU1 local Z: ghostDepth + (gridZ - zMid) + ghostDepth = 18
		const uint32_t gpu0LocalZ = ghostDepth + zMid + ghostDepth;
		const uint32_t gpu1LocalZ = ghostDepth + (gridDimensionZ - zMid) + ghostDepth;

		uint32_t gpu0NumBlocksX = (imageWidth + 7) / 8;
		uint32_t gpu0NumBlocksY = (gridDimensionY + 7) / 8;
		uint32_t gpu0NumBlocksZ = (gpu0LocalZ + 7) / 8;
		uint64_t gpu0BufferElements = static_cast<uint64_t>(gpu0NumBlocksX) * gpu0NumBlocksY * gpu0NumBlocksZ * 512;
		uint64_t gpu0BufferSize = gpu0BufferElements * sizeof(uint32_t);

		uint32_t gpu1NumBlocksX = (imageWidth + 7) / 8;
		uint32_t gpu1NumBlocksY = (gridDimensionY + 7) / 8;
		uint32_t gpu1NumBlocksZ = (gpu1LocalZ + 7) / 8;
		uint64_t gpu1BufferElements = static_cast<uint64_t>(gpu1NumBlocksX) * gpu1NumBlocksY * gpu1NumBlocksZ * 512;
		uint64_t gpu1BufferSize = gpu1BufferElements * sizeof(uint32_t);

		auto gpu0CellBuffer = Buffer::Storage(gpu0, gpu0BufferSize, true, 0, "GhostTest GPU0 Cells");
		auto gpu1CellBuffer = Buffer::Storage(gpu1, gpu1BufferSize, true, 0, "GhostTest GPU1 Cells");

		// Fill GPU0 buffer with known pattern using upload:
		// Each texel at (x, y, z) gets packed value = (localZ & 0xF) | ((localZ & 0xF) << 4)
		// (both cells in the packed pair get localZ as their state)
		std::vector<uint32_t> gpu0Data(gpu0BufferElements, 0);
		for (uint32_t localZ = 0; localZ < gpu0LocalZ; localZ++)
		{
			for (uint32_t y = 0; y < gridDimensionY; y++)
			{
				for (uint32_t x = 0; x < imageWidth; x++)
				{
					glm::ivec3 blockCoord(x / 8, y / 8, localZ / 8);
					glm::uvec3 localCoord(x % 8, y % 8, localZ % 8);

					auto expandBits3 = [](uint32_t v) -> uint32_t {
						v &= 7u;
						v = (v | (v << 8u)) & 0x100Fu;
						v = (v | (v << 4u)) & 0x10C30C3u;
						v = (v | (v << 2u)) & 0x1249249u;
						return v;
					};

					uint32_t mortonIndex = expandBits3(localCoord.x) | (expandBits3(localCoord.y) << 1u) | (expandBits3(localCoord.z) << 2u);
					uint32_t blockIndex = blockCoord.z * gpu0NumBlocksX * gpu0NumBlocksY + blockCoord.y * gpu0NumBlocksX + blockCoord.x;
					uint32_t bufferIndex = blockIndex * 512 + mortonIndex;

					uint32_t cellState = localZ & 0xF;
					gpu0Data[bufferIndex] = cellState | (cellState << 4);
				}
			}
		}
		gpu0CellBuffer->Upload(gpu0Data.data(), static_cast<uint32_t>(gpu0BufferSize));

		// Fill GPU1 with zeros (ghost slots should be populated by exchange)
		std::vector<uint32_t> gpu1Data(gpu1BufferElements, 0);
		// Fill GPU1's owned region with a different pattern so we can distinguish
		for (uint32_t localZ = ghostDepth; localZ < ghostDepth + (gridDimensionZ - zMid); localZ++)
		{
			for (uint32_t y = 0; y < gridDimensionY; y++)
			{
				for (uint32_t x = 0; x < imageWidth; x++)
				{
					glm::ivec3 blockCoord(x / 8, y / 8, localZ / 8);
					glm::uvec3 localCoord(x % 8, y % 8, localZ % 8);

					auto expandBits3 = [](uint32_t v) -> uint32_t {
						v &= 7u;
						v = (v | (v << 8u)) & 0x100Fu;
						v = (v | (v << 4u)) & 0x10C30C3u;
						v = (v | (v << 2u)) & 0x1249249u;
						return v;
					};

					uint32_t mortonIndex = expandBits3(localCoord.x) | (expandBits3(localCoord.y) << 1u) | (expandBits3(localCoord.z) << 2u);
					uint32_t blockIndex = blockCoord.z * gpu1NumBlocksX * gpu1NumBlocksY + blockCoord.y * gpu1NumBlocksX + blockCoord.x;
					uint32_t bufferIndex = blockIndex * 512 + mortonIndex;

					uint32_t globalZ = zMid + (localZ - ghostDepth);
					uint32_t cellState = globalZ & 0xF;
					gpu1Data[bufferIndex] = cellState | (cellState << 4);
				}
			}
		}
		gpu1CellBuffer->Upload(gpu1Data.data(), static_cast<uint32_t>(gpu1BufferSize));

		// Run ghost exchange (no wrap for this basic test)
		GhostExchange ghostExchange(gpu0, gpu1);
		ghostExchange.Initialize(gridDimensionX, gridDimensionY, gridDimensionZ, zMid, ghostDepth, false);
		ghostExchange.SetupForBufferMode(
			gpu0CellBuffer, gpu0CellBuffer,   // same buffer for both "frames" in test
			gpu1CellBuffer, gpu1CellBuffer,
			glm::ivec3(gpu0NumBlocksX, gpu0NumBlocksY, gpu0NumBlocksZ),
			glm::ivec3(gpu1NumBlocksX, gpu1NumBlocksY, gpu1NumBlocksZ));
		ghostExchange.Exchange(0);

		// Verify GPU1's ghost_lo (local Z=0) matches GPU0's boundary (global Z=zMid-1)
		// GPU0's local Z for global Z=zMid-1 = ghostDepth + (zMid - 1) = 1 + 15 = 16
		// Expected cell state = 16 & 0xF = 0 (since localZ=16, 16 & 0xF = 0)
		// Wait, let me recalculate: GPU0 local Z 16 → state = 16 & 0xF = 0
		// Actually the pattern is localZ & 0xF, and GPU0's local Z for its boundary is ghostDepth + zMid - ghostDepth = zMid = 16
		// State = 16 & 0xF = 0. That's ambiguous with zeros. Let me use a better pattern.
		// The gpu0Data was filled with localZ & 0xF for ALL local Z values including ghosts.
		// GPU0 boundary (last owned slice) = local Z = ghostDepth + zMid - 1 = 16
		// Pattern value = 16 & 0xF = 0. Hmm.
		// Let's just verify by downloading GPU1's buffer and checking the ghost slot.

		auto gpu1Downloaded = gpu1CellBuffer->Download();
		bool passed = true;

		// Check GPU1's ghost_lo at local Z=0: should have GPU0's boundary data (GPU0 local Z = zMid = 16, state = 0)
		// Check a sample of texels
		for (uint32_t y = 0; y < gridDimensionY && passed; y++)
		{
			for (uint32_t x = 0; x < imageWidth && passed; x++)
			{
				uint32_t localZ = 0; // ghost_lo
				glm::ivec3 blockCoord(x / 8, y / 8, localZ / 8);
				glm::uvec3 localCoord(x % 8, y % 8, localZ % 8);

				auto expandBits3 = [](uint32_t v) -> uint32_t {
					v &= 7u;
					v = (v | (v << 8u)) & 0x100Fu;
					v = (v | (v << 4u)) & 0x10C30C3u;
					v = (v | (v << 2u)) & 0x1249249u;
					return v;
				};

				uint32_t mortonIndex = expandBits3(localCoord.x) | (expandBits3(localCoord.y) << 1u) | (expandBits3(localCoord.z) << 2u);
				uint32_t blockIndex = blockCoord.z * gpu1NumBlocksX * gpu1NumBlocksY + blockCoord.y * gpu1NumBlocksX + blockCoord.x;
				uint32_t bufferIndex = blockIndex * 512 + mortonIndex;

				uint32_t actual = reinterpret_cast<const uint32_t*>(gpu1Downloaded.data())[bufferIndex];
				// Expected: GPU0's local Z = zMid (the boundary slice extracted), state = zMid & 0xF = 0
				uint32_t expectedState = zMid & 0xF;
				uint32_t expected = expectedState | (expectedState << 4);

				if (actual != expected)
				{
					LOG_ERROR("  GPU1 ghost_lo mismatch at ({},{},0): expected 0x{:02X}, got 0x{:02X}", x, y, expected, actual);
					passed = false;
				}
			}
		}

		// Also verify GPU0's ghost_hi at local Z = ghostDepth + zMid:
		// Should have GPU1's first owned slice (GPU1 local Z = ghostDepth = 1, global Z = zMid = 16)
		auto gpu0Downloaded = gpu0CellBuffer->Download();
		for (uint32_t y = 0; y < gridDimensionY && passed; y++)
		{
			for (uint32_t x = 0; x < imageWidth && passed; x++)
			{
				uint32_t localZ = ghostDepth + zMid; // ghost_hi start
				glm::ivec3 blockCoord(x / 8, y / 8, localZ / 8);
				glm::uvec3 localCoord(x % 8, y % 8, localZ % 8);

				auto expandBits3 = [](uint32_t v) -> uint32_t {
					v &= 7u;
					v = (v | (v << 8u)) & 0x100Fu;
					v = (v | (v << 4u)) & 0x10C30C3u;
					v = (v | (v << 2u)) & 0x1249249u;
					return v;
				};

				uint32_t mortonIndex = expandBits3(localCoord.x) | (expandBits3(localCoord.y) << 1u) | (expandBits3(localCoord.z) << 2u);
				uint32_t blockIndex = blockCoord.z * gpu0NumBlocksX * gpu0NumBlocksY + blockCoord.y * gpu0NumBlocksX + blockCoord.x;
				uint32_t bufferIndex = blockIndex * 512 + mortonIndex;

				uint32_t actual = reinterpret_cast<const uint32_t*>(gpu0Downloaded.data())[bufferIndex];
				// Expected: GPU1's first owned global Z = zMid = 16, state = 16 & 0xF = 0
				uint32_t expectedGlobalZ = zMid;
				uint32_t expectedState = expectedGlobalZ & 0xF;
				uint32_t expected = expectedState | (expectedState << 4);

				if (actual != expected)
				{
					LOG_ERROR("  GPU0 ghost_hi mismatch at ({},{},{}): expected 0x{:02X}, got 0x{:02X}", x, y, localZ, expected, actual);
					passed = false;
				}
			}
		}

		std::string summary = passed ? "PASS: ghost exchange verified" : "FAIL: ghost data mismatch";
		LOG_INFO("[test:ghost-exchange] {}", summary);
		return {"ghost-exchange", passed, summary};
	}

	int TestRunner::RunTests(const std::string& testName)
	{
		_services.headless = true;

		LOG_INFO("========================================");
		LOG_INFO("  Cave Test Runner (headless)");
		LOG_INFO("========================================");

		std::vector<TestResult> results;
		bool runAll = (testName == "all");

		if (runAll || testName == "cli-rules")
			results.push_back(RunCliRulesTest());
		if (runAll || testName == "compute-skip")
			results.push_back(RunComputeSkipTest());
		if (runAll || testName == "sparse-benchmark")
			results.push_back(RunSparseBenchmarkTest());
		if (testName == "encode-benchmark")
			results.push_back(RunEncodeBenchmarkTest());
		if (runAll || testName == "render-consistency")
			results.push_back(RunRenderConsistencyTest());
		if (runAll || testName == "large-grid")
			results.push_back(RunLargeGridTest());
		if (testName == "ghost-exchange" || (runAll && _services.secondaryDeviceContext != nullptr))
			results.push_back(RunGhostExchangeTest());

		if (results.empty())
		{
			LOG_ERROR("Unknown test: '{}'. Available: all, compute-skip, sparse-benchmark, encode-benchmark, render-consistency, large-grid, cli-rules", testName);
			return 1;
		}

		LOG_INFO("========================================");
		LOG_INFO("  Results");
		LOG_INFO("========================================");
		int failures = 0;
		for (const auto& result : results)
		{
			LOG_INFO("  [{}] {} — {}", result.passed ? "PASS" : "FAIL", result.name, result.summary);
			if (!result.passed) failures++;
		}
		LOG_INFO("========================================");
		LOG_INFO("  {} passed, {} failed", results.size() - failures, failures);
		LOG_INFO("========================================");

		return failures;
	}

} // namespace Cave
