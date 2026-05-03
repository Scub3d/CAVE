#pragma once

#include <string>
#include "applicationMode.h"

namespace Cave
{
	enum class CellStateStorageMode { Buffer, Image };

	struct StartupConfig
	{
		bool autoStart = false;
		ApplicationMode mode = ApplicationMode::None;

		// Shared params
		int gridSize = 49;
		int spawnSize = 25;
		int spawnMode = 1; // 0 = random, 1 = filled
		float spawnRandomDensity = 0.5f; // [0,1] fraction of random spawn cells that fill
		int maxCellState = 5;
		std::string neighborhood = "F";
		int shape = 0; // 0 = cube, 1 = elongated dodecahedron

		// Search params
		int minMaxCellState = 3;
		int maxMaxCellState = 10;
		int maxTicks = 100;
		std::string searchOutputFolder = "data/search/";

		// Sweep axes (search mode). When enabled, each axis multiplies the total chunk count.
		bool searchGridSizeSweepEnabled = false;
		int searchGridSizeSweepMin = 30;
		int searchGridSizeSweepMax = 60;
		int searchGridSizeSweepStep = 10;

		bool searchDensitySweepEnabled = false;
		float searchDensitySweepMin = 0.25f;
		float searchDensitySweepMax = 1.0f;
		float searchDensitySweepStep = 0.25f;

		int searchSeedSweepCount = 1;

		// Cap on the number of bits set in each rule bitmask (birth and survival) during
		// search iteration. 0 means "unbounded" — iterate the full 2^N space. K > 0
		// prunes dense rules (e.g. K=3 restricts to rules with ≤ 3 birth bits and ≤ 3
		// survival bits set, matching Conway's-Life-class sparseness). Makes large-N
		// neighborhoods like FEC actually tractable to enumerate.
		int searchMaxRuleBits = 0;

		// VRAM budget per GPU for the chunk scheduler. Default 40 GB (leaves ~8 GB
		// overhead on RTX 8000 for driver/framebuffers). The scheduler estimates each
		// chunk's cost and packs as many as fit into the budget, spawning new chunks
		// as old ones free memory. Set via --gpu-vram-budget-gb or the search GUI.
		float searchGpuVramBudgetGb = 40.0f;

		// Upfront partitioning of each (grid × density × seed) chunk's rule space.
		// N > 1 emits N copies of each chunk, each pre-assigned a non-overlapping slice
		// of the survival rule range. Lets a single-grid filled-spawn search saturate
		// dual GPUs without relying on reactive cross-chunk work-stealing.
		int chunksPerConfig = 64;

		// Threads per workgroup for the simulation compute kernel. Lower values produce
		// more workgroups in flight per dispatch (better SM coverage when grids are small)
		// at the cost of more launch overhead per warp. Valid: 32, 64, 128, 256, 512.
		int searchWorkgroupSize = 256;

		// Rendering params
		std::string birthRules;
		std::string survivalRules;

		// Video encoding params
		std::string jsonPath;
		bool encodeAll = false;
		int selectedIndex = -1;
		int videoWidth = 1920;
		int videoHeight = 1080;
		int videoFps = 30;
		int videoDurationTicks = 300;
		std::string videoOutputFolder = "data/videos/";

		// Camera params
		float orbitSpeed = -1.0f;      // -1 = use default
		float orbitAngle = 1e30f;      // large sentinel = unset (angle in radians)
		float orbitElevation = 1e30f;  // unset (radians)
		float orbitRadius = -1.0f;     // -1 = unset
		bool disableCameraZoom = false; // video mode: hold camera at endRadius for entire clip

		// Automation params
		bool enableComputeSkip = true;
		int maxFrames = -1;                                  // -1 = unlimited
		int captureFrame = -1;                               // -1 = no capture
		std::string captureOutputPath = "data/capture.png";
		bool logToFile = false;
		bool benchmark = false;
		bool headless = false;
		std::string testName;
		int forcedGpuIndex = -1; // -1 = auto-select, 0+ = force specific GPU
		bool dualGpu = false;
		CellStateStorageMode storageMode = CellStateStorageMode::Image;
	};
}
