#pragma once

#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>

#include "../vulkanInstance.h"
#include "../deviceContext.h"
#include "../../simulation/simulation.h"
#include "../../common/applicationMode.h"

namespace Cave
{
	struct GuiState
	{
		// Shape
		int shape = 0; // 0 = Cube

		// Simulation rules
		int rulesInputMode = 0; // 0 = Rules Editor, 1 = Raw Parameters
		std::string birthRulesText;
		std::string survivalRulesText;
		int maxCellState = 5;
		bool faceNeighbors = true;
		bool edgeNeighbors = false;
		bool cornerNeighbors = false;
		bool wrapAtBoundary = false;
		char rawBirthAndMaxCellStateRulesInput[32] = "0";
		char rawSurvivalAndNeighborhoodRulesInput[32] = "0";

		// Grid & spawn
		int gridDimensionX = 225;
		int gridDimensionY = 225;
		int gridDimensionZ = 225;
		bool uniformGridDimensions = true;
		int spawnDimensionX = 25;
		int spawnDimensionY = 25;
		int spawnDimensionZ = 25;
		int spawnMode = 1; // 0 = random, 1 = filled
		float spawnRandomDensity = 0.5f; // [0,1] probability per cell when spawnMode == 0

		// Colors
		float aliveColor[3] = {0.373f, 0.294f, 0.545f};
		float deadColor[3] = {0.902f, 0.604f, 0.553f};

		// Rendering
		bool enableComputeSkip = true;
		bool computeNeighborCount = true;
		bool debugRandomCellColors = false;

		// Simulation state (updated by Engine each frame)
		uint32_t simulationTickCount = 0;

		// Performance timings (updated by Engine each frame)
		PerformanceTimings performanceTimings;

		// Lighting
		bool enableLighting = false;
		float ambientStrength = 0.3f;
		float diffuseStrength = 0.7f;
		float specularStrength = 0.5f;
		float shininess = 32.0f;
		float lightDirection[3] = {-0.5f, -1.0f, -0.3f};

		// Culling planes — hide cells past each enabled plane to reveal interior structure.
		// Plane equation: cells with dot(normal, worldPos) > offset are hidden.
		bool cullingPlaneEnabled[4] = {false, false, false, false};
		float cullingPlaneNormal[4][3] = {
			{1.0f, 0.0f, 0.0f},
			{0.0f, 1.0f, 0.0f},
			{0.0f, 0.0f, 1.0f},
			{1.0f, 0.0f, 0.0f},
		};
		float cullingPlaneOffset[4] = {0.0f, 0.0f, 0.0f, 0.0f};

		// Search mode parameters
		int searchShape = 0; // 0 = Cube
		int searchMinMaxCellState = 3;
		int searchMaxMaxCellState = 10;
		bool searchFaceNeighbors = true;
		bool searchEdgeNeighbors = false;
		bool searchCornerNeighbors = false;
		bool searchWrapAtBoundary = false;
		int searchGridDimensionX = 49;
		int searchGridDimensionY = 49;
		int searchGridDimensionZ = 49;
		int searchSpawnDimensionX = 25;
		int searchSpawnDimensionY = 25;
		int searchSpawnDimensionZ = 25;
		int searchSpawnMode = 1; // 0 = random, 1 = filled
		float searchSpawnRandomDensity = 0.5f; // [0,1], used when searchSpawnMode == 0
		int searchMaxTicksToSurvive = 100;
		std::string searchOutputFolderPath = "data/search/";

		// Sweep axes — each dimension that is enabled multiplies the total chunk count.
		// A sweep with grid × density × seed enabled produces (nGrids × nDensities × nSeeds)
		// chunks, all running serially on a single GPU path.
		bool searchGridSizeSweepEnabled = false;
		int searchGridSizeSweepMin = 30;
		int searchGridSizeSweepMax = 60;
		int searchGridSizeSweepStep = 10;

		// Density sweep — only meaningful when spawn mode == Random. Floats in [0,1]
		// are converted into density indices 0..15.
		bool searchDensitySweepEnabled = false;
		float searchDensitySweepMin = 0.25f;
		float searchDensitySweepMax = 1.0f;
		float searchDensitySweepStep = 0.25f;

		// Seed sweep — runs the current (grid, density, …) chunk K times with different
		// per-cell random seeds so rules that die due to unlucky initial conditions get
		// multiple chances. K=1 means no sweep.
		int searchSeedSweepCount = 1;

		// Cap on bits set in each rule bitmask (birth/survival). 0 = unbounded; K ≥ 1
		// prunes rules with > K bits set on either axis, making large-N neighborhoods
		// tractable. Conway's Life has birth=3 (1 bit), survival=2,3 (2 bits) = K=2
		// is enough to cover it.
		int searchMaxRuleBits = 0;

		// VRAM budget per GPU (gigabytes). The chunk scheduler packs as many chunks as
		// fit into this per-GPU budget, spawning new chunks as old ones free memory.
		float searchGpuVramBudgetGb = 40.0f;

		// Upfront chunk partitioning. N > 1 emits N copies of each chunk, each taking
		// a non-overlapping slice of the survival rule range — lets a single-grid
		// filled-spawn search saturate multiple GPUs without waiting for work-stealing.
		int searchChunksPerConfig = 64;

		// Simulation kernel workgroup size (threads per local block). Hardcoded into
		// the generated shader's `layout(local_size_x = N)` and the host-side dispatch
		// math. Valid: 32, 64, 128, 256, 512.
		int searchWorkgroupSize = 256;

		// Looking Glass parameters (M5)
		// Only displayed in the GUI when ApplicationMode::LookingGlass is active.
		// Display index 0 is the first available LG display reported by Bridge.
		// Viewcone width drives the per-view eye offset in the quilt shader (35°
		// is LG Portrait's stock value; range 0-60 covers most useful settings).
		// Zoom is forwarded to draw_interop_quilt_texture_gl as the zoom param.
		// View count lets the user trade quality for fps — Bridge interpolates
		// when fewer views are supplied than the native 48.
		int lookingGlassDisplayIndex = 0;
		float lookingGlassViewconeDegrees = 35.0f;
		float lookingGlassZoom = 1.0f;
		int lookingGlassViewCount = 48; // Valid: 24, 32, 45, 48

		// Video encoding parameters
		int videoSourceMode = 0; // 0 = import JSON, 1 = manual entry (raw uint64)
		char videoManualBirthRulesInput[32] = "0";
		char videoManualSurvivalRulesInput[32] = "0";
		std::string videoSourceJsonPath;
		std::vector<SimulationParameters> videoImportedPermutations;
		std::vector<uint32_t> videoImportedTicksSurvived;
		int videoSelectedPermutationIndex = -1; // -1 = not selected
		bool videoEncodeAll = true;
		int videoResolutionWidth = 1920;
		int videoResolutionHeight = 1080;
		int videoFps = 30;
		int videoDurationTicks = 300;
		std::string videoOutputFolderPath = "data/videos/";
		float videoEncodingProgress = 0.0f;
		uint32_t videoCurrentFrame = 0;
		uint32_t videoTotalFrames = 0;
		uint32_t videoCurrentJob = 0;
		uint32_t videoTotalJobs = 0;

		// Metadata overlay baked into encoded frames
		bool videoShowOverlay = true;
		int videoOverlayPosition = 0; // 0 = top-left, 1 = top-right

		// Camera
		int cameraMode = 0; // 0 = FixedPaceOrbit, 1 = FixedRateOrbit, 2 = FreeCamera
		float orbitRadius = 50.0f;
		float orbitAngularVelocity = 0.01f;
		float orbitElevation = 0.4f;
		float fieldOfViewDegrees = 45.0f;
		float movementSpeed = 20.0f;
		float lookSpeed = 1.5f;
	};

	class GuiSystem
	{
	private:
		VulkanInstance& _vulkanInstance;
		DeviceContext& _deviceContext;
		vk::DescriptorPool _imguiDescriptorPool;

		GuiState _state;

		bool _resetRequested = false;
		bool _startRequested = false;
		bool _stopRequested = false;

		ApplicationMode _enterModeRequested = ApplicationMode::None;
		bool _exitModeRequested = false;
		bool _pauseRequested = false;
		bool _resumeRequested = false;
		bool _stepRequested = false;
		bool _encodeRequested = false;
		bool _cancelEncodeRequested = false;

		void CreateImGuiDescriptorPool();
		void BuildModeSelector();

	public:
		// Per-frame ImGui panel for Looking Glass mode runtime controls.
		// Called from LookingGlassMode::OnFrame between ImGui NewFrame and Render.
		// Reactive: viewcone/zoom feed straight into per-view shader push constants.
		// Display index changes are debounced (the user must re-enter the mode for
		// Bridge to re-create its on-display GL window).
		void BuildLookingGlassPanel();

		// Rule string helpers
		static std::string BitmaskToRuleString(uint64_t bitmask);
		static uint64_t RuleStringToBitmask(const std::string& ruleString);
		GuiSystem(VulkanInstance& vulkanInstance, DeviceContext& deviceContext, vk::Format swapchainImageFormat);
		~GuiSystem();

		GuiSystem(const GuiSystem&) = delete;
		GuiSystem& operator=(const GuiSystem&) = delete;

		// Builds a full ImGui frame for the mode selector (ApplicationMode::None).
		void BuildIdleFrame();
		void RecordDrawCommands(vk::CommandBuffer commandBuffer);

		GuiState& GetState() { return _state; }
		const GuiState& GetState() const { return _state; }

		bool ConsumeResetRequest();
		bool ConsumeStartRequest();
		bool ConsumeStopRequest();
		ApplicationMode ConsumeEnterModeRequest();
		bool ConsumeExitModeRequest();
		bool ConsumePauseRequest();
		bool ConsumeResumeRequest();
		bool ConsumeStepRequest();
		bool ConsumeEncodeRequest();
		bool ConsumeCancelEncodeRequest();

		// Programmatic request injection (for CLI auto-start and mode-owned GUI builders)
		void InjectStartRequest() { _startRequested = true; }
		void InjectEncodeRequest() { _encodeRequested = true; }
		void InjectStopRequest() { _stopRequested = true; }
		void InjectResetRequest() { _resetRequested = true; }
		void InjectPauseRequest() { _pauseRequested = true; }
		void InjectResumeRequest() { _resumeRequested = true; }
		void InjectStepRequest() { _stepRequested = true; }
		void InjectCancelEncodeRequest() { _cancelEncodeRequested = true; }

		// Frame lifecycle for mode-owned GUI builders.
		// BeginModeFrame: starts ImGui frame, opens Settings window with "Back" button.
		// EndModeFrame: closes Settings window, draws FPS overlay, calls ImGui::Render.
		void BeginModeFrame();
		void EndModeFrame();

		// Build SimulationParameters and ColorRules from current GUI state
		SimulationParameters BuildSimulationParameters() const;
		ColorRules BuildColorRules() const;
	};
}
