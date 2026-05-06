// GLM defines must come before any glm header. vulkanInstance.h sets these but
// other headers in this TU pull glm in earlier, so we set them up front to
// keep the mangled name of glm::uvec3 consistent with simulation.cpp's TU
// (otherwise the Simulation constructor lookup fails to link).
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "lookingGlassMode.h"
#include "../common/logger.h"
#include "../common/startupConfig.h"
#include "../engine.h"
#include "../vulkan/deviceContext.h"
#include "../vulkan/vulkanInstance.h"
#include "../vulkan/systems/guiSystem.h"
#include "../camera.h"
#include "../shapes/cube.h"
#include "../shapes/elongatedRhombicDodecahedron.h"

#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

#ifdef LOOKING_GLASS_HAS_BRIDGE
// Bridge SDK header. Vendored at lib/bridge/bridge.h. We use the inline
// Controller class defined in bridge.h, which LoadLibrary's bridge_inproc.dll
// at runtime (auto-discovers the installed Bridge service via its
// settings.json — no copy of the DLL needs to live next to Cave.exe).
//
// bridge.h pulls in <Winsock2.h> and <Windows.h>. Keep the include LAST in
// this TU and #undef Windows macros that collide with vk-hpp method names.
#include <bridge.h>
#ifdef CreateSemaphore
#undef CreateSemaphore
#endif
#ifdef CreateFence
#undef CreateFence
#endif
#endif

namespace Cave
{
	void LookingGlassMode::InitializeSimulation(ModeServices& services)
	{
		GuiState& guiState = services.guiSystem.GetState();

		// Apply LG-mode defaults if the user hasn't supplied them already.
		// Default grid is 128 — at the Portrait's native quilt resolution
		// (3360×3360, 420×560 per view) each cell takes several pixels and
		// reads as a crisp block. Larger grids (255³, 511³, 1023³) are still
		// supported via --grid; cell edges just soften proportionally.
		if (guiState.gridDimensionX < 64) guiState.gridDimensionX = 128;
		if (guiState.gridDimensionY < 64) guiState.gridDimensionY = 128;
		if (guiState.gridDimensionZ < 64) guiState.gridDimensionZ = 128;
		if (guiState.spawnDimensionX < 1) guiState.spawnDimensionX = 25;
		if (guiState.spawnDimensionY < 1) guiState.spawnDimensionY = 25;
		if (guiState.spawnDimensionZ < 1) guiState.spawnDimensionZ = 25;

		// CA rules — Conway-class 4-survives, 1/4-6 birth (matches RenderingMode default).
		guiState.birthRulesText = "1,4-6";
		guiState.survivalRulesText = "4";
		guiState.maxCellState = 5;
		guiState.faceNeighbors = true;
		guiState.edgeNeighbors = false;
		guiState.cornerNeighbors = false;
		guiState.wrapAtBoundary = false;

		// Lighting — enabled by default for LG mode and tuned for harder edges
		// to compensate for the lenticular display's optical softening at the
		// front/back of the focus plane. Strong specular + tight shininess
		// produces crisp highlights that read as sharper than flat shading.
		guiState.enableLighting = true;
		guiState.ambientStrength = 0.20f;
		guiState.diffuseStrength = 0.85f;
		guiState.specularStrength = 0.85f;
		guiState.shininess = 64.0f;
		guiState.lightDirection[0] = -0.5f;
		guiState.lightDirection[1] = -1.0f;
		guiState.lightDirection[2] = -0.3f;

		// Random spawn instead of filled — yields more visually interesting
		// dynamics (filled-spawn collapses to equilibrium quickly with these
		// rules and reads as a static hologram).
		guiState.spawnMode = 0;
		if (guiState.spawnRandomDensity <= 0.0f || guiState.spawnRandomDensity > 1.0f)
			guiState.spawnRandomDensity = 0.5f;

		// Bump the orbit camera so motion is obvious through the holographic
		// view even when the CA stabilizes. Default rendering-mode orbit (0.01
		// rad/sec ≈ 0.6°/sec) reads as still on a small display.
		guiState.cameraMode = 0; // FixedPaceOrbit
		guiState.orbitAngularVelocity = 0.15f; // ~8.6°/sec — visible without being disorienting

		// Build the Simulation via the same constructor RenderingMode uses (single
		// shot, all params at once). Single-GPU storage = Image (production).
		const CellStateStorageMode storageMode =
			(services.startupConfig && services.startupConfig->storageMode == CellStateStorageMode::Buffer)
				? CellStateStorageMode::Buffer : CellStateStorageMode::Image;

		glm::uvec3 gridDims(guiState.gridDimensionX, guiState.gridDimensionY, guiState.gridDimensionZ);
		glm::uvec3 spawnDims(guiState.spawnDimensionX, guiState.spawnDimensionY, guiState.spawnDimensionZ);
		std::shared_ptr<Shape> shape = (guiState.shape == 1)
			? ElongatedRhombicDodecahedron::Create()
			: Cube::Create();
		ColorRules colorRules = services.guiSystem.BuildColorRules();
		SimulationParameters simParams = services.guiSystem.BuildSimulationParameters();

		_simulationContext.simulation = Simulation(
			gridDims, spawnDims, guiState.spawnMode,
			simParams, colorRules, shape, guiState.spawnRandomDensity);

		_simulationContext.simulationRenderer = std::make_unique<VulkanSimulationRenderer>(
			services.deviceContext, _simulationContext.simulation, services.framesInFlight, storageMode);
		_simulationContext.computeSystem = std::make_unique<ComputeSystem>(
			services.deviceContext, *_simulationContext.simulationRenderer);

		// Camera oriented for the per-tile aspect (4:3 portrait → 0.75).
		float aspect = static_cast<float>(QuiltRenderSystem::kViewWidth)
		             / static_cast<float>(QuiltRenderSystem::kViewHeight);
		_simulationContext.camera = std::make_unique<Camera>(
			services.vulkanInstance.GetVulkanGLFWWindow(), aspect);

		// Camera defaults to its own internal state (radius=50, angular
		// velocity=0.01 rad/sec). Push our LG-mode defaults so the orbit is
		// actually visible and centered correctly for the grid we're rendering.
		_simulationContext.camera->SetCameraMode(CameraMode::FixedPaceOrbit);
		// Place camera at distance ~0.55× max grid extent. Grid spans ±halfGrid
		// from origin (halfGrid ≈ 0.5*gridDim), so radius=0.55*gridDim puts the
		// camera ~0.05*gridDim outside the nearest face — close enough that the
		// CA fills most of the hologram volume, far enough that grid corners
		// stay inside the projection frustum.
		int maxGridDim = guiState.gridDimensionX;
		if (guiState.gridDimensionY > maxGridDim) maxGridDim = guiState.gridDimensionY;
		if (guiState.gridDimensionZ > maxGridDim) maxGridDim = guiState.gridDimensionZ;
		float orbitRadius = 0.55f * static_cast<float>(maxGridDim);
		_simulationContext.camera->SetOrbitRadius(orbitRadius);
		_simulationContext.camera->SetOrbitAngularVelocity(0.15f);
		_simulationContext.camera->SetOrbitElevation(0.4f);
		guiState.orbitRadius = orbitRadius;

		LOG_INFO("LookingGlassMode: simulation initialized — grid {}x{}x{}, spawn {}x{}x{}",
			guiState.gridDimensionX, guiState.gridDimensionY, guiState.gridDimensionZ,
			guiState.spawnDimensionX, guiState.spawnDimensionY, guiState.spawnDimensionZ);
	}

	void LookingGlassMode::Enter(ModeServices& services)
	{
		LOG_INFO("LookingGlassMode::Enter");

		// Always set up the simulation + quilt renderer — these work even
		// without Bridge. Without Bridge, OnFrame falls through to a clean
		// error path; the simulation infrastructure is built so a future user
		// could route the quilt elsewhere (debug PNG output, alternative
		// holographic display, etc.).
		InitializeSimulation(services);
		_quiltRenderSystem = std::make_unique<QuiltRenderSystem>(services.deviceContext);

#ifndef LOOKING_GLASS_HAS_BRIDGE
		LOG_ERROR("Looking Glass Bridge SDK not vendored. To enable hologram output:");
		LOG_ERROR("  1. Install Looking Glass Bridge from https://lookingglassfactory.com/software/looking-glass-bridge");
		LOG_ERROR("  2. Copy bridge.h, bridge_inproc.lib, bridge_inproc.dll into lib/bridge/ in this repo");
		LOG_ERROR("  3. Reconfigure (cmake -B build) and rebuild — CMake auto-detects lib/bridge/");
		LOG_ERROR("  4. Re-run --mode looking-glass");
		LOG_ERROR("Until then, you can validate the quilt format with: --lg-test=save-quilt");
		_fatalError = true;
		return;
#else
		// 1. Construct the Controller. It loads bridge_inproc.dll at runtime
		//    from the installed Bridge service location (parsed from
		//    %APPDATA%/Looking Glass/Bridge/settings.json). No DLL needs to
		//    sit next to Cave.exe.
		Controller* controller = new Controller();
		_bridgeController = controller;
		if (!controller->Initialize(L"Cave Looking Glass"))
		{
			LOG_ERROR("LookingGlass: Controller::Initialize() failed. Is the Looking Glass Bridge service installed and running?");
			delete controller;
			_bridgeController = nullptr;
			_fatalError = true;
			return;
		}
		_bridgeInitialized = true;

		// 2. Enumerate connected Looking Glass displays. Bridge's "display
		//    index" is an opaque ID (e.g. index=3 for the only connected
		//    Portrait), not a 0-based position. We collect them here so the
		//    user-facing GuiState.lookingGlassDisplayIndex can stay 0-based
		//    (0 = first listed Portrait) while we hand Bridge the real index.
		std::vector<unsigned long> bridgeDisplayIndices;
		{
			int displayCount = 0;
			controller->GetDisplays(&displayCount, nullptr);
			LOG_INFO("LookingGlass: Bridge reports {} display(s) connected", displayCount);
			if (displayCount > 0)
			{
				bridgeDisplayIndices.resize(static_cast<size_t>(displayCount));
				if (controller->GetDisplays(&displayCount, bridgeDisplayIndices.data()))
				{
					for (int i = 0; i < displayCount; ++i)
					{
						unsigned long w = 0, h = 0;
						float viewcone = 0.0f;
						controller->GetDimensionsForDisplay(bridgeDisplayIndices[i], &w, &h);
						controller->GetViewConeForDisplay(bridgeDisplayIndices[i], &viewcone);
						LOG_INFO("  display[{}] bridgeIndex={} size={}x{} viewcone={:.1f}°",
							i, bridgeDisplayIndices[i], w, h, viewcone);
					}
				}
				else
				{
					bridgeDisplayIndices.clear();
				}
			}
		}

		if (bridgeDisplayIndices.empty())
		{
			LOG_ERROR("LookingGlass: no Looking Glass displays detected. Make sure the Portrait is plugged into a GPU output Windows treats as a connected display, that Bridge service is running, and that the Portrait is powered on.");
			controller->Uninitialize();
			delete controller;
			_bridgeController = nullptr;
			_bridgeInitialized = false;
			_fatalError = true;
			return;
		}

		// 3. Pick which display in the enumeration list to use (0 = first).
		int desiredListPos = (services.startupConfig && services.startupConfig->lookingGlassDisplayIndex >= 0)
			? services.startupConfig->lookingGlassDisplayIndex
			: 0;
		if (desiredListPos >= static_cast<int>(bridgeDisplayIndices.size()))
		{
			LOG_WARNING("LookingGlass: requested display position {} >= count {}; falling back to position 0",
				desiredListPos, static_cast<int>(bridgeDisplayIndices.size()));
			desiredListPos = 0;
		}
		unsigned long bridgeDisplayIndex = bridgeDisplayIndices[static_cast<size_t>(desiredListPos)];

		// 4. Build our local GL interop context FIRST. Bridge's
		//    instance_window_gl needs a current GL context to share resources
		//    with via wglShareLists — without one, window creation fails.
		_glContext = std::make_unique<GlInteropContext>();
		if (!_glContext->Initialize())
		{
			LOG_ERROR("LookingGlass: GL context init failed");
			controller->Uninitialize();
			delete controller;
			_bridgeController = nullptr;
			_bridgeInitialized = false;
			_fatalError = true;
			return;
		}
		_glContext->MakeCurrent();

		// 5. Create the on-display GL window. Bridge owns this window and
		//    shares resources with whatever GL context is current on this
		//    thread.
		WINDOW_HANDLE bridgeWnd = 0;
		bool gotWindow = controller->InstanceWindowGL(&bridgeWnd, bridgeDisplayIndex);
		if (!gotWindow)
		{
			LOG_WARNING("LookingGlass: InstanceWindowGL({}) failed; retrying with FIRST_LOOKING_GLASS_DEVICE",
				bridgeDisplayIndex);
			gotWindow = controller->InstanceWindowGL(&bridgeWnd, static_cast<unsigned long>(FIRST_LOOKING_GLASS_DEVICE));
		}
		if (!gotWindow)
		{
			LOG_ERROR("LookingGlass: InstanceWindowGL() failed for both explicit index {} and FIRST_LOOKING_GLASS_DEVICE. "
				"This often means the Bridge service has stale state — try restarting LookingGlassBridge.exe.",
				bridgeDisplayIndex);
			_glContext.reset();
			controller->Uninitialize();
			delete controller;
			_bridgeController = nullptr;
			_bridgeInitialized = false;
			_fatalError = true;
			return;
		}
		LOG_INFO("LookingGlass: opened GL window (bridgeWnd={})", bridgeWnd);
		_bridgeWindow = reinterpret_cast<void*>(static_cast<uintptr_t>(bridgeWnd));

		// Pull the calibrated viewcone + default quilt settings from Bridge so
		// per-view eye shifts and the submit aspect match this exact Portrait.
		// Stock LG Portrait reports viewcone=40°, aspect=0.75 — but always
		// prefer the device-reported values to handle hardware revisions.
		{
			float viewconeDeg = 0.0f;
			if (controller->GetViewCone(bridgeWnd, &viewconeDeg) && viewconeDeg > 0.0f)
			{
				_bridgeViewconeRadians = glm::radians(viewconeDeg);
			}
			float quiltAspect = 0.0f;
			int qw = 0, qh = 0, qcols = 0, qrows = 0;
			if (controller->GetDefaultQuiltSettings(bridgeWnd, &quiltAspect, &qw, &qh, &qcols, &qrows))
			{
				_bridgeQuiltAspect = quiltAspect;
				LOG_INFO("LookingGlass: calibrated viewcone={:.1f}° quilt_aspect={:.3f} default_quilt={}x{} ({}x{}={} views)",
					viewconeDeg, quiltAspect, qw, qh, qcols, qrows, qcols * qrows);
			}
			else
			{
				LOG_INFO("LookingGlass: calibrated viewcone={:.1f}° (using default quilt aspect 0.75)", viewconeDeg);
			}
		}

		_vkSignalGlSem = std::make_unique<ExternalBinarySemaphore>(services.deviceContext);
		_glSignalVkSem = std::make_unique<ExternalBinarySemaphore>(services.deviceContext);

		// 6. Import the quilt texture + both semaphores into GL. Bridge's
		//    InstanceWindowGL likely made its own context current; switch back
		//    to ours before issuing GL import calls.
		_glContext->MakeCurrent();
		_glQuiltTexture = _glContext->ImportImageHandleAsTexture(
			_quiltRenderSystem->GetQuiltMemoryWin32Handle(),
			_quiltRenderSystem->GetQuiltMemorySize(),
			QuiltRenderSystem::kQuiltWidth, QuiltRenderSystem::kQuiltHeight,
			0x8058 /* GL_RGBA8 */);
		_glVkToGlSemaphore = _glContext->ImportSemaphoreHandle(_vkSignalGlSem->GetWin32Handle());
		_glGlToVkSemaphore = _glContext->ImportSemaphoreHandle(_glSignalVkSem->GetWin32Handle());

		LOG_INFO("LookingGlass: ready — bridge display index {}, quilt {}x{} (48 views)",
			bridgeDisplayIndex, QuiltRenderSystem::kQuiltWidth, QuiltRenderSystem::kQuiltHeight);
#endif
	}

	bool LookingGlassMode::OnFrame(ModeServices& services, float deltaTimeInSeconds)
	{
		if (_fatalError) return false;

		// Tick the camera (orbit auto-rotation).
		if (_simulationContext.camera)
			_simulationContext.camera->Update(deltaTimeInSeconds);

		// Compute tick. frameIndex selects the ping-pong direction
		// (read=buffer[frameIndex], write=buffer[(frameIndex+1)%framesInFlight])
		// so we MUST alternate it each tick or the simulation reads the same
		// state forever. enableComputeSkip=true uses the skip-grid optimization
		// for sparse-state speedup. renderFence is passed as null since
		// LookingGlassMode doesn't use the framesInFlight double-buffering model
		// that PresentSystem requires — we serialize per-frame here (acceptable
		// at the small grid sizes LG targets).
		uint32_t frameIndex = _tickCount % services.framesInFlight;
		_simulationContext.computeSystem->Tick(frameIndex, vk::Fence{}, true /*dispatchCompute*/, true /*enableComputeSkip*/);

		// Build per-frame data. SimulationContext::BuildCameraData fills in the
		// matrices from the camera (model+view+projection) and lighting params
		// from GuiState.
		GuiState& guiState = services.guiSystem.GetState();
		CameraData cameraData = _simulationContext.BuildCameraData(guiState);
		RayMarchPushConstants pushConstants = _simulationContext.BuildRayMarchPushConstants(guiState);

		// Prefer the calibrated viewcone from Bridge (set in Enter); fall back to
		// the GuiState slider if calibration wasn't read.
		float viewconeRadians = (_bridgeViewconeRadians > 0.0f)
			? _bridgeViewconeRadians
			: glm::radians(guiState.lookingGlassViewconeDegrees);

		// Parallax scale — multiplies the effective viewcone. Smaller values
		// reduce per-view eye displacement, which expands the "depth budget"
		// where content stays sharp on the lenticular display at the cost of
		// less dramatic 3D pop. 0.5× the calibrated 40° gives a usable balance:
		// front-of-grid cells (which suffer most from optical defocus due to
		// perspective amplification) read noticeably crisper.
		viewconeRadians *= 0.5f;

		// Render the quilt. Wait on compute timeline; signal binary external sem
		// for GL handshake (only when Bridge is present — otherwise pass nullptr
		// and the quilt is just rendered for the next frame's reuse).
		// renderFromFrame is the buffer the compute pass JUST wrote to —
		// (frameIndex+1) % framesInFlight.
		vk::Semaphore computeSem = _simulationContext.computeSystem->GetCompletedSemaphore(frameIndex);
		uint64_t computeValue = _simulationContext.computeSystem->GetCompletedSemaphoreSignalValue(frameIndex);
		uint32_t renderFromFrame = (frameIndex + 1) % services.framesInFlight;

#ifdef LOOKING_GLASS_HAS_BRIDGE
		_quiltRenderSystem->RenderFrame(
			*_simulationContext.simulationRenderer,
			cameraData, pushConstants, viewconeRadians,
			computeSem, computeValue,
			_vkSignalGlSem.get(),
			renderFromFrame);

		// GL handshake + Bridge submit.
		_glContext->MakeCurrent();
		_glContext->WaitSemaphoreOnTexture(_glVkToGlSemaphore, _glQuiltTexture);

		if (_bridgeWindow && _bridgeController)
		{
			WINDOW_HANDLE wnd = static_cast<WINDOW_HANDLE>(reinterpret_cast<uintptr_t>(_bridgeWindow));
			Controller* controller = static_cast<Controller*>(_bridgeController);
			// Use Bridge's reported quilt aspect when available; fall back to
			// the LG Portrait stock value of 0.75 if the SDK didn't return one.
			float quiltAspect = (_bridgeQuiltAspect > 0.0f) ? _bridgeQuiltAspect : 0.75f;
			controller->DrawInteropQuiltTextureGL(
				wnd,
				static_cast<unsigned long long>(_glQuiltTexture),
				PixelFormats::RGBA,
				QuiltRenderSystem::kQuiltWidth,
				QuiltRenderSystem::kQuiltHeight,
				QuiltRenderSystem::kQuiltColumns,
				QuiltRenderSystem::kQuiltRows,
				quiltAspect,
				guiState.lookingGlassZoom);
		}

		_glContext->SignalSemaphoreOnTexture(_glGlToVkSemaphore, _glQuiltTexture);
		_glContext->Flush();
#else
		(void)computeSem; (void)computeValue;
		// Without Bridge, OnFrame should never be reached (Enter sets _fatalError).
		return false;
#endif
		_tickCount++;

		// Once per second, log fps and tick rate. Confirms the simulation is
		// actually ticking (vs. one frame stuck on the Portrait).
		double now = glfwGetTime();
		if (_lastFpsLogTime == 0.0) _lastFpsLogTime = now;
		if (now - _lastFpsLogTime >= 1.0)
		{
			uint32_t ticksThisInterval = _tickCount - _lastFpsLogTickCount;
			float fps = static_cast<float>(ticksThisInterval) / static_cast<float>(now - _lastFpsLogTime);
			LOG_INFO("LookingGlass: tick={} fps={:.1f}", _tickCount, fps);
			_lastFpsLogTime = now;
			_lastFpsLogTickCount = _tickCount;
		}

		return true;
	}

	void LookingGlassMode::Exit(ModeServices& services)
	{
		LOG_INFO("LookingGlassMode::Exit");
		services.deviceContext.GetDevice().waitIdle();

#ifdef LOOKING_GLASS_HAS_BRIDGE
		if (_glContext)
		{
			_glContext->MakeCurrent();
			if (_glQuiltTexture)    _glContext->DeleteTexture(_glQuiltTexture);
			if (_glVkToGlSemaphore) _glContext->DeleteSemaphore(_glVkToGlSemaphore);
			if (_glGlToVkSemaphore) _glContext->DeleteSemaphore(_glGlToVkSemaphore);
		}
		_glQuiltTexture = 0;
		_glVkToGlSemaphore = 0;
		_glGlToVkSemaphore = 0;

		// HANDLEs were transferred to GL on import — the wrapper destructors
		// must skip CloseHandle to avoid double-close.
		if (_vkSignalGlSem) _vkSignalGlSem->MarkHandleTransferredToGl();
		if (_glSignalVkSem) _glSignalVkSem->MarkHandleTransferredToGl();
#endif

		_quiltRenderSystem.reset();
		_vkSignalGlSem.reset();
		_glSignalVkSem.reset();
		_glContext.reset();
		_simulationContext.DestroyGpuResources();

#ifdef LOOKING_GLASS_HAS_BRIDGE
		if (_bridgeController)
		{
			Controller* controller = static_cast<Controller*>(_bridgeController);
			if (_bridgeInitialized)
			{
				controller->Uninitialize();
				_bridgeInitialized = false;
			}
			delete controller;
			_bridgeController = nullptr;
		}
		_bridgeWindow = nullptr;
#endif
	}

// --------------------------------------------------------------------------
// Standalone Bridge displays enumeration. Useful for diagnosing "no Portrait
// detected" — invokes Controller::GetDisplays and prints what Bridge sees,
// then exits. Wired in via --lg-test=displays.
//
// Defined here (not in glInterop.cpp) because <bridge.h> pulls in <Windows.h>
// which collides with the constexpr GL constants in glInterop.cpp.
// --------------------------------------------------------------------------
int RunBridgeDisplaysTest()
{
#ifndef LOOKING_GLASS_HAS_BRIDGE
	LOG_ERROR("--lg-test=displays needs LOOKING_GLASS_HAS_BRIDGE (Bridge SDK not vendored at lib/bridge/)");
	return 1;
#else
	LOG_INFO("=== Looking Glass smoke test: enumerate displays ===");
	Controller controller;
	if (!controller.Initialize(L"Cave LG Displays Test"))
	{
		LOG_ERROR("Controller::Initialize() failed — is Looking Glass Bridge installed/running?");
		return 1;
	}

	int count = 0;
	if (!controller.GetDisplays(&count, nullptr))
	{
		LOG_ERROR("GetDisplays(query count) returned false");
		controller.Uninitialize();
		return 1;
	}
	LOG_INFO("Bridge reports {} Looking Glass display(s) connected", count);

	if (count <= 0)
	{
		LOG_WARNING("No displays detected. Common causes: HDMI not on the same GPU as the visible Windows display, BIOS iGPU disabled, Portrait powered off, or Bridge service not seeing the device.");
		controller.Uninitialize();
		return 0;
	}

	std::vector<unsigned long> indices(static_cast<size_t>(count));
	if (!controller.GetDisplays(&count, indices.data()))
	{
		LOG_ERROR("GetDisplays(fetch indices) returned false");
		controller.Uninitialize();
		return 1;
	}

	for (int i = 0; i < count; ++i)
	{
		unsigned long idx = indices[static_cast<size_t>(i)];
		unsigned long w = 0, h = 0;
		float viewcone = 0.0f;
		float aspect = 0.0f;
		int qw = 0, qh = 0, qcols = 0, qrows = 0;
		controller.GetDimensionsForDisplay(idx, &w, &h);
		controller.GetViewConeForDisplay(idx, &viewcone);
		controller.GetDefaultQuiltSettingsForDisplay(idx, &aspect, &qw, &qh, &qcols, &qrows);

		// Device name comes back as a wchar_t array; query length first.
		int nameLen = 0;
		controller.GetDeviceNameForDisplay(idx, &nameLen, nullptr);
		std::wstring name(static_cast<size_t>(nameLen > 0 ? nameLen : 0), L'\0');
		if (nameLen > 0)
			controller.GetDeviceNameForDisplay(idx, &nameLen, name.data());

		// Truncate trailing nulls and convert to narrow for logging.
		while (!name.empty() && name.back() == L'\0') name.pop_back();
		std::string narrow(name.begin(), name.end());

		LOG_INFO("  display[{}] index={} name='{}' size={}x{} viewcone={:.1f}° aspect={:.3f} quilt={}x{} ({}x{}={} views)",
			i, idx, narrow, w, h, viewcone, aspect, qw, qh, qcols, qrows, qcols * qrows);
	}

	controller.Uninitialize();
	return 0;
#endif
}

} // namespace Cave
