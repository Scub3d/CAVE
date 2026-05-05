#pragma once

#include "applicationModeHandler.h"
#include "../vulkan/glInterop.h"
#include "../vulkan/externalSemaphore.h"
#include "../vulkan/systems/quiltRenderSystem.h"
#include "../simulation/simulationContext.h"

#include <memory>

namespace Cave
{
	// Real-time hologram output to a Looking Glass Portrait via the Bridge SDK.
	//
	// Pipeline per frame:
	//   1. ComputeSystem ticks the cellular automata.
	//   2. QuiltRenderSystem produces a 3360×3360 RGBA8 quilt of 48 views.
	//   3. Vulkan signals a binary external semaphore on quilt-write completion.
	//   4. GL waits on the semaphore, calls Bridge's draw_interop_quilt_texture_gl
	//      to push the quilt to the Portrait.
	//   5. GL signals back; Vulkan waits before reusing the quilt next frame.
	//
	// Without lib/bridge/ vendored at compile time (LOOKING_GLASS_HAS_BRIDGE
	// undefined), Enter() logs a clear error pointing the user at the SDK
	// install page and OnFrame returns false.
	class LookingGlassMode : public ApplicationModeHandler
	{
	public:
		LookingGlassMode() = default;
		~LookingGlassMode() override = default;

		void Enter(ModeServices& services) override;
		bool OnFrame(ModeServices& services, float deltaTimeInSeconds) override;
		void Exit(ModeServices& services) override;

		uint32_t GetSimulationTickCount() const override { return _tickCount; }

	private:
		// Simulation pipeline (single-GPU). Initialized in Enter() with sensible
		// defaults pulled from GuiState (grid size, neighborhood, rules).
		SimulationContext _simulationContext;

		// Quilt render path (Vulkan side).
		std::unique_ptr<QuiltRenderSystem>       _quiltRenderSystem;

		// GL interop primitives (only needed when Bridge is present).
		std::unique_ptr<GlInteropContext>        _glContext;
		std::unique_ptr<ExternalBinarySemaphore> _vkSignalGlSem;
		std::unique_ptr<ExternalBinarySemaphore> _glSignalVkSem;
		uint32_t _glQuiltTexture = 0;
		uint32_t _glVkToGlSemaphore = 0;
		uint32_t _glGlToVkSemaphore = 0;

		uint32_t _tickCount = 0;
		bool _bridgeInitialized = false;
		bool _fatalError = false;

		// Calibrated values pulled from Bridge during Enter() — used instead of
		// the GuiState defaults so the per-view shift and Bridge submit match
		// the actual Portrait's optical calibration.
		float _bridgeViewconeRadians = 0.0f; // 0 means "no calibration; use GuiState"
		float _bridgeQuiltAspect = 0.0f;
		double _lastFpsLogTime = 0.0;
		uint32_t _lastFpsLogTickCount = 0;

		// Bridge window handle (cast to WINDOW_HANDLE at the Bridge call site so
		// this header doesn't need bridge.h transitively). WINDOW_HANDLE is a
		// typedef for `unsigned long` — we widen it to uintptr_t for storage.
		void* _bridgeWindow = nullptr;
		// Bridge Controller. Defined inline in bridge.h, so we PIMPL via void*
		// to keep <Windows.h> contained to the cpp.
		void* _bridgeController = nullptr;

		// Per-frame helpers
		void InitializeSimulation(ModeServices& services);
	};

} // namespace Cave
