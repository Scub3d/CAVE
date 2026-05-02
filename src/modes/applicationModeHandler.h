#pragma once

#include <memory>
#include <cstdint>

namespace Cave
{
	class VulkanInstance;
	class DeviceContext;
	class PresentSystem;
	class GuiSystem;
	class QueryManager;
	struct StartupConfig;

	struct ModeServices
	{
		VulkanInstance& vulkanInstance;
		DeviceContext& deviceContext;
		DeviceContext* secondaryDeviceContext = nullptr; // nullptr when single-GPU
		PresentSystem& presentSystem;
		GuiSystem& guiSystem;
		std::shared_ptr<QueryManager> queryManager;
		const StartupConfig* startupConfig;
		bool headless;
		uint32_t framesInFlight;
	};

	class ApplicationModeHandler
	{
	public:
		virtual ~ApplicationModeHandler() = default;

		// Called once when the mode is entered.
		virtual void Enter(ModeServices& services) = 0;

		// Called once per frame. Returns false to signal application exit.
		virtual bool OnFrame(ModeServices& services, float deltaTimeInSeconds) = 0;

		// Called once when exiting the mode. Device waitIdle is called by Engine before this.
		virtual void Exit(ModeServices& services) = 0;

		// Returns simulation tick count for --frames auto-exit.
		virtual uint32_t GetSimulationTickCount() const { return 0; }
	};

} // namespace Cave
