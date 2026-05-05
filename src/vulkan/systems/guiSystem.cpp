#include "guiSystem.h"
#include "../../common/logger.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

namespace Cave
{
	GuiSystem::GuiSystem(VulkanInstance& vulkanInstance, DeviceContext& deviceContext, vk::Format swapchainImageFormat)
		: _vulkanInstance{vulkanInstance}, _deviceContext{deviceContext}
	{
		CreateImGuiDescriptorPool();

		ImGui::CreateContext();
		ImGui::StyleColorsDark();

		ImGui_ImplGlfw_InitForVulkan(_vulkanInstance.GetVulkanGLFWWindow(), true);

		vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo = vk::PipelineRenderingCreateInfo(
			0,                      // viewMask
			1,                      // colorAttachmentCount
			&swapchainImageFormat,  // pColorAttachmentFormats
			vk::Format::eUndefined, // depthAttachmentFormat
			vk::Format::eUndefined  // stencilAttachmentFormat
		);

		ImGui_ImplVulkan_InitInfo initInfo{};
		initInfo.Instance = _vulkanInstance.GetInstance();
		initInfo.PhysicalDevice = _deviceContext.GetPhysicalDevice();
		initInfo.Device = _deviceContext.GetDevice();
		initInfo.QueueFamily = _deviceContext.GetQueueFamilies().GraphicsFamily.Index.value();
		initInfo.Queue = _deviceContext.GetGraphicsQueue();
		initInfo.PipelineCache = nullptr;
		initInfo.DescriptorPool = _imguiDescriptorPool;
		initInfo.RenderPass = nullptr;
		initInfo.Subpass = 0;
		initInfo.MinImageCount = 2;
		initInfo.ImageCount = _deviceContext.GetFramesInFlight();
		initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
		initInfo.Allocator = nullptr;
		initInfo.UseDynamicRendering = true;
		initInfo.PipelineRenderingCreateInfo = *reinterpret_cast<VkPipelineRenderingCreateInfo*>(&pipelineRenderingCreateInfo);

		ImGui_ImplVulkan_Init(&initInfo);

		// Upload font textures now rather than letting ImGui do it lazily on the
		// first frame.  The lazy upload submits a one-time command buffer to the
		// graphics queue with a blocking wait, which can break the frame
		// synchronization and cause an intermittent black screen.
		ImGui_ImplVulkan_CreateFontsTexture();
		_deviceContext.GetDevice().waitIdle();

		LOG_INFO("GuiSystem initialized with ImGui {}", IMGUI_VERSION);
	}

	GuiSystem::~GuiSystem()
	{
		_deviceContext.GetDevice().waitIdle();

		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		_deviceContext.GetDevice().destroyDescriptorPool(_imguiDescriptorPool);
	}

	void GuiSystem::CreateImGuiDescriptorPool()
	{
		vk::DescriptorPoolSize poolSize = vk::DescriptorPoolSize(
			vk::DescriptorType::eCombinedImageSampler, // type
			1                                          // descriptorCount
		);

		vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo(
			vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, // flags
			1,                                                     // maxSets
			1,                                                     // poolSizeCount
			&poolSize                                              // pPoolSizes
		);

		_imguiDescriptorPool = _deviceContext.GetDevice().createDescriptorPool(descriptorPoolCreateInfo);
	}

	std::string GuiSystem::BitmaskToRuleString(uint64_t bitmask)
	{
		// Collect all set bit positions as 1-based rule numbers
		std::vector<int> setBits;
		for (int bitPosition = 0; bitPosition < 60; bitPosition++)
			if ((bitmask >> bitPosition) & 1)
				setBits.push_back(bitPosition + 1);

		if (setBits.empty())
			return "-";

		// Group consecutive runs and format as "1,3-6,9"
		std::string output;
		size_t i = 0;
		while (i < setBits.size())
		{
			size_t runStart = i;
			while (i + 1 < setBits.size() && setBits[i + 1] == setBits[i] + 1)
				i++;

			if (!output.empty()) output += ",";

			if (i - runStart >= 2)
				output += std::to_string(setBits[runStart]) + "-" + std::to_string(setBits[i]);
			else
			{
				for (size_t j = runStart; j <= i; j++)
				{
					if (j > runStart) output += ",";
					output += std::to_string(setBits[j]);
				}
			}
			i++;
		}
		return output;
	}

	uint64_t GuiSystem::RuleStringToBitmask(const std::string& ruleString)
	{
		uint64_t bitmask = 0;

		std::stringstream stream(ruleString);
		std::string token;

		while (std::getline(stream, token, ','))
		{
			// Trim whitespace
			token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
			if (token.empty()) continue;

			size_t dashPosition = token.find('-');
			if (dashPosition != std::string::npos)
			{
				int rangeStart = std::stoi(token.substr(0, dashPosition));
				int rangeEnd = std::stoi(token.substr(dashPosition + 1));
				for (int i = rangeStart; i <= rangeEnd && i <= 60; i++)
				{
					if (i >= 1) bitmask |= (1ULL << (i - 1));
				}
			}
			else
			{
				int value = std::stoi(token);
				if (value >= 1 && value <= 60)
				{
					bitmask |= (1ULL << (value - 1));
				}
			}
		}

		return bitmask;
	}

	void GuiSystem::BeginModeFrame()
	{
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(350, 0), ImGuiCond_FirstUseEver);

		ImGui::Begin("Settings");
		if (ImGui::Button("Back to Mode Selection"))
			_exitModeRequested = true;
		ImGui::Separator();
	}

	void GuiSystem::EndModeFrame()
	{
		ImGui::End(); // Settings window

		// --- FPS Overlay (always visible) ---
		ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 10, 10), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
		ImGui::SetNextWindowBgAlpha(0.3f);
		if (ImGui::Begin("Info", nullptr,
			ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
			ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
		{
			ImGui::Text("%.1f FPS (%.2f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
		}
		ImGui::End();

		ImGui::Render();
	}

	void GuiSystem::BuildIdleFrame()
	{
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		BuildModeSelector();

		// --- FPS Overlay ---
		ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 10, 10), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
		ImGui::SetNextWindowBgAlpha(0.3f);
		if (ImGui::Begin("Info", nullptr,
			ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
			ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
		{
			ImGui::Text("%.1f FPS (%.2f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
		}
		ImGui::End();

		ImGui::Render();
	}

	void GuiSystem::RecordDrawCommands(vk::CommandBuffer commandBuffer)
	{
		ImDrawData* drawData = ImGui::GetDrawData();
		if (drawData)
		{
			ImGui_ImplVulkan_RenderDrawData(drawData, commandBuffer);
		}
	}

	bool GuiSystem::ConsumeResetRequest()
	{
		bool requested = _resetRequested;
		_resetRequested = false;
		return requested;
	}

	bool GuiSystem::ConsumeStartRequest()
	{
		bool requested = _startRequested;
		_startRequested = false;
		return requested;
	}

	bool GuiSystem::ConsumeStopRequest()
	{
		bool requested = _stopRequested;
		_stopRequested = false;
		return requested;
	}

	bool GuiSystem::ConsumePauseRequest()
	{
		bool requested = _pauseRequested;
		_pauseRequested = false;
		return requested;
	}

	bool GuiSystem::ConsumeStepRequest()
	{
		bool requested = _stepRequested;
		_stepRequested = false;
		return requested;
	}

	bool GuiSystem::ConsumeResumeRequest()
	{
		bool requested = _resumeRequested;
		_resumeRequested = false;
		return requested;
	}

	bool GuiSystem::ConsumeEncodeRequest()
	{
		bool requested = _encodeRequested;
		_encodeRequested = false;
		return requested;
	}

	bool GuiSystem::ConsumeCancelEncodeRequest()
	{
		bool requested = _cancelEncodeRequested;
		_cancelEncodeRequested = false;
		return requested;
	}

	ApplicationMode GuiSystem::ConsumeEnterModeRequest()
	{
		ApplicationMode mode = _enterModeRequested;
		_enterModeRequested = ApplicationMode::None;
		return mode;
	}

	bool GuiSystem::ConsumeExitModeRequest()
	{
		bool requested = _exitModeRequested;
		_exitModeRequested = false;
		return requested;
	}

	void GuiSystem::BuildModeSelector()
	{
		ImVec2 displaySize = ImGui::GetIO().DisplaySize;
		ImVec2 windowSize(300, 200);
		ImGui::SetNextWindowPos(ImVec2((displaySize.x - windowSize.x) * 0.5f, (displaySize.y - windowSize.y) * 0.5f), ImGuiCond_Always);
		ImGui::SetNextWindowSize(windowSize, ImGuiCond_Always);

		if (ImGui::Begin("Cave - Mode Selection", nullptr,
			ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse))
		{
			float buttonWidth = ImGui::GetContentRegionAvail().x;
			float buttonHeight = 40.0f;

			if (ImGui::Button("Rendering Mode", ImVec2(buttonWidth, buttonHeight)))
				_enterModeRequested = ApplicationMode::Rendering;

			ImGui::Spacing();

			if (ImGui::Button("Search Mode", ImVec2(buttonWidth, buttonHeight)))
				_enterModeRequested = ApplicationMode::Search;

			ImGui::Spacing();

			if (ImGui::Button("Video Encoding Mode", ImVec2(buttonWidth, buttonHeight)))
				_enterModeRequested = ApplicationMode::VideoEncoding;

			ImGui::Spacing();

			if (ImGui::Button("Looking Glass Mode", ImVec2(buttonWidth, buttonHeight)))
				_enterModeRequested = ApplicationMode::LookingGlass;
		}
		ImGui::End();
	}

	void GuiSystem::BuildLookingGlassPanel()
	{
		// Compact panel docked top-right when LG mode is active. Sliders are
		// reactive — LookingGlassMode reads these every frame and forwards
		// viewcone/zoom into the per-view camera offset push constant. Display
		// index changes require a mode re-enter (Bridge owns the GL window).
		ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(320, 200), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Looking Glass", nullptr, ImGuiWindowFlags_None))
		{
			ImGui::Text("Display index (Bridge-reported)");
			ImGui::DragInt("##lgDisplay", &_state.lookingGlassDisplayIndex, 0.1f, 0, 7);
			ImGui::TextDisabled("(re-enter mode to apply)");
			ImGui::Separator();

			ImGui::Text("View cone (degrees)");
			ImGui::SliderFloat("##lgViewcone", &_state.lookingGlassViewconeDegrees, 0.0f, 60.0f, "%.1f");

			ImGui::Text("Zoom");
			ImGui::SliderFloat("##lgZoom", &_state.lookingGlassZoom, 0.5f, 2.0f, "%.2f");

			ImGui::Text("View count");
			const char* viewCountLabels[] = { "24", "32", "45", "48" };
			const int viewCountValues[] = { 24, 32, 45, 48 };
			int currentIdx = 3;
			for (int i = 0; i < 4; i++)
				if (viewCountValues[i] == _state.lookingGlassViewCount) currentIdx = i;
			if (ImGui::Combo("##lgViewCount", &currentIdx, viewCountLabels, 4))
				_state.lookingGlassViewCount = viewCountValues[currentIdx];

			ImGui::TextDisabled("Bridge interpolates if < 48 views supplied.");
		}
		ImGui::End();
	}

	SimulationParameters GuiSystem::BuildSimulationParameters() const
	{
		uint64_t birthBitmask = RuleStringToBitmask(_state.birthRulesText);
		uint64_t survivalBitmask = RuleStringToBitmask(_state.survivalRulesText);

		uint64_t maxStateEncoded = Simulation::EncodeMaxCellState(static_cast<uint64_t>(_state.maxCellState));

		uint64_t neighborhoodFlags = 0;
		if (_state.faceNeighbors) neighborhoodFlags |= Simulation::FACE_NEIGHBORS_MASK;
		if (_state.edgeNeighbors) neighborhoodFlags |= Simulation::EDGE_NEIGHBORS_MASK;
		if (_state.cornerNeighbors) neighborhoodFlags |= Simulation::CORNER_NEIGHBORS_MASK;
		if (_state.wrapAtBoundary) neighborhoodFlags |= Simulation::WRAP_NEIGHBORS_MASK;

		SimulationParameters parameters{};
		parameters.birthAndMaxCellStateRules = (maxStateEncoded << Simulation::BIT_SHIFT) | birthBitmask;
		parameters.survivalAndNeighborhoodRules = (neighborhoodFlags << Simulation::BIT_SHIFT) | survivalBitmask;
		parameters.shapeAndGridConfiguration = 0; // Cube = shape ID 0
		return parameters;
	}

	ColorRules GuiSystem::BuildColorRules() const
	{
		return ColorRules{
			glm::vec4(_state.aliveColor[0], _state.aliveColor[1], _state.aliveColor[2], 1.0f),
			glm::vec4(_state.deadColor[0], _state.deadColor[1], _state.deadColor[2], 1.0f)
		};
	}
}
