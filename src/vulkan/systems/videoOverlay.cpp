#include "videoOverlay.h"

#include "../vulkanInstance.h"
#include "../../common/logger.h"

#include "imgui.h"
#include "imgui_impl_vulkan.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace Cave
{
	namespace
	{
		// RAII helper to temporarily switch the active ImGui context, restoring the
		// previous one on scope exit. ImGui is a single-context-at-a-time API and the
		// main GuiSystem may be mid-frame, so we never want to leave its context unset.
		struct ScopedImGuiContext
		{
			ImGuiContext* previous;
			explicit ScopedImGuiContext(ImGuiContext* next)
				: previous(ImGui::GetCurrentContext())
			{
				ImGui::SetCurrentContext(next);
			}
			~ScopedImGuiContext()
			{
				ImGui::SetCurrentContext(previous);
			}
		};
	}

	VideoOverlay::VideoOverlay(VulkanInstance& vulkanInstance, DeviceContext& deviceContext,
		vk::Format colorFormat, vk::Format depthFormat, vk::Extent2D extent, uint32_t framesInFlight)
		: _deviceContext{deviceContext}, _extent{extent}
	{
		// Descriptor pool sized for ImGui Vulkan backend (one combined-image-sampler for
		// the font atlas — same shape as GuiSystem::CreateImGuiDescriptorPool).
		vk::DescriptorPoolSize poolSize = vk::DescriptorPoolSize(
			vk::DescriptorType::eCombinedImageSampler,    // type
			1                                              // descriptorCount
		);
		vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo(
			vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,    // flags
			1,                                                        // maxSets
			1,                                                        // poolSizeCount
			&poolSize                                                 // pPoolSizes
		);
		_descriptorPool = _deviceContext.GetDevice().createDescriptorPool(descriptorPoolCreateInfo);

		// Save outer context (GuiSystem's), create + activate ours.
		ImGuiContext* outerContext = ImGui::GetCurrentContext();
		_overlayContext = ImGui::CreateContext();
		ImGui::SetCurrentContext(_overlayContext);
		ImGui::StyleColorsDark();

		// Tell ImGui what display we're rendering to so window positioning works.
		ImGui::GetIO().DisplaySize = ImVec2(static_cast<float>(_extent.width), static_cast<float>(_extent.height));
		ImGui::GetIO().IniFilename = nullptr;  // no imgui.ini for overlay

		// Scale UI relative to video height so the overlay stays legible at 4K and beyond.
		// Reference baseline is 720p (1.0x). 1080p ~= 1.5x, 1440p ~= 2.0x, 4K ~= 3.0x.
		// Below 720p we keep the default size so the overlay doesn't shrink below readable.
		float fontScale = std::max(1.0f, static_cast<float>(_extent.height) / 720.0f);
		ImGui::GetIO().FontGlobalScale = fontScale;
		// ScaleAllSizes scales window padding, item spacing, frame padding etc. so the layout
		// stays proportional to the (now larger) text instead of cramped against scaled-up font.
		ImGui::GetStyle().ScaleAllSizes(fontScale);

		vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo = vk::PipelineRenderingCreateInfo(
			0,                          // viewMask
			1,                          // colorAttachmentCount
			&colorFormat,               // pColorAttachmentFormats
			depthFormat,                // depthAttachmentFormat
			vk::Format::eUndefined      // stencilAttachmentFormat
		);

		ImGui_ImplVulkan_InitInfo initInfo{};
		initInfo.Instance = vulkanInstance.GetInstance();
		initInfo.PhysicalDevice = _deviceContext.GetPhysicalDevice();
		initInfo.Device = _deviceContext.GetDevice();
		initInfo.QueueFamily = _deviceContext.GetQueueFamilies().GraphicsFamily.Index.value();
		initInfo.Queue = _deviceContext.GetGraphicsQueue();
		initInfo.PipelineCache = nullptr;
		initInfo.DescriptorPool = _descriptorPool;
		initInfo.RenderPass = nullptr;
		initInfo.Subpass = 0;
		initInfo.MinImageCount = 2;
		initInfo.ImageCount = framesInFlight;
		initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
		initInfo.Allocator = nullptr;
		initInfo.UseDynamicRendering = true;
		initInfo.PipelineRenderingCreateInfo = *reinterpret_cast<VkPipelineRenderingCreateInfo*>(&pipelineRenderingCreateInfo);

		ImGui_ImplVulkan_Init(&initInfo);
		ImGui_ImplVulkan_CreateFontsTexture();
		_deviceContext.GetDevice().waitIdle();

		// Prime ImGui's per-frame state by running one full NewFrame/Build/Render cycle.
		// Without rendering any widgets, the first real encoded frame produces no overlay at
		// all (per-frame ImGui buffers / font binding aren't initialized lazily by an empty
		// frame). BuildOverlayWindow uses a fixed window width (see BuildOverlayWindow) so
		// the warmup doesn't poison the auto-size for frame 0.
		ImGui_ImplVulkan_NewFrame();
		ImGui::NewFrame();
		BuildOverlayWindow();
		ImGui::Render();

		// Restore main context so subsequent main-GUI code is unaffected.
		ImGui::SetCurrentContext(outerContext);

		LOG_INFO("VideoOverlay initialized ({}x{}, format={})",
			_extent.width, _extent.height, vk::to_string(colorFormat));
	}

	VideoOverlay::~VideoOverlay()
	{
		_deviceContext.GetDevice().waitIdle();

		ImGuiContext* outerContext = ImGui::GetCurrentContext();
		ImGui::SetCurrentContext(_overlayContext);
		ImGui_ImplVulkan_Shutdown();
		ImGui::DestroyContext(_overlayContext);
		// Restore — but only if outer wasn't us, in which case set to null.
		ImGui::SetCurrentContext(outerContext == _overlayContext ? nullptr : outerContext);

		_deviceContext.GetDevice().destroyDescriptorPool(_descriptorPool);
	}

	void VideoOverlay::BuildOverlayWindow()
	{
		const float fontScale = std::max(1.0f, static_cast<float>(_extent.height) / 720.0f);
		// Margin from viewport edge — scaled with font so it stays visually consistent.
		const float padding = 10.0f * fontScale;
		ImVec2 windowPos;
		ImVec2 pivot;
		if (_info.position == OverlayPosition::TopRight)
		{
			windowPos = ImVec2(static_cast<float>(_extent.width) - padding, padding);
			pivot = ImVec2(1.0f, 0.0f);
		}
		else
		{
			windowPos = ImVec2(padding, padding);
			pivot = ImVec2(0.0f, 0.0f);
		}

		// Pre-format every line we'll render. Width is computed from the actual rendered
		// text so the box hugs the longest line — and is used both for SetNextWindowSize
		// (driving the window width) and for the TextUnformatted calls inside Begin (so we
		// only format each line once).
		char buffer[256];
		std::vector<std::string> lines;
		if (_info.gridX == _info.gridY && _info.gridY == _info.gridZ)
			snprintf(buffer, sizeof(buffer), "Grid: %u^3", _info.gridX);
		else
			snprintf(buffer, sizeof(buffer), "Grid: %u x %u x %u", _info.gridX, _info.gridY, _info.gridZ);
		lines.emplace_back(buffer);

		if (_info.spawnX == _info.spawnY && _info.spawnY == _info.spawnZ)
			snprintf(buffer, sizeof(buffer), "Spawn: %u^3", _info.spawnX);
		else
			snprintf(buffer, sizeof(buffer), "Spawn: %u x %u x %u", _info.spawnX, _info.spawnY, _info.spawnZ);
		lines.emplace_back(buffer);

		snprintf(buffer, sizeof(buffer), "Shape: %s", _info.shapeName.c_str());
		lines.emplace_back(buffer);

		// "FEC W" style neighborhood string from per-axis flags.
		char neighborhoodBuffer[8] = {};
		int neighborhoodCharIndex = 0;
		if (_info.faceNeighbors)   neighborhoodBuffer[neighborhoodCharIndex++] = 'F';
		if (_info.edgeNeighbors)   neighborhoodBuffer[neighborhoodCharIndex++] = 'E';
		if (_info.cornerNeighbors) neighborhoodBuffer[neighborhoodCharIndex++] = 'C';
		if (_info.wrapAtBoundary)  neighborhoodBuffer[neighborhoodCharIndex++] = 'W';
		snprintf(buffer, sizeof(buffer), "Neighborhood: %s", neighborhoodBuffer);
		lines.emplace_back(buffer);

		snprintf(buffer, sizeof(buffer), "Birth: %s", _info.birthRules.c_str());
		lines.emplace_back(buffer);
		snprintf(buffer, sizeof(buffer), "Survival: %s", _info.survivalRules.c_str());
		lines.emplace_back(buffer);
		snprintf(buffer, sizeof(buffer), "Max Cell State: %u", _info.maxCellState);
		lines.emplace_back(buffer);
		// Separator (rendered as a horizontal line, not a text line).
		const size_t separatorIndex = lines.size();
		snprintf(buffer, sizeof(buffer), "Tick: %u / %u", _info.currentTick, _info.maxTicks);
		lines.emplace_back(buffer);
		snprintf(buffer, sizeof(buffer), "Ticks/sec: %u", _info.ticksPerSecond);
		lines.emplace_back(buffer);

		// Width = longest text line + the window's left+right padding so text never gets clipped.
		float maxTextWidth = 0.0f;
		for (const auto& line : lines)
			maxTextWidth = std::max(maxTextWidth, ImGui::CalcTextSize(line.c_str()).x);
		float windowWidth = maxTextWidth + ImGui::GetStyle().WindowPadding.x * 2.0f;

		ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always, pivot);
		ImGui::SetNextWindowSize(ImVec2(windowWidth, 0.0f), ImGuiCond_Always);
		ImGui::SetNextWindowBgAlpha(0.55f);
		const ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
			ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
			ImGuiWindowFlags_NoDecoration;

		if (ImGui::Begin("##VideoOverlay", nullptr, flags))
		{
			for (size_t lineIndex = 0; lineIndex < lines.size(); lineIndex++)
			{
				if (lineIndex == separatorIndex)
					ImGui::Separator();
				ImGui::TextUnformatted(lines[lineIndex].c_str());
			}
		}
		ImGui::End();
	}

	void VideoOverlay::RecordDrawCommands(vk::CommandBuffer commandBuffer)
	{
		ScopedImGuiContext guard(_overlayContext);

		// Update display size in case the encoder was resized between init and now.
		ImGui::GetIO().DisplaySize = ImVec2(static_cast<float>(_extent.width), static_cast<float>(_extent.height));

		ImGui_ImplVulkan_NewFrame();
		ImGui::NewFrame();

		BuildOverlayWindow();

		ImGui::Render();
		ImDrawData* drawData = ImGui::GetDrawData();
		if (drawData != nullptr)
		{
			ImGui_ImplVulkan_RenderDrawData(drawData, commandBuffer);
		}
	}
}
