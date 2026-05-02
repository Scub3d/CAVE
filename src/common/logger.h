#pragma once
#include <string_view>
#include <chrono>
#include <atomic>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"

#include <vulkan/vulkan.hpp>

#include "../common/utils.h"

// Global validation error counter — defined in main.cpp
extern std::atomic<int> g_validationErrorCount;

#define LOG_FATAL(message, ...) Cave::LogHandler::Get().Log(Cave::LogType::Fatal, message, ##__VA_ARGS__);
#define LOG_ERROR(message, ...) Cave::LogHandler::Get().Log(Cave::LogType::Error, message, ##__VA_ARGS__);
#define LOG_INFO(message, ...) Cave::LogHandler::Get().Log(Cave::LogType::Info, message, ##__VA_ARGS__);
#define LOG_WARNING(message, ...) Cave::LogHandler::Get().Log(Cave::LogType::Warning, message, ##__VA_ARGS__);
#define LOG_DEBUG(message, ...) Cave::LogHandler::Get().Log(Cave::LogType::Debug, message, ##__VA_ARGS__);

namespace Cave
{

enum class LogType
{
	Fatal,
	Error,
	Info,
	Warning,
	Debug
};

class LogHandler
{
public:
	template <typename... Args>
	inline static void Log(LogType type, std::string_view message, Args&&... args)
	{
		switch (type)
		{
		case LogType::Fatal:
			spdlog::critical(fmt::runtime(message), std::forward<Args>(args)...);
			break;
		case LogType::Error:
			spdlog::error(fmt::runtime(message), std::forward<Args>(args)...);
			break;
		case LogType::Warning:
			spdlog::warn(fmt::runtime(message), std::forward<Args>(args)...);
			break;
		case LogType::Debug:
			spdlog::debug(fmt::runtime(message), std::forward<Args>(args)...);
			break;
		case LogType::Info:
			spdlog::info(fmt::runtime(message), std::forward<Args>(args)...);
			break;
		}

		if (type == LogType::Fatal)
		{
			abort();
		}
	}

	inline static LogHandler &Get()
	{
		static LogHandler handler{};
		return handler;
	}

	static const char *GetDebugMessageType(VkDebugUtilsMessageTypeFlagsEXT messageType)
	{
		switch (messageType)
		{
		case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:     return "General";
		case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:  return "Validation";
		case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT: return "Performance";
		default:                                               return "Unknown";
		}
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
		void *pUserData)
	{
		const char *messageTypeString = GetDebugMessageType(messageType);

		switch (messageSeverity)
		{
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
			LOG_DEBUG("[Vulkan/{}] {}", messageTypeString, pCallbackData->pMessage);
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
			LOG_INFO("[Vulkan/{}] {}", messageTypeString, pCallbackData->pMessage);
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
			LOG_WARNING("[Vulkan/{}] {}", messageTypeString, pCallbackData->pMessage);
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
			LOG_ERROR("[Vulkan/{}] {}", messageTypeString, pCallbackData->pMessage);
			g_validationErrorCount.fetch_add(1, std::memory_order_relaxed);
			for (uint32_t objectIndex = 0; objectIndex < pCallbackData->objectCount; objectIndex++)
			{
				LOG_ERROR("  Object[{}] handle: {:#x}", objectIndex, pCallbackData->pObjects[objectIndex].objectHandle);
			}
			break;
		default:
			LOG_WARNING("[Vulkan/Unknown severity] {}", pCallbackData->pMessage);
			break;
		}

		return VK_FALSE;
	}

	static vk::DebugUtilsMessengerEXT CreateVulkanDebugMessenger(vk::Instance &instance, vk::DispatchLoaderDynamic &dldy)
	{
		vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo = vk::DebugUtilsMessengerCreateInfoEXT(
			vk::DebugUtilsMessengerCreateFlagsEXT(),
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
			vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
			VulkanDebugCallback,
			nullptr
		);

		return instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfo, nullptr, dldy);
	};
};

} // namespace Cave
