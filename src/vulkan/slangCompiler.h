#pragma once

#include <slang.h>
#include <slang-com-ptr.h>
#include <vulkan/vulkan.hpp>
#include <string>
#include <map>

namespace Cave
{
	class DeviceContext;

	class SlangCompiler
	{
	private:
		Slang::ComPtr<slang::IGlobalSession> _globalSession;

	public:
		SlangCompiler();
		~SlangCompiler() = default;

		SlangCompiler(const SlangCompiler&) = delete;
		SlangCompiler& operator=(const SlangCompiler&) = delete;

		// Compile a .slang module to SPIR-V and create a VkShaderModule.
		// shaderDirectory: path containing .slang files (for import resolution)
		// moduleName: which module to load (e.g., "rayMarchCompute" loads rayMarchCompute.slang)
		// entryPointName: shader entry point (e.g., "computeMain")
		// defines: preprocessor macros injected before compilation
		vk::ShaderModule Compile(
			DeviceContext& deviceContext,
			const std::string& shaderDirectory,
			const std::string& moduleName,
			const std::string& entryPointName,
			const std::map<std::string, std::string>& defines = {});
	};
}
