#include "slangCompiler.h"
#include "deviceContext.h"
#include "../common/logger.h"
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

namespace Cave
{
	SlangCompiler::SlangCompiler()
	{
		SlangResult result = slang::createGlobalSession(_globalSession.writeRef());
		if (SLANG_FAILED(result))
		{
			LOG_FATAL("Failed to create Slang global session");
			throw std::runtime_error("Failed to create Slang global session");
		}
		LOG_INFO("Slang compiler initialized (version: {})", _globalSession->getBuildTagString());
	}

	vk::ShaderModule SlangCompiler::Compile(
		DeviceContext& deviceContext,
		const std::string& shaderDirectory,
		const std::string& moduleName,
		const std::string& entryPointName,
		const std::map<std::string, std::string>& defines)
	{
		// Build preprocessor macro descriptors
		std::vector<slang::PreprocessorMacroDesc> macros;
		// Keep string storage alive until session creation
		std::vector<std::pair<std::string, std::string>> macroStorage(defines.begin(), defines.end());
		for (auto& [name, value] : macroStorage)
		{
			slang::PreprocessorMacroDesc macro;
			macro.name = name.c_str();
			macro.value = value.c_str();
			macros.push_back(macro);
		}

		// Configure compilation target: SPIR-V 1.5
		slang::TargetDesc targetDesc = {};
		targetDesc.structureSize = sizeof(slang::TargetDesc);
		targetDesc.format = SLANG_SPIRV;
		targetDesc.profile = _globalSession->findProfile("spirv_1_5");

		// Set up search paths for import resolution. We register the root
		// shaderDirectory plus every subdirectory under it so that `import foo;`
		// resolves regardless of which subfolder `foo.slang` happens to live in.
		std::vector<std::string> searchPathStorage;
		searchPathStorage.push_back(shaderDirectory);
		{
			std::error_code ec;
			std::filesystem::recursive_directory_iterator it(shaderDirectory, ec);
			if (!ec)
			{
				for (const auto& entry : it)
				{
					if (entry.is_directory())
						searchPathStorage.push_back(entry.path().string());
				}
			}
		}
		std::vector<const char*> searchPaths;
		searchPaths.reserve(searchPathStorage.size());
		for (const auto& path : searchPathStorage)
			searchPaths.push_back(path.c_str());

		// Create session
		slang::SessionDesc sessionDesc = {};
		sessionDesc.structureSize = sizeof(slang::SessionDesc);
		sessionDesc.targets = &targetDesc;
		sessionDesc.targetCount = 1;
		sessionDesc.searchPaths = searchPaths.data();
		sessionDesc.searchPathCount = static_cast<SlangInt>(searchPaths.size());
		sessionDesc.preprocessorMacros = macros.data();
		sessionDesc.preprocessorMacroCount = static_cast<SlangInt>(macros.size());
		// Column-major matrix layout matches GLSL/Vulkan conventions and the column-major
		// matrices uploaded from glm. Slang's default is row-major (HLSL convention) which
		// produces transposed matrices in shaders and breaks rasterization.
		sessionDesc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;

		Slang::ComPtr<slang::ISession> session;
		SlangResult result = _globalSession->createSession(sessionDesc, session.writeRef());
		if (SLANG_FAILED(result))
		{
			LOG_ERROR("Failed to create Slang session for module '{}'", moduleName);
			return {};
		}

		// Load module from file
		Slang::ComPtr<slang::IBlob> diagnosticsBlob;
		slang::IModule* module = session->loadModule(moduleName.c_str(), diagnosticsBlob.writeRef());
		if (!module)
		{
			const char* diagnostics = diagnosticsBlob
				? static_cast<const char*>(diagnosticsBlob->getBufferPointer())
				: "unknown error";
			LOG_ERROR("Slang module '{}' load failed: {}", moduleName, diagnostics);
			return {};
		}

		// Find entry point
		Slang::ComPtr<slang::IEntryPoint> entryPoint;
		result = module->findEntryPointByName(entryPointName.c_str(), entryPoint.writeRef());
		if (SLANG_FAILED(result) || !entryPoint)
		{
			LOG_ERROR("Slang entry point '{}' not found in module '{}'", entryPointName, moduleName);
			return {};
		}

		// Compose program (module + entry point)
		slang::IComponentType* components[] = {module, entryPoint.get()};
		Slang::ComPtr<slang::IComponentType> composedProgram;
		result = session->createCompositeComponentType(
			components, 2, composedProgram.writeRef(), diagnosticsBlob.writeRef());
		if (SLANG_FAILED(result))
		{
			const char* diagnostics = diagnosticsBlob
				? static_cast<const char*>(diagnosticsBlob->getBufferPointer())
				: "unknown error";
			LOG_ERROR("Slang composition failed for '{}': {}", moduleName, diagnostics);
			return {};
		}

		// Link
		Slang::ComPtr<slang::IComponentType> linkedProgram;
		result = composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef());
		if (SLANG_FAILED(result))
		{
			const char* diagnostics = diagnosticsBlob
				? static_cast<const char*>(diagnosticsBlob->getBufferPointer())
				: "unknown error";
			LOG_ERROR("Slang link failed for '{}': {}", moduleName, diagnostics);
			return {};
		}

		// Get SPIR-V code
		Slang::ComPtr<slang::IBlob> spirvBlob;
		result = linkedProgram->getEntryPointCode(0, 0, spirvBlob.writeRef(), diagnosticsBlob.writeRef());
		if (SLANG_FAILED(result) || !spirvBlob)
		{
			const char* diagnostics = diagnosticsBlob
				? static_cast<const char*>(diagnosticsBlob->getBufferPointer())
				: "unknown error";
			LOG_ERROR("Slang SPIR-V codegen failed for '{}': {}", moduleName, diagnostics);
			return {};
		}

		LOG_INFO("Slang compiled '{}::{}' to {} bytes of SPIR-V",
			moduleName, entryPointName, spirvBlob->getBufferSize());

		// Dump debug SPIR-V
		{
			std::string debugPath = "data/debug_" + moduleName + ".comp.spv";
			std::ofstream debugFile(debugPath, std::ios::binary);
			if (debugFile.is_open())
				debugFile.write(static_cast<const char*>(spirvBlob->getBufferPointer()), spirvBlob->getBufferSize());
		}

		// Create VkShaderModule
		vk::ShaderModuleCreateInfo shaderModuleCreateInfo = vk::ShaderModuleCreateInfo(
			{},																		// flags
			spirvBlob->getBufferSize(),												// codeSize
			static_cast<const uint32_t*>(spirvBlob->getBufferPointer())				// pCode
		);

		try
		{
			return deviceContext.GetDevice().createShaderModule(shaderModuleCreateInfo);
		}
		catch (vk::SystemError& error)
		{
			LOG_ERROR("Failed to create VkShaderModule from Slang SPIR-V: {}", error.what());
			return {};
		}
	}
}
