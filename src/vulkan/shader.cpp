#include "shader.h"
#include "deviceContext.h"
#include "../common/logger.h"

#include <fstream>

namespace Cave
{
	Shader::Shader(DeviceContext &deviceContext, vk::ShaderModule shaderModule)
		: _deviceContext{deviceContext}, _shaderModule{shaderModule}
	{
	}

	Shader::~Shader()
	{
	}

	std::vector<char> Shader::ReadFile(const std::string &filepath)
	{
		std::ifstream file(filepath, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			LOG_ERROR("Failed to load \"{}\"", filepath);
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> fileContents(fileSize);

		file.seekg(0);
		file.read(fileContents.data(), fileSize);

		file.close();

		return fileContents;
	}
}
