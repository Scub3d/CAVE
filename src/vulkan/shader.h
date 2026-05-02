#pragma once

#include <vulkan/vulkan.hpp>
#include <string>
#include <vector>

namespace Cave
{

class DeviceContext;

class Shader
	{
	private:
		DeviceContext &_deviceContext;
		vk::ShaderModule _shaderModule{};

	public:
		Shader(DeviceContext &deviceContext, vk::ShaderModule shaderModule);
		~Shader();

		vk::ShaderModule GetShaderModule() const { return _shaderModule; }

		// Read a precompiled SPIR-V file into a byte buffer. Used by the video encoder
		// for the RGB→YCbCr shader, which is built offline by shaders/compileShaders.bat
		// and shipped as a .spv file in shaders/compiled/.
		static std::vector<char> ReadFile(const std::string &filepath);
	};
}
